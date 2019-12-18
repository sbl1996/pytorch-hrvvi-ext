import inspect
import logging
import time
from collections import defaultdict

from ignite._utils import _to_hours_mins_secs
from ignite.engine import Events, State


class Engine(object):
    """Runs a given process_function over each batch of a dataset, emitting events as it goes.

    Args:
        process_function (callable): A function receiving a handle to the engine and the current batch
            in each iteration, and returns data to be stored in the engine's state.

    Example usage:

    .. code-block:: python

        def train_and_store_loss(engine, batch):
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            return loss.item()

        engine = Engine(train_and_store_loss)
        engine.run(data_loader)

        # Loss value is now stored in `engine.state.output`.

    """
    def __init__(self, process_function):
        self._event_handlers = defaultdict(list)
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())
        self._process_function = process_function
        self.should_terminate = False
        self.state = None
        self._allowed_events = []

        self.register_events(*Events)

        if self._process_function is None:
            raise ValueError("Engine must be given a processing function in order to run.")

        self._check_signature(process_function, 'process_function', None)

    def register_events(self, *event_names):
        """Add events that can be fired.

        Registering an event will let the user fire these events at any point.
        This opens the door to make the :meth:`~ignite.engine.Engine.run` loop even more
        configurable.

        By default, the events from :class:`~ignite.engine.Events` are registerd.

        Args:
            *event_names: An object (ideally a string or int) to define the
                name of the event being supported.

        Example usage:

        .. code-block:: python

            from enum import Enum

            class Custom_Events(Enum):
                FOO_EVENT = "foo_event"
                BAR_EVENT = "bar_event"

            engine = Engine(process_function)
            engine.register_events(*Custom_Events)

        """
        for name in event_names:
            self._allowed_events.append(name)

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        """Add an event handler to be executed when the specified event is fired.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.Events`
                or any `event_name` added by :meth:`~ignite.engine.Engine.register_events`.
            handler (callable): the callable event handler that should be invoked
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.

        Notes:
              The handler function's first argument will be `self`, the :class:`~ignite.engine.Engine` object it
              was bound to.

              Note that other arguments can be passed to the handler in addition to the `*args` and  `**kwargs`
              passed here, for example during :attr:`~ignite.engine.Events.EXCEPTION_RAISED`.

        Example usage:

        .. code-block:: python

            engine = Engine(process_function)

            def print_epoch(engine):
                print("Epoch: {}".format(engine.state.epoch))

            engine.add_event_handler(Events.EPOCH_COMPLETED, print_epoch)

        """
        if event_name not in self._allowed_events:
            self._logger.error("attempt to add event handler to an invalid event %s.", event_name)
            raise ValueError("Event {} is not a valid event for this Engine.".format(event_name))

        event_args = (Exception(), ) if event_name == Events.EXCEPTION_RAISED else ()
        self._check_signature(handler, 'handler', *(event_args + args), **kwargs)

        self._event_handlers[event_name].append((handler, args, kwargs))
        self._logger.debug("added handler for event %s.", event_name)

    def has_event_handler(self, handler, event_name=None):
        """Check if the specified event has the specified handler.

        Args:
            handler (callable): the callable event handler.
            event_name: The event the handler attached to. Set this
                to ``None`` to search all events.
        """
        if event_name is not None:
            if event_name not in self._event_handlers:
                return False
            events = [event_name]
        else:
            events = self._event_handlers
        for e in events:
            for h, _, _ in self._event_handlers[e]:
                if h == handler:
                    return True
        return False

    def remove_event_handler(self, handler, event_name):
        """Remove event handler `handler` from registered handlers of the engine

        Args:
            handler (callable): the callable event handler that should be removed
            event_name: The event the handler attached to.

        """
        if event_name not in self._event_handlers:
            raise ValueError("Input event name '{}' does not exist".format(event_name))

        new_event_handlers = [(h, args, kwargs) for h, args, kwargs in self._event_handlers[event_name]
                              if h != handler]
        if len(new_event_handlers) == len(self._event_handlers[event_name]):
            raise ValueError("Input handler '{}' is not found among registered event handlers".format(handler))
        self._event_handlers[event_name] = new_event_handlers

    def _check_signature(self, fn, fn_description, *args, **kwargs):
        exception_msg = None

        signature = inspect.signature(fn)
        try:
            signature.bind(self, *args, **kwargs)
        except TypeError as exc:
            fn_params = list(signature.parameters)
            exception_msg = str(exc)

        if exception_msg:
            passed_params = [self] + list(args) + list(kwargs)
            raise ValueError("Error adding {} '{}': "
                             "takes parameters {} but will be called with {} "
                             "({}).".format(
                                 fn, fn_description, fn_params, passed_params, exception_msg))

    def on(self, event_name, *args, **kwargs):
        """Decorator shortcut for add_event_handler.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.Events` or
                any `event_name` added by :meth:`~ignite.engine.Engine.register_events`.
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.

        """
        def decorator(f):
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f
        return decorator

    def _fire_event(self, event_name, *event_args, **event_kwargs):
        """Execute all the handlers associated with given event.

        This method executes all handlers associated with the event
        `event_name`. Optional positional and keyword arguments can be used to
        pass arguments to **all** handlers added with this event. These
        aguments updates arguments passed using :meth:`~ignite.engine.Engine.add_event_handler`.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.Events` or any `event_name` added by
                :meth:`~ignite.engine.Engine.register_events`.
            *event_args: optional args to be passed to all handlers.
            **event_kwargs: optional keyword args to be passed to all handlers.

        """
        if event_name in self._allowed_events:
            self._logger.debug("firing handlers for event %s ", event_name)
            for func, args, kwargs in self._event_handlers[event_name]:
                kwargs.update(event_kwargs)
                func(self, *(event_args + args), **kwargs)

    def fire_event(self, event_name):
        """Execute all the handlers associated with given event.

        This method executes all handlers associated with the event
        `event_name`. This is the method used in :meth:`~ignite.engine.Engine.run` to call the
        core events found in :class:`~ignite.engine.Events`.

        Custom events can be fired if they have been registered before with
        :meth:`~ignite.engine.Engine.register_events`. The engine `state` attribute should be used
        to exchange "dynamic" data among `process_function` and handlers.

        This method is called automatically for core events. If no custom
        events are used in the engine, there is no need for the user to call
        the method.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.Events` or any `event_name` added by
                :meth:`~ignite.engine.Engine.register_events`.

        """
        return self._fire_event(event_name)

    def terminate(self):
        """Sends terminate signal to the engine, so that it terminates completely the run after the current iteration.
        """
        self._logger.info("Terminate signaled. Engine will stop after current iteration is finished.")
        self.should_terminate = True

    def _run_once_on_dataset(self, max_iters):
        start_time = time.time()

        try:
            for batch in self.state.dataloader:
                self.state.batch = batch
                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)
                self.state.output = self._process_function(self, batch)
                self._fire_event(Events.ITERATION_COMPLETED)
                if self.should_terminate or self.state.iteration >= max_iters:
                    break

        except BaseException as e:
            self._logger.error("Current run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)

        return hours, mins, secs

    def _handle_exception(self, e):
        if Events.EXCEPTION_RAISED in self._event_handlers:
            self._fire_event(Events.EXCEPTION_RAISED, e)
        else:
            raise e

    def run(self, data, max_iters=1):
        """Runs the process_function over the passed data.

        Args:
            data (Iterable): Collection of batches allowing repeated iteration (e.g., list or `DataLoader`).
            max_iters (int, optional): max iterations to run for (default: 1).

        Returns:
            State: output state.
        """

        self.state = State(dataloader=data, max_iters=max_iters, metrics={})

        try:
            self._run_once_on_dataset(max_iters)
            self._fire_event(Events.COMPLETED)

        except BaseException as e:
            self._logger.error("Engine run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)

        return self.state
