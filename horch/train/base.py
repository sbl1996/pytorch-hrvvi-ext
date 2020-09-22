from collections import OrderedDict
from typing import Mapping


class Serializable:

    def state_dict(self) -> OrderedDict:
        pass

    def load_state_dict(self, state_dict: Mapping) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("Argument state_dict should be a dictionary, but given {}".format(type(state_dict)))


class StatefulList(Serializable):

    def __init__(self, xs):
        for x in xs:
            if not hasattr(x, "state_dict"):
                raise TypeError("Object {} should have `state_dict` method".format(type(x)))
        self.xs = xs

    def state_dict(self):
        d = {"states": []}
        for x in self.xs:
            d['states'].append(x.state_dict())
        return d

    def load_state_dict(self, d):
        for x, state_dict in zip(self.xs, d['states']):
            x.load_state_dict(state_dict)