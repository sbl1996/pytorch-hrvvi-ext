import re

from horch.ext.checkpoint import ModelCheckpoint


class ByMetric:

    def __init__(self, metric, patience=0):
        self.metric = metric
        self.patience = patience

    def parse(self, trainer):
        mat = re.match(
            "(?P<sign>-?)(?P<metric>val_[a-zA-Z]+)", self.metric)
        assert mat, "save by metric must be of form `-?val_<evaluate_metric>`"
        sign = -1 if mat.group('sign') else 1
        save_metric = mat.group('metric')
        assert save_metric[4:] in trainer.evaluate_metrics, "the metric used must be one of \
            evaluate_metrics"

        def score_function(e): return sign * \
                                      trainer.metric_history[save_metric][-1]

        checkpoint_handler = ModelCheckpoint(
            trainer.save_path, trainer.name,
            score_name=save_metric, patience=self.patience,
            score_function=score_function,
            save_as_state_dict=True, require_empty=False)
        checkpoint_handler._iteration = trainer.epochs()
        return checkpoint_handler

class PerEpochs:

    def __init__(self, epochs):
        self.epochs = epochs

    def parse(self, trainer):
        checkpoint_handler = ModelCheckpoint(
            trainer.save_path, trainer.name, self.epochs, save_as_state_dict=True, require_empty=False)
        checkpoint_handler._iteration = trainer.epochs()
        return checkpoint_handler
