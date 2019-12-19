import re

from horch.ext.checkpoint import ModelCheckpoint


class ByMetric:

    def __init__(self, metric, patience=0):
        self.metric = metric
        self.patience = patience

    def parse(self, trainer):
        mat = re.match(
            "(?P<sign>-?)(?P<dataset>[a-zA-Z0-9]+)_(?P<metric>[a-zA-Z0-9]+)", self.metric)
        assert mat, "save by metric must be of form `-?<valset_name>?_<test_metric>`"
        sign = -1 if mat.group('sign') else 1
        metric_name = mat.group("metric")
        save_metric = mat.group("dataset") + "_" + metric_name
        assert metric_name in trainer.test_metrics, "Metric %s not in traier's test_metrics" % metric_name
        def score_function(e): return sign * trainer.metric_history[save_metric][-1]

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
