try:
    from SpringCommonInterface import Metric as SCIMetric
except ImportError:
    SCIMetric = object


class Metric(SCIMetric):
    """"Base class for a metric"""
    def __init__(self, metric_dict={}):
        self.metric = metric_dict
        super(Metric, self).__init__(self.metric)

    def __str__(self):
        return f'metric={self.metric} key={self.cmp_key}'

    def __repr__(self):
        return f'metric={self.metric} key={self.cmp_key}'

    def update(self, up_dict={}):
        self.metric.update(up_dict)

    def set_cmp_key(self, key):
        self.cmp_key = key
        self.v = self.metric[self.cmp_key]


class Evaluator(object):
    """Base class for a evaluator"""
    def __init__(self):
        pass

    def eval(self, res_file, **kwargs):
        """
        This should return a dict with keys of metric names,
        values of metric values.

        Arguments:
            res_file (str): file that holds classification results
        """
        raise NotImplementedError

    @staticmethod
    def add_subparser(self, name, subparsers):
        raise NotImplementedError

    @staticmethod
    def from_args(cls, args):
        raise NotImplementedError
