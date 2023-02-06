from abc import abstractclassmethod

from pygraphon.graphons import Graphon


class BaseMetric:

    @abstractclassmethod
    def __call__(self, graphon: Graphon, estimator:Graphon):
        pass

    @abstractclassmethod
    def __str__(self):
        pass
