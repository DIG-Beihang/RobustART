from __future__ import division
import random
import numpy as np
from .register_factory import TRANSFORM

__all__ = ["Compose", "OneOf", "OneOrOther"]


@TRANSFORM.register("compose")
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item, **kwargs):

        for t in self.transforms:
            item = t(item, **kwargs)
        if isinstance(item, tuple) or isinstance(item, list):
            for i in item:
                if hasattr(i, "lazy"):
                    if i.lazy():
                        i.update_lazy_action()
            res = []
            for i in item:
                if hasattr(i, "params"):
                    res.append(i.params())
                else:
                    res.append(i)
            return res
        elif isinstance(item, np.ndarray):
            return item
        else:
            if item.lazy():
                item.update_lazy_action()

            return item.params()


@TRANSFORM.register("oneof")
class OneOf(Compose):
    def __call__(self, item, **kwargs):
        return random.choice(self.transforms)(item, **kwargs)


@TRANSFORM.register("oneorother")
class OneOrOther(Compose):
    def __init__(self, transforms, p):
        assert len(transforms) == 2, f"OneOrOther compose only support two transforms, given: {transforms}!"
        super(OneOrOther, self).__init__(transforms)
        self.p = p

    def __call__(self, item, **kwargs):
        idx = 0 if random.random() < self.p else 1
        return self.transforms[idx](item, **kwargs)
