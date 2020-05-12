import uuid
import random


class Transform(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, input, target):
        raise NotImplementedError

    def __str__(self):
        return str(self._id)

    def __repr__(self):
        return pprint(self)


class JointTransform(Transform):

    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

    def __call__(self, input, target):
        return self.transform(input, target)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class InputTransform(Transform):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __call__(self, input, target):
        return self.transform(input), target


class TargetTransform(Transform):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __call__(self, input, target):
        return input, self.transform(target)


class Compose(Transform):

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            if isinstance(t, Transform):
                img, target = t(img, target)
            else:
                img = t(img)
        return img, target


class UseOriginal(Transform):
    """Use the original image and annotations.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, img, target):
        return img, target


class RandomApply(Transform):

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            for t in self.transforms:
                img, target = t(img, target)
        return img, target


class RandomChoice(Transform):

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, img, target):
        t = random.choice(self.transforms)
        img, target = t(img, target)
        return img, target


def pprint(t, level=0, sep='    '):
    pre = sep * level
    if not isinstance(t, Transform) or isinstance(t, JointTransform):
        return pre + repr(t)
    format_string = pre + type(t).__name__ + '('
    if hasattr(t, 'transforms'):
        for t in getattr(t, 'transforms'):
            format_string += '\n'
            format_string += pprint(t, level + 1)
        format_string += '\n'
        format_string += pre + ')'
    elif hasattr(t, 'transform'):
        format_string += '\n'
        format_string += pprint(getattr(t, 'transform'), level + 1)
        format_string += '\n'
        format_string += pre + ')'
    else:
        format_string += ')'
    return format_string
