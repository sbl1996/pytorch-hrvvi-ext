from horch.core.config.yaml_helpers import serializable


class Catalog:

    def __init__(self):
        pass


def register_op(cls, serialize=True, force=False):
    if force:
        setattr(Catalog, cls.__name__, cls)
    else:
        if not hasattr(Catalog, cls.__name__):
            setattr(Catalog, cls.__name__, cls)
        else:
            raise KeyError("The {} class has been registered.".format(cls.__name__))
    return serializable(cls) if serialize else cls

