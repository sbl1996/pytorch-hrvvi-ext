from horch.core.catalog import Catalog


def get_optimizer(cfg, net):
    optimizer_type = cfg.type
    cfg.pop("type")
    optimizer = getattr(Catalog, optimizer_type)(net.parameters(), **cfg)
    return optimizer


def get_lr_scheduler(cfg, optimizer):
    lr_scheduler_type = cfg.type
    cfg.pop("type")
    lr_scheduler = getattr(Catalog, lr_scheduler_type)(optimizer, **cfg)
    return lr_scheduler
