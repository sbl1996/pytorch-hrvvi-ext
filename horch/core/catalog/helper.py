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


def get_dataloader(cfg, ds):
    if cfg.get("use_process", True):
        from torch.utils.data import DataLoader
        return DataLoader(ds,
                          batch_size=cfg.batch_size,
                          num_workers=cfg.get("num_workers", 1),
                          shuffle=cfg.get("shuffle", False),
                          pin_memory=cfg.get("pin_memory", True))
    else:
        from horch.dataloader.dataloader import DataLoader
        return DataLoader(ds,
                          batch_size=cfg.batch_size,
                          num_workers=cfg.get("num_workers", 1),
                          shuffle=cfg.get("shuffle", False),
                          pin_memory=False)


def get_model(cfg, pkg):
    return getattr(pkg, cfg.Model)(**cfg.get(cfg.Model))