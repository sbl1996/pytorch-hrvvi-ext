import torch.optim
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.transforms import Compose

import horch.optim
import horch.transforms.classification
import horch.optim.lr_scheduler
import horch.train.cls.mix


def getattr_mods(mods, name):
    for mod in mods:
        try:
            return getattr(mod, name)
        except:
            continue


def get_transform(cfg):
    transforms = []
    for t in cfg:
        name, params = list(t.items())[0]
        if params is None:
            params = {}
        t = getattr_mods([
            torchvision.transforms,
            horch.transforms,
        ], name)(**params)
        transforms.append(t)
    return Compose(transforms)


def get_model(cfg, mod):
    typ = cfg.type
    kwargs = {**cfg}
    kwargs.pop("type")
    return getattr(mod, typ)(**kwargs)


def get_lr_scheduler(cfg, optimizer, epochs):
    typ = cfg.type
    kwargs = {**cfg}
    kwargs.pop("type")
    lr_scheduler = getattr_mods([
        torch.optim.lr_scheduler,
        horch.optim.lr_scheduler,
    ], typ)(optimizer=optimizer, epochs=epochs, **kwargs)
    return lr_scheduler


def get_optimizer(cfg, params):
    typ = cfg.type
    kwargs = {**cfg}
    kwargs.pop("type")
    optimizer = getattr_mods([
        torch.optim, horch.optim,
    ], typ)(params=params, **kwargs)
    return optimizer


def get_data_loader(cfg, ds):
    batch_size = cfg.batch_size
    shuffle = cfg.get("shuffle", False)
    num_workers = cfg.get("num_workers", 2)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_mix(cfg, ds):
    if cfg is None:
        return None
    typ = cfg.type
    kwargs = {**cfg}
    kwargs.pop("type")
    mix = getattr(horch.train.cls.mix, typ)(dataset=ds, **kwargs)
    return mix


def resume(trainer, resume_arg):
    try:
        epoch = int(resume_arg)
        if epoch == 0:
            trainer.load()
        else:
            fp = trainer.work_dir / ("epoch_%d.pt" % epoch)
            trainer.load(fp)
    except ValueError:
        trainer.load(resume_arg)
    except Exception as e:
        raise e
