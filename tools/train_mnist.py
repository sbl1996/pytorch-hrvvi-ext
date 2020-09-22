import argparse

from torchvision.datasets import MNIST

from hhutil.io import read_text, fmt_path

from horch.config.helper import get_transform, get_data_loader, get_model, get_lr_scheduler, get_optimizer, get_mix, \
    resume
from horch.datasets import train_test_split
from horch.defaults import update_defaults

from horch.nn.loss import CrossEntropyLoss

import horch.models.cifar
import horch.optim.lr_scheduler

from horch.config.config import get_config, override
from horch.train import manual_seed
from horch.train.cls import Trainer
from horch.train.cls.metrics import Accuracy
from horch.train.metrics import TrainLoss, Loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MNIST')
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-r', '--resume', help='resume from checkpoints')
    args = parser.parse_args()

    print(read_text(args.config))
    cfg = get_config(args.config)
    dataset = cfg.Dataset.type
    assert dataset == 'MNIST'

    fp16 = cfg.get("fp16", False)
    if fp16:
        assert cfg.device == 'gpu'

    update_defaults(cfg.get("Global", {}))

    manual_seed(cfg.seed)

    data_home = cfg.Dataset.data_home

    train_transform = get_transform(cfg.Dataset.Train.transforms)
    val_transform = get_transform(cfg.Dataset.Val.transforms)
    test_transform = get_transform(cfg.Dataset.Test.transforms)

    num_classes = 10

    if cfg.get("hpo"):
        import nni
        RCV_CONFIG = nni.get_next_parameter()
        for k, v in RCV_CONFIG.items():
            ks = k.split(".")
            override(cfg, ks, v)

    ds_train = MNIST(data_home, train=True)
    ds_train, ds_val = train_test_split(
        ds_train, test_ratio=cfg.Dataset.Split.val_ratio,
        shuffle=cfg.Dataset.Split.get("shuffle", True), random_state=cfg.Dataset.Split.get("seed", 42),
        transform=train_transform, test_transform=val_transform)
    ds_test = MNIST(data_home, train=False, transform=test_transform)

    if cfg.get("Debug") and cfg.Debug.get("subset"):
        ratio = cfg.Debug.subset
        ds_train = train_test_split(ds_train, ratio)[1]
        ds_val = train_test_split(ds_val, ratio)[1]
        ds_test = train_test_split(ds_test, ratio)[1]

    use_mix = cfg.get("Mix") is not None
    if use_mix:
        cfg.Mix.num_classes = num_classes
        ds_train = get_mix(cfg.Mix, ds_train)

    train_loader = get_data_loader(cfg.Dataset.Train, ds_train)
    val_loader = get_data_loader(cfg.Dataset.Val, ds_val)
    test_loader = get_data_loader(cfg.Dataset.Test, ds_test)

    cfg.Model.num_classes = num_classes
    model = get_model(cfg.Model, horch.models.cifar)

    criterion = CrossEntropyLoss(non_sparse=use_mix, label_smoothing=cfg.get("label_smooth"))

    epochs = cfg.epochs
    optimizer = get_optimizer(cfg.Optimizer, model.parameters())
    lr_scheduler = get_lr_scheduler(cfg.LRScheduler, optimizer, epochs)

    train_metrics = {
        'loss': TrainLoss()
    }
    if not use_mix:
        train_metrics['acc'] = Accuracy()

    test_metrics = {
        'loss': Loss(CrossEntropyLoss()),
        'acc': Accuracy(),
    }

    work_dir = fmt_path(cfg.get("work_dir"))
    trainer = Trainer(model, criterion, optimizer, lr_scheduler, train_metrics, test_metrics,
                      work_dir=work_dir, fp16=fp16, device=cfg.get("device", 'auto'))

    if args.resume:
        resume(trainer, args.resume)

    eval_freq = cfg.get("eval_freq", 1)
    save_freq = cfg.get("save_freq")

    if save_freq:
        assert work_dir is not None

    trainer.fit(train_loader, epochs, test_loader, eval_freq, save_freq)

    trainer.evaluate(test_loader)