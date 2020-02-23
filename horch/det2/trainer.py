import os
from numbers import Number

import torch
import torch.nn as nn
from typing import Any, Dict, List, Set

from detectron2.config import get_cfg, CfgNode
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.engine import DefaultTrainer


def collect_params(
        model: nn.Module, base_lr: float,
        weight_decay: Number, weight_decay_norm: Number = None,
        bias_lr_factor: Number = 1.0, weight_decay_bias: Number = None) -> List[Dict]:
    """
    Build an optimizer from config.
    """
    weight_decay_norm = weight_decay_norm or weight_decay
    weight_decay_bias = weight_decay_bias or weight_decay
    norm_module_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = base_lr
            m_weight_decay = weight_decay
            if isinstance(module, norm_module_types):
                m_weight_decay = weight_decay_norm
            elif key == "bias":
                lr = base_lr * bias_lr_factor
                m_weight_decay = weight_decay_bias
            params += [{"params": [value], "lr": lr, "weight_decay": m_weight_decay}]
    return params


def get_size(cfg, is_train):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )
    return min_size, max_size, sample_style


class BaseTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg, False)
        mapper.tfm_gens = cls.build_transform_gen(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, True)
        mapper.tfm_gens = cls.build_transform_gen(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_transform_gen(cls, cfg, is_train):
        tfm_gens = []
        min_size, max_size, sample_style = get_size(cfg, is_train)
        tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
        if is_train:
            tfm_gens.append(T.RandomFlip())
        return tfm_gens


def key_to_upper(d):
    if not isinstance(d, dict):
        return d
    return {
        k.upper(): key_to_upper(v)
        for k, v in d.items()
    }


def add_efficientdet_config(cfg):
    _C = cfg
    _C.MODEL.EFFICIENTNET = CfgNode()
    _C.MODEL.EFFICIENTNET.VERSION = "b0"
    _C.MODEL.EFFICIENTNET.PRETRAINED = True

    _C.MODEL.BIFPN = CfgNode()
    _C.MODEL.BIFPN.F_CHANNELS = 64
    _C.MODEL.BIFPN.NUM_LAYERS = 2


def get_config(d):
    cfg = get_cfg()
    add_efficientdet_config(cfg)
    cfg.merge_from_other_cfg(CfgNode(key_to_upper(d)))
    return cfg

