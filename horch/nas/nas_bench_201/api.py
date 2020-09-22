import numpy as np
import torch


def simplify(d):
    min_keys = {
        'eval_acc1es': 'ori-test@199',
        'eval_losses': 'ori-test@199',
        'eval_times': 'ori-test@199',
        'train_acc1es': 199,
        'train_acc5es': None,
        'train_losses': 199,
        'train_times': 199
    }
    for i in range(d['total_archs']):
        print(i)
        x = d['arch2infos'][i]
        del x['less']
        for dk, res in x['full']['all_results'].items():
            for k, v in res.items():
                if k in min_keys:
                    mk = min_keys[k]
                    if mk:
                        res[k] = v[mk]


class SimpleNASBench201:

    def __init__(self, fp):
        self.d = torch.load(fp)

        self._arch2index = {}
        for i, arch in enumerate(self.d['meta_archs']):
            self._arch2index[arch] = i

        self._eval_acc_rank = {}
        for d in self.datasets:
            ranks = sorted(range(self.total_archs), key=lambda i: np.mean(self.query_eval_acc(i, d)), reverse=True)
            rrank = [0] * self.total_archs
            for r, arch_index in enumerate(ranks):
                rrank[arch_index] = r
            self._eval_acc_rank[d] = rrank

    @property
    def datasets(self):
        return ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']

    @property
    def total_archs(self):
        return self.d['total_archs']

    def query_all(self, arch_or_index, dataset='cifar10'):
        index = self.get_index(arch_or_index)
        x = self.d['arch2infos'][index]['full']
        seeds = x['dataset_seed'][dataset]
        results = {}
        for seed in seeds:
            results[seed] = x['all_results'][(dataset, seed)]
        return results

    def query_eval_acc(self, arch_or_index, dataset='cifar10'):
        results = self.query_all(arch_or_index, dataset)
        return [r['eval_acc1es'] for r in results.values()]

    def query_eval_acc_rank(self, arch_or_index, dataset='cifar10'):
        index = self.get_index(arch_or_index)
        return self._eval_acc_rank[dataset][index]

    def get_index(self, arch_or_index):
        if isinstance(arch_or_index, str):
            index = self._arch2index[arch_or_index]
        else:
            index = arch_or_index
        return index