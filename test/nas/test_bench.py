import numpy as np

from horch.nas.nas_bench_201.search.darts import Network
from horch.nas.nas_bench_201.api import SimpleNASBench201

api = SimpleNASBench201("/Users/hrvvi/Code/study/pytorch/datasets/NAS-Bench-201-v1_1-096897-simple.pth")
net = Network(4, 8)

val_accs = []
ranks = []
for i in range(100):
    net._initialize_alphas()
    s = net.genotype()
    val_accs.append(np.mean(api.query_eval_acc(s)))
    ranks.append(api.query_eval_acc_rank(s))
