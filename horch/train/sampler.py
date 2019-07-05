from math import inf

from torch.utils.data.sampler import Sampler, RandomSampler, BatchSampler, SequentialSampler


class IterationBasedBatchSampler(Sampler):
    """
    Wraps a BatchSampler, re-sampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


class IterSampler(Sampler):
    """
    Wraps a BatchSampler, re-sampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, data_source, batch_size, shuffle=True, num_iterations=inf, start_iter=0):
        super().__init__(data_source)
        if shuffle:
            sampler = RandomSampler(data_source)
        else:
            sampler = SequentialSampler(data_source)
        self.data_source = data_source
        self.num_iterations = num_iterations
        self.batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        # while iteration <= self.num_iterations:
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations
