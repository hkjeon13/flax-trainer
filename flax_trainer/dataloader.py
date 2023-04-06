import datasets
from flax.training.common_utils import shard
from collections import defaultdict
import jax.numpy as jnp
import numpy as np


class BatchLoader(object):
    def __init__(self, data, batch_size: int = 4):
        self.length = len(data)
        self.dataset = data
        if isinstance(self.dataset, (datasets.Dataset, datasets.IterableDataset)):
            self.dataset = self.dataset.with_format("jax")
        self.batch_size = batch_size

    def __iter__(self):
        if isinstance(self.dataset, (datasets.Dataset, datasets.IterableDataset)):
            for d in self.dataset.iter(batch_size=self.batch_size):
                yield {k: shard(v) for k, v in d.items()}

        else:
            outputs, batch_len = defaultdict(list), 0
            for values in self.dataset:
                batch_len += 1
                for k, v in values.items():
                    outputs[k].append(v)
                if not batch_len % self.batch_size:
                    yield {k: shard(jnp.array(np.array(v))) for k, v in outputs.items()}
                    outputs.clear()

            if outputs:
                return {k: shard(jnp.array(np.array(v))) for k, v in outputs.items()}

    def __len__(self):
        return (self.length // self.batch_size) + 1 * (self.length % self.batch_size != 0)