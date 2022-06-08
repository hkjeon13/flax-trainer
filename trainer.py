from packaging import version
from typing import Optional, Union, Callable
import importlib.util
from itertools import chain
from functools import partial
from tqdm import tqdm


import jax
from jax.lax import pmean
from jax import pmap, jit

import flax
from flax.jax_utils import unreplicate, replicate
from flax.training.train_state import TrainState
from flax.training.common_utils import onehot, shard

from optax import softmax_cross_entropy

import datasets
from datasets import load_metric

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, IterableDataset

from transformers.trainer_utils import RemoveColumnsCollator
from transformers.utils import logging

from dataloader import BatchLoader
from utils import *

logger = logging.get_logger(__name__)
_datasets_available = importlib.util.find_spec("datasets") is not None


def logit_function(logits):
    return logits.argmax(-1)


class FlaxTrainer(object):
    def __init__(self,
                 model,
                 args,
                 rng_seed: int = 12,
                 train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
                 eval_dataset: Optional[Union[Dataset, IterableDataset]] = None):

        self.model, self.args = model, args

        self.rng = jax.random.PRNGKey(rng_seed)

        self.train_batch_loader, self.eval_batch_loader = None, None
        if train_dataset:
            self.train_batch_loader = BatchLoader(train_dataset, args.per_device_train_batch_size)

        if eval_dataset:
            self.eval_batch_loader = BatchLoader(eval_dataset, args.per_device_eval_batch_size)

        self.scheduler = get_linear_scheduler(
            learning_rate=args.learning_rate, end_value=1e-6,
            warmup_steps=args.warmup_steps
        )

        self.optimizer = get_adam_optimizer(
            learning_rate=self.scheduler,
            b1=args.adam_beta1, b2=args.adam_beta2, eps=args.adam_epsilon, weight_decay=args.weight_decay,
        )

        self.state = TrainState.create(
            apply_fn=self.model.__call__,
            params=self.model.params,
            tx=self.optimizer
        )

    def train(self, train_dataset: Optional[Union[Dataset, IterableDataset]] = None, batch_size: int = 16):
        state = replicate(self.state)
        try:
            if train_dataset:
                self.train_batch_loader = BatchLoader(train_dataset, batch_size)

            for epoch in range(int(self.args.num_train_epochs)):
                parallel_train_step = pmap(self.train_step, "batch", donate_argnums=(0,))
                updates, dropout_rngs = [], jax.random.split(self.rng, jax.local_device_count())

                with tqdm(total=len(self.train_batch_loader), leave=True, position=0) as pbar:
                    for batch in self.train_batch_loader:
                        state, train_metrics, dropout_rngs = parallel_train_step(state, batch, dropout_rngs)
                        updates.append(train_metrics)
                        pbar.update(1)
                pbar.set_postfix(get_updates(epoch + 1, updates))
        except Exception as e:
            logger.info(e)
        finally:
            self.state = unreplicate(state)

    @partial(jit, static_argnums=0)
    def train_step(self, state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        labels = batch.pop("labels", None)

        @jit
        def loss_fn(params):
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            onehot_labels = onehot(labels, logits.shape[-1])
            return softmax_cross_entropy(logits, onehot_labels).mean()

        grad_func = jax.value_and_grad(loss_fn)
        loss, grad = grad_func(state.params)
        grad = pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)
        self.scheduler(state.step)
        return new_state, pmean({"loss": loss}, axis_name="batch"), new_dropout_rng

    def get_train_dataloader(self) -> DataLoader:

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset, data_collator = self.train_dataset, self.data_collator

        if _datasets_available and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    def _get_collator_with_removed_columns(
            self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    @partial(jit, static_argnums=0)
    def eval_step(self, state, batch):
        logits = state.apply_fn(**batch, params=state.params, train=False)[0]
        return logit_function(logits)

    def evaluate(self, eval_dataset: Optional[Union[Dataset, IterableDataset]] = None):
        state = flax.jax_utils.replicate(self.state)
        if eval_dataset:
            self.eval_batch_loader = BatchLoader(eval_dataset, self.args.per_device_eval_batch_size)

        metric_fn = load_metric('glue', "stsb")
        parallel_eval_step = jax.pmap(self.eval_step, axis_name="batch")
        with tqdm(total=len(self.eval_batch_loader), desc="Evaluating...", leave=True, position=0) as pbar:
            for batch in self.eval_batch_loader:
                labels = batch.pop("labels")
                predictions = parallel_eval_step(state, batch)
                metric_fn.add_batch(predictions=chain(*predictions), references=chain(*labels))
                pbar.update(1)

            eval_metric = metric_fn.compute()
            eval_metric = {k: v for k, v in eval_metric.items()}
            pbar.set_postfix({k: v for k, v in eval_metric.items()})
        self.state = flax.jax_utils.unreplicate(state)