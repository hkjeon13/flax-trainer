from typing import Optional, Union, Callable, Dict, Any, List, Iterable
from packaging import version
from itertools import chain
from functools import partial
from tqdm import tqdm
import importlib.util

import jax
import jax.numpy as jnp
from jax.lax import pmean
from jax import pmap, jit


import flax
from flax.jax_utils import unreplicate, replicate
from flax.training.train_state import TrainState
from flax.training.common_utils import onehot

from optax import softmax_cross_entropy

import datasets
from datasets import load_metric

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, IterableDataset

from transformers.trainer_utils import (
    RemoveColumnsCollator,
    TrainerMemoryTracker,
)
from transformers.trainer_callback import (
    DefaultFlowCallback,
    TrainerCallback,
    ProgressCallback
)

from transformers import (
    TrainingArguments,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    logging,
    FlaxPreTrainedModel,
)

from dataloader import BatchLoader
from utils import *


logger = logging.get_logger(__name__)

_datasets_available = importlib.util.find_spec("datasets") is not None
DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


def logit_function(logits):
    return logits.argmax(-1)


class FlaxTrainer(object):
    def __init__(self,
                 model: FlaxPreTrainedModel,
                 args: TrainingArguments,
                 data_collator: Optional[DataCollator] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
                 eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
                 model_init: Callable[[], FlaxPreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 rng_seed: int = 432):

        self.model, self.args = model, args

        self.rng = jax.random.PRNGKey(rng_seed)

        self.train_batch_loader, self.eval_batch_loader = None, None

        if train_dataset:
            self.train_batch_loader = BatchLoader(train_dataset, args.per_device_train_batch_size)

        if eval_dataset:
            self.eval_batch_loader = BatchLoader(eval_dataset, args.per_device_eval_batch_size)

        self.compute_metrics = compute_metrics

        self.lr_scheduler = get_linear_scheduler(
            learning_rate=args.learning_rate, end_value=1e-6,
            warmup_steps=args.warmup_steps
        )

        self.optimizer = get_adam_optimizer(
            scheduler=self.lr_scheduler,
            b1=args.adam_beta1, b2=args.adam_beta2, eps=args.adam_epsilon, weight_decay=args.weight_decay,
        )

        self.state = TrainState.create(
            apply_fn=self.model.__call__,
            params=self.model.params,
            tx=self.optimizer
        )

        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        self._memory_tracker.stop_and_update_metrics()

    @partial(jit, static_argnums=0)
    def train_step(self, state: TrainState, batch: Dict[str, jax.numpy.DeviceArray], dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        labels = batch.pop("labels", None)

        def loss_fn(params):
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            onehot_labels = onehot(labels, logits.shape[-1])
            return softmax_cross_entropy(logits, onehot_labels).mean()

        loss, grad = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=pmean(grad, "batch"))
        self.lr_scheduler(state.step)
        return new_state, pmean({"loss": loss}, axis_name="batch"), new_dropout_rng

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs):
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        state = replicate(self.state)
        try:
            for epoch in range(int(self.args.num_train_epochs)):
                updates, dropout_rngs = [], jax.random.split(self.rng, jax.local_device_count())
                u_append = updates.append

                parallel_train_step = pmap(self.train_step, "batch", donate_argnums=(0,))
                with tqdm(total=len(self.train_batch_loader), leave=True, position=0) as pbar:
                    for batch in self.train_batch_loader:
                        state, train_metrics, dropout_rngs = parallel_train_step(state, batch, dropout_rngs)
                        u_append(train_metrics)
                        pbar.update(1)
                pbar.set_postfix(get_updates(epoch + 1, updates))

            if self.args.do_eval and self.eval_batch_loader:
                self.evaluate(eval_dataset=self.eval_batch_loader)
        except Exception as e:
            logger.error(e)
        finally:
            self.state = unreplicate(state)

    @partial(jit, static_argnums=0)
    def eval_step(self, state, batch):
        logits = state.apply_fn(**batch, params=state.params, train=False)[0]
        return logits

    def evaluate(self, eval_dataset: Optional[Union[Dataset, IterableDataset]] = None):
        state = flax.jax_utils.replicate(self.state)
        if eval_dataset:
            self.eval_batch_loader = BatchLoader(eval_dataset, self.args.per_device_eval_batch_size)

        _predictions, _labels = [], []
        parallel_eval_step = jax.pmap(self.eval_step, axis_name="batch")
        with tqdm(total=len(self.eval_batch_loader), desc="Evaluating...", leave=True, position=0) as pbar:
            for i,batch in enumerate(self.eval_batch_loader):
                if "labels" in batch:
                    _labels.append(batch.pop('labels'))
                _predictions.append(parallel_eval_step(state, batch))
                pbar.update(1)
                if i > 2:
                    break


        _predictions = jnp.squeeze(jnp.concatenate(_predictions, axis=1))
        _labels = jnp.squeeze(jnp.concatenate(_labels, axis=1))
        print(f"predictions 3: {_predictions}")
        print(f"labels 3: {_labels}")

        if self.compute_metrics:
            eval_metric = self.compute_metrics((_predictions, _labels))
            eval_metric = {k: v for k, v in eval_metric.items()}
            pbar.set_postfix({k: v for k, v in eval_metric.items()})

        self.state = flax.jax_utils.unreplicate(state)

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