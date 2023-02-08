import jax.numpy as jnp
from flax_trainer.trainer import FlaxTrainer
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    FlaxBertForSequenceClassification,
    TrainingArguments,
    HfArgumentParser
)

@dataclass
class DataArguments:
    data_name_or_path: str = field(
        default="nsmc"
    )
    text_column_name: str = field(
        default="document"
    )

    label_column_name: str = field(
        default="label"
    )

    padding: str = field(
        default="max_length"
    )

    max_length: int = field(
        default=256
    )

    truncation: bool = field(
        default=True
    )

    train_split: str = field(
        default="train"
    )

    eval_split: str = field(
        default="test"
    )

    train_samples: int = field(
        default=None
    )

    eval_samples: int = field(
        default=None
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="klue/bert-base"
    )

    from_pt: bool = field(
        default=True
    )
    metric_name: str = field(
        default="accuracy"
    )

@dataclass
class TrainArguments(TrainingArguments):
    output_dir = "runs/"


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    dataset = load_dataset(data_args.data_name_or_path)
    if data_args.train_samples:
        dataset[data_args.train_split] = dataset[data_args.train_split].select(range(data_args.train_samples))

    if data_args.eval_samples:
        dataset[data_args.eval_split] = dataset[data_args.eval_split].select(range(data_args.eval_samples))

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    model = FlaxBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, from_pt=model_args.from_pt
    )

    def example_fn(examples):
        tokenized = tokenizer(
            examples[data_args.text_column_name],
            padding=data_args.padding,
            max_length=data_args.max_length,
            truncation=data_args.truncation
        )
        tokenized['labels'] = examples[data_args.label_column_name]
        return tokenized

    dataset[data_args.train_split] = dataset[data_args.train_split].map(
        example_fn,
        batched=True,
        remove_columns=dataset[data_args.train_split].column_names
    )

    dataset[data_args.eval_split] = dataset[data_args.eval_split].map(
        example_fn,
        batched=True,
        remove_columns=dataset[data_args.eval_split].column_names
    )

    _metric = load_metric(model_args.metric_name)
    def compute_metrics(p):
        preds, labels = p
        preds = jnp.argmax(preds, axis=-1)
        return _metric.compute(predictions=preds, references=labels)

    trainer = FlaxTrainer(
        model,
        args=train_args,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset[data_args.eval_split],
        compute_metrics=compute_metrics
    )

    if train_args.do_train:
        trainer.train()

    elif train_args.do_eval:
        trainer.evaluate(dataset[data_args.eval_split])


if __name__ == "__main__":
    main()