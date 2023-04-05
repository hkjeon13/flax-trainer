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
        default="nsmc", metadata={"help": "데이터셋 이름 또는 로컬 경로를 설정합니다."}
    )
    text_column_name: str = field(
        default="document", metadata={"help": "텍스트 데이터를 가지고 있는 컬럼의 이름을 설정합니다."}
    )

    label_column_name: str = field(
        default="label", metadata={"help": "레이블 데이터를 가지고 있는 컬럼의 이름을 설정합니다."}
    )

    padding: str = field(
        default="max_length", metadata={"help": "패딩을 어떻게 할지 설정합니다."}
    )

    max_length: int = field(
        default=256, metadata={"help": "최대 길이를 설정합니다."}
    )

    truncation: bool = field(
        default=True, metadata={"help": "문장이 길 경우 자를지 설정합니다."}
    )

    train_split: str = field(
        default="train", metadata={"help": "학습 데이터를 가지고 있는 split의 이름을 설정합니다."}
    )

    eval_split: str = field(
        default="test", metadata={"help": "평가 데이터를 가지고 있는 split의 이름을 설정합니다."}
    )

    train_samples: int = field(
        default=None, metadata={"help": "학습 데이터의 샘플 수를 설정합니다."}
    )

    eval_samples: int = field(
        default=None, metadata={"help": "평가 데이터의 샘플 수를 설정합니다."}
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="klue/bert-base", metadata={"help": "모델 이름 또는 로컬 경로를 설정합니다."}
    )

    from_pt: bool = field(
        default=True, metadata={"help": "flax 모델이 아닌 pytorch 모델을 로드할지 설정합니다."}
    )

    metric_name: str = field(
        default="accuracy", metadata={"help": "평가에 사용할 metric의 이름을 설정합니다."}
    )


@dataclass
class TrainArguments(TrainingArguments):
    output_dir = "runs/"


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    dataset = load_dataset(data_args.data_name_or_path)

    if data_args.train_samples:
        max_train_length = dataset[data_args.train_split].num_rows
        dataset[data_args.train_split] = dataset[data_args.train_split].select(range(min(data_args.train_samples, max_train_length)))

    if data_args.eval_samples:
        max_eval_length = dataset[data_args.eval_split].num_rows
        dataset[data_args.eval_split] = dataset[data_args.eval_split].select(range(min(data_args.eval_samples, max_eval_length)))


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