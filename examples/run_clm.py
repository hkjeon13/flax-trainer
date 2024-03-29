from itertools import chain
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    FlaxGPT2LMHeadModel,
    TrainingArguments,
    HfArgumentParser
)
from dataclasses import dataclass, field
from flax_trainer.trainer import FlaxTrainerForCausalLM


@dataclass
class ModelParams:
    model_name_or_path: str = field(
        default="skt/kogpt2-base-v2",
        metadata={"help": "모델 이름 또는 로컬 경로를 설정합니다."}
    )

    from_pt: bool = field(
        default=True,
        metadata={"help": "flax 모델이 아닌 pytorch 모델을 로드할지 설정합니다."}
    )

    max_length: int = field(
        default=512,
        metadata={"help": "최대 길이를 설정합니다."}
    )
    group_texts: bool = field(
        default=True,
        metadata={"help": "최대 길이(max_length) 기준으로 텍스트를 그룹화 여부를 설정합니다."}
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "그룹화된 텍스트를 얼마나 크게 나눌지 설정합니다."}
    )


@dataclass
class DataParams:
    data_name_or_path: str = field(
        default="psyche/kowiki",
        metadata={"help": "데이터셋 이름 또는 로컬 경로를 설정합니다."}
    )

    text_column_name: str = field(
        default="text",
        metadata={"help": "텍스트 데이터를 가지고 있는 컬럼의 이름을 설정합니다."}
    )

    train_split: str = field(
        default="train",
        metadata={"help": "학습 데이터를 가지고 있는 split의 이름을 설정합니다."}
    )

    eval_split: Optional[str] = field(
        default='validation',
        metadata={"help": "평가 데이터를 가지고 있는 split의 이름을 설정합니다."}
    )

    train_samples: int = field(
        default=None,
        metadata={"help": "학습 데이터의 샘플 수를 설정합니다."}
    )

    eval_samples: int = field(
        default=None,
        metadata={"help": "평가 데이터의 샘플 수를 설정합니다."}
    )


def main():
    parser = HfArgumentParser((ModelParams, DataParams, TrainingArguments))
    model_params, data_params, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_params.model_name_or_path)
    model = FlaxGPT2LMHeadModel.from_pretrained(model_params.model_name_or_path, from_pt=model_params.from_pt)

    dataset = load_dataset(data_params.data_name_or_path)

    if data_params.train_samples is not None:
        dataset[data_params.train_split] = dataset[data_params.train_split].select(
            range(min(data_params.train_samples, len(dataset[data_params.train_split])))
        )

    if data_params.eval_samples is not None:
        dataset[data_params.eval_split] = dataset[data_params.eval_split].select(
            range(min(data_params.eval_samples, len(dataset[data_params.eval_split])))
        )


    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples[data_params.text_column_name])
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset[data_params.train_split].column_names,
        batched=True,
    )

    _block_size = model_params.block_size if model_params.block_size is not None else model_params.max_length

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= _block_size:
            total_length = (total_length // _block_size) * _block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + _block_size] for i in range(0, total_length, _block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if model_params.group_texts:
        dataset = dataset.map(group_texts, batched=True, remove_columns=dataset[data_params.train_split].column_names)

    dataset.set_format(type="jax", columns=["input_ids", "attention_mask", "labels"])

    trainer = FlaxTrainerForCausalLM(
        model=model,
        args=training_args,
        train_dataset=dataset[data_params.train_split],
        eval_dataset=dataset[data_params.eval_split] if data_params.eval_split is not None else None,
        tokenizer=tokenizer
    )

    if training_args.do_train:
        trainer.train()

    elif training_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()



