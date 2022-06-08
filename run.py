def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer, FlaxBertForSequenceClassification, TrainingArguments
    from trainer import FlaxTrainer

    MODEL = "klue/bert-base"
    dataset = load_dataset("nsmc")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = FlaxBertForSequenceClassification.from_pretrained(MODEL, from_pt=True)

    def example_fn(examples):
        tokenized = tokenizer(examples["document"], padding='max_length', max_length=256, truncation=True)
        tokenized['labels'] = examples['label']
        return tokenized

    dataset['train'] = dataset['train'].map(example_fn, batched=True, remove_columns=dataset['train'].column_names)
    dataset['test'] = dataset['test'].map(example_fn, batched=True, remove_columns=dataset['test'].column_names)

    training_args = TrainingArguments("runs/", num_train_epochs=3, per_device_train_batch_size=32,
                                      per_device_eval_batch_size=16)

    trainer = FlaxTrainer(model, args=training_args, train_dataset=dataset['train'], eval_dataset=dataset['test'])
    trainer.train()


if __name__ == "__main__":
    main()