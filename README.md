# Trainer For Flax Model (with Huggingface 'transformers')

---

## Installation
```
pip install flax-trainer
```

## Getting started

### Prerequisites
```
pip install -r requirements.txt
```

### Finetuning and Test (with default parameters)
```
python examples/run_clm.py --do_train=True --do_eval=True --model_name_or_path=skt/kogpt2-base-v2 \
--output_dir=runs --data_name_or_path=psyche/kowiki --text_column=text --per_device_train_batch_size=16 \
--per_device_eval_batch_size=16
```
