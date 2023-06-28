# Knowledge-Rumination


```bash
conda create -n rumination python=3.8
```
First install the transformers:
```bash
cd transformers-4.3.0
pip install --editable .
cd ..
pip install -r transformers
```

## Initialize the model
```bash
    python initialization.py
```

```bash
export COQA_DIR=./data/coqa
python cli.py \
--task_name coqa \
--model_type roberta \
--model_name_or_path roberta-base \
--do_train \
--do_eval \
--data_dir $COQA_DIR \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--output_dir ./outputs/models_roberta/roberta_base \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output
```