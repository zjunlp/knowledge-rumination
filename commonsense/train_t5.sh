export COQA_DIR=./data/coqa
CUDA_VISIBLE_DEVICES=2 python cli_t5.py \
--task_name coqa \
--model_type t5 \
--model_name_or_path t5-base \
--do_train \
--do_eval \
--data_dir $COQA_DIR \
--learning_rate 1e-4 \
--num_train_epochs 10 \
--max_seq_length 120 \
--max_target_length 20 \
--output_dir ./outputs/models_t5/t5_base \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps 1 \
--overwrite_output \
--warmup_steps 100 \
--seed 1 \
--num_beams 1 \
--early_stopping \
--fp16
