export TASK=coqa
export EPOCH=5
export BATCH=4
export RUMI_LEN=5
export GRADIENT_ACCUMULATION_STEPS=4
export DEVICE=0
export WARMUP_STEPS=150
export LR=1e-5
export EVAL_STEPS=300
export WEIGHT_DECAY=0.005
export SEED=42
export MAX_SEQ_LEN=80
export DATA_DIR=./data/${TASK}


CUDA_VISIBLE_DEVICES=${DEVICE} python cli_rumi.py \
--task_name ${TASK} \
--model_type deberta \
--model_name_or_path ./models/deberta_large \
--do_train \
--do_eval \
--do_predict \
--data_dir $DATA_DIR \
--learning_rate ${LR} \
--num_train_epochs ${EPOCH} \
--max_seq_length ${MAX_SEQ_LEN} \
--rumi_length ${RUMI_LEN} \
--output_dir ./outputs/models_deberta/coqa/concat/_lr${LR}_batch${BATCH}_warmup${WARMUP_STEPS}_decay${WEIGHT_DECAY}_rumi${RUMI_LEN} \
--per_device_eval_batch_size=16 \
--per_device_train_batch_size=${BATCH} \
--seed ${SEED} \
--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
--evaluation_strategy "steps" \
--eval_steps ${EVAL_STEPS} \
--warmup_steps ${WARMUP_STEPS} \
--weight_decay ${WEIGHT_DECAY} \
--logging_steps ${EVAL_STEPS} \
--save_steps ${EVAL_STEPS} \
--save_total_limit 1 \
--metric_for_best_model acc \
--overwrite_output

#export ANLI_DIR=./data/anli
#CUDA_VISIBLE_DEVICES=2 python cli_rumi.py \
#--task_name anli \
#--model_type roberta \
#--model_name_or_path ./models/roberta_large \
#--do_train \
#--do_eval \
#--do_predict \
#--data_dir $ANLI_DIR \
#--learning_rate 5e-6 \
#--num_train_epochs 3 \
#--max_seq_length 128 \
#--rumi_length 15 \
#--output_dir ./outputs/models_roberta/anli/ffn/roberta_large \
#--per_device_eval_batch_size=16 \
#--per_device_train_batch_size=16 \
#--seed 42 \
#--gradient_accumulation_steps 1 \
#--evaluation_strategy "steps" \
#--eval_steps 1000 \
#--logging_steps 1000 \
#--save_steps 1000 \
#--save_total_limit 1 \
#--metric_for_best_model acc \
#--overwrite_output
#
#export PIQA_DIR=./data/piqa/seed42
#CUDA_VISIBLE_DEVICES=5 python cli_rumi.py \
#--task_name piqa \
#--model_type roberta \
#--model_name_or_path ./models/roberta_large \
#--do_train \
#--do_eval \
#--do_predict \
#--data_dir $PIQA_DIR \
#--learning_rate 1e-5 \
#--num_train_epochs 20 \
#--max_seq_length 150 \
#--rumi_length 3 \
#--output_dir ./outputs/models_roberta/piqa/concat \
#--per_device_eval_batch_size=16 \
#--per_device_train_batch_size=4 \
#--seed 42 \
#--gradient_accumulation_steps 8 \
#--evaluation_strategy "steps" \
#--eval_steps 150 \
#--warmup_steps 150 \
#--weight_decay 0.005 \
#--logging_steps 150 \
#--save_steps 150 \
#--save_total_limit 1 \
#--metric_for_best_model acc \
#--overwrite_output
#
#export HELLASWAG_DIR=./data/hellaswag/seed42
#CUDA_VISIBLE_DEVICES=2 python cli_rumi.py \
#--task_name hellaswag \
#--model_type roberta \
#--model_name_or_path ./models/roberta_large \
#--do_train \
#--do_eval \
#--do_predict \
#--data_dir $HELLASWAG_DIR \
#--learning_rate 1e-5 \
#--num_train_epochs 3 \
#--max_seq_length 160 \
#--rumi_length 3 \
#--output_dir ./outputs/models_roberta/hellaswag/ffn \
#--per_device_eval_batch_size=16 \
#--per_device_train_batch_size=4 \
#--seed 42 \
#--gradient_accumulation_steps 2 \
#--evaluation_strategy "steps" \
#--eval_steps 1000 \
#--warmup_steps 450 \
#--weight_decay 0.005 \
#--logging_steps 1000 \
#--save_steps 1000 \
#--save_total_limit 1 \
#--metric_for_best_model acc \
#--overwrite_output