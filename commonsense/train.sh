export TASK=coqa
export EPOCH=6
export BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=4
export DEVICE=0
export WARMUP_STEPS=150
export LR=1e-5
export EVAL_STEPS=500
export WEIGHT_DECAY=0.005
export SEED=42
export MAX_SEQ_LEN=128
export DATA_DIR=./data/${TASK}

CUDA_VISIBLE_DEVICES=${DEVICE} python cli.py \
--task_name coqa \
--model_type deberta \
--model_name_or_path ./models/vanilla_deberta/deberta-large \
--do_train \
--do_eval \
--do_predict \
--data_dir ${DATA_DIR} \
--learning_rate ${LR} \
--num_train_epochs ${EPOCH} \
--max_seq_length ${MAX_SEQ_LEN} \
--output_dir ./outputs/models_deberta/baseline/${TASK}/_lr${LR}_batch${BATCH_SIZE}_warmup${WARMUP_STEPS} \
--per_device_eval_batch_size=16 \
--per_device_train_batch_size=${BATCH_SIZE} \
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