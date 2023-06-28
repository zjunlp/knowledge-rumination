#lrs=(1e-5 5e-6)
#bsz=(2 4 8)
#warmup_steps=(20 50 100)
#rumi_lengths=(3 5 7)
#weight_decays=(0.01 0.1 0.005)
#
#for lr in "${lrs[@]}"
#do
#  for bs in "${bsz[@]}"
#  do
#    for warmup_step in "${warmup_steps[@]}"
#        do
#            for rumi_length in "${rumi_lengths[@]}"
#            do
#                for weight_decay in "${weight_decays[@]}"
#                do
#                    CUDA_VISIBLE_DEVICES=3 python run.py \
#                    --task_name SST-2 \
#                    --data_dir data/k-shot/SST-2/16-42 \
#                    --overwrite_output_dir \
#                    --do_train \
#                    --do_eval \
#                    --do_predict \
#                    --evaluation_strategy "steps" \
#                    --model_name_or_path ./models/roberta_large \
#                    --few_shot_type prompt \
#                    --num_k 16 \
#                    --max_steps 1000 \
#                    --eval_steps  30 \
#                    --logging_steps 30 \
#                    --warmup_steps ${warmup_step} \
#                    --per_device_train_batch_size ${bs} \
#                    --gradient_accumulation_steps 1 \
#                    --learning_rate 1e-5 \
#                    --output_dir ./result/tmp/lr${lr}_bs${bs}_warmupstep${warmup_step}_rumi${rumi_length}_decay${weight_decay} \
#                    --seed 42 \
#                    --rumi_length ${rumi_length} \
#                    --weight_decay ${weight_decay} \
#                    --rumi_template "*cls**sent_0*_As_far_as_i_know*mask*.*sep+*" \
#                    --template "*cls**sent_0*_It_was*mask*.*sep+*" \
#                    --mapping "{'0':'terrible','1':'great'}" \
#                    --num_sample 16
#                done
#            done
#        done
#  done
#done
# seq_lens=(130 140 150 160 170 180 190 200 210 220 230 240 250)
# for seq_len in "${seq_lens[@]}"
# do
#   CUDA_VISIBLE_DEVICES=5 python run.py \
#   --task_name QNLI \
#   --data_dir data/k-shot/QNLI/16-42 \
#   --overwrite_output_dir \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --evaluation_strategy "steps" \
#   --model_name_or_path ./models/roberta_large \
#   --few_shot_type prompt \
#   --num_k 16 \
#   --max_steps 1000 \
#   --max_seq_len ${seq_len} \
#   --warmup_steps 20 \
#   --eval_steps  30 \
#   --logging_steps 30 \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 1e-5 \
#   --output_dir ./result/tmp/seq_len${seq_len} \
#   --seed 42 \
#   --weight_decay 0.007 \
#   --template "*cls**sent-_0*?*mask*,*+sentl_1**sep+*" \
#   --rumi_length 5 \
#   --rumi_template "*cls**sent-_0*,*+sentl_1**_About_natural_language_inference,_as_far_as_i_know*mask**sep+*" \
#   --mapping "{'entailment': 'Yes', 'not_entailment': 'No'}" \
#   --num_sample 16
# done

#--rumi_template "*cls**sent-_0*,*about_natural_language_inference,_as_far_as_i_know*mask*,*+sentl_1**sep+*" \
#--rumi_template "*cls**sent-_0*,*+sentl_1**_About_natural_language_inference,_as_far_as_i_know*mask**sep+*" \
#CUDA_VISIBLE_DEVICES=3 python run.py \
#--task_name SST-2 \
#--data_dir data/k-shot/SST-2/16-42 \
#--overwrite_output_dir \
#--do_train \
#--do_eval \
#--do_predict \
#--evaluation_strategy "steps" \
#--model_name_or_path ./models/roberta_large \
#--few_shot_type prompt \
#--num_k 16 \
#--max_steps 1000 \
#--eval_steps  30 \
#--logging_steps 30 \
#--warmup_steps 50 \
#--per_device_train_batch_size 2 \
#--gradient_accumulation_steps 1 \
#--learning_rate 1e-5 \
#--output_dir result/tmp \
#--overwrite_cache \
#--seed 42 \
#--rumi_length 5 \
#--weight_decay 0.01 \
#--rumi_template "*cls**sent_0*_about_sentiment_analysis,_as_far_as_i_know*mask*.*sep+*" \
#--template "*cls**sent_0*_It_was*mask*.*sep+*" \
#--mapping "{'0':'terrible','1':'great'}" \
#--num_sample 16

export seq_len=128
export task_name=MRPC
export wu=137

  CUDA_VISIBLE_DEVICES=0 python run.py \
  --task_name ${task_name} \
  --data_dir data/original/${task_name} \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --evaluation_strategy "steps" \
  --model_name_or_path ./models/roberta-large \
  --few_shot_type finetune \
  --num_k 16 \
  --max_steps 1200 \
  --max_seq_len ${seq_len} \
  --warmup_steps ${wu} \
  --eval_steps  50 \
  --logging_steps 50 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --output_dir ./result/${task_name}/finetune_seq_len${seq_len}_wu${wu} \
  --seed 42 \
  --weight_decay 0.007 \
  --template "*cls**sent_0**mask*,*+sentl_1**sep+*" \
  --rumi_length 7 \
  --rumi_template "*cls**sent_0*_about_sentences_paraphrase,_as_far_as_i_know*mask*.*sep+*" \
  --mapping "{'0':'No','1':'Yes'}" \