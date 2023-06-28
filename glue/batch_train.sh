warmup_steps=(20 50 100)
rumi_lengths=(5 7 10 15)
weight_decays=(0.01 0.1 0.005)

for warmup_step in ${warmup_steps[@]}
do
    for rumi_length in ${rumi_lengths[@]}
    do
        for weight_decay in ${weight_decays[@]}
        do
            CUDA_VISIBLE_DEVICES=4 python run.py \
                --task_name QNLI \
                --data_dir data/original/QNLI \
                --overwrite_output_dir \
                --do_train \
                --do_eval \
                --do_predict \
                --evaluation_strategy "steps" \
                --save_total_limit 1 \
                --load_best_model_at_end \
                --metric_for_best_model "eval_acc" \
                --model_name_or_path ./models/roberta_large \
                --few_shot_type prompt \
                --num_k 16 \
                --max_steps 1000 \
                --eval_steps  30 \
                --logging_steps 30 \
                --warmup_steps ${warmup_step} \
                --per_device_train_batch_size 2 \
                --gradient_accumulation_steps 1 \
                --learning_rate 1e-5 \
                --output_dir result/full_data_lr1e-5_rumi${rumi_length}_wu${warmup_step}_wd${weight_decay} \
                --overwrite_cache \
                --seed 42 \
                --rumi_length ${rumi_length} \
                --weight_decay ${weight_decay} \
                --rumi_template "*cls**sent_0*_about_natural_language_inference,_as_far_as_i_know*mask*.*sep+*" \
                --template "*cls**sent-_0*?*mask*,*+sentl_1**sep+*" \
                --mapping "{'not_entailment':'No','entailment':'Yes'}" \
                --num_sample 16
        done
    done
done



