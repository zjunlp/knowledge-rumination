'''
Utility to train a T5 model, based on arguments
provided in the `training_args.json`, using the following sequence,

1. Load a T5 model and tokenizer either from huggingface or from a
local model config or checkpoint.

2. Load the training and validation datasets, and shuffle if requested.

3. Initialize wandb for logging training progress.

4. Initialize the Trainer class. Some of the important arguments for training
are: per_device_train_batch_size, per_device_eval_batch_size,
gradient_accumulation_steps, learning_rate, num_train_epochs, save_steps.

5. Begin training, and Trainer will checkpoint as often as save_steps.
After training model config/weights will be saved to specified output directory.

6. It is important to override the T2TDataCollator class to format the targets
for the model, and also not calculate loss for the padding tokens
(identified as TOKENIZER_PAD_TOKEN_ID)

This script has been adapted from
https://github.com/huggingface/transformers/blob/master/examples/multiple-choice/
'''
import logging
import os
import sys
import re


from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

import transformers

from transformers import (
    HfArgumentParser,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    set_seed,
)
from model import RumiT5
from utils_commonsense import MultipleChoiceDatasetWithSeq2Seq, Split

logger = logging.getLogger(__name__)

TOKENIZER_PAD_TOKEN_ID = 0

# Make the logging level as INFO
transformers.logging.set_verbosity_info()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

@dataclass
class ModelArguments:
    '''
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    '''

    model_type: str= field(
        metadata={"help": "Model type is T5"}
    )

    model_name_or_path: str = field(
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={'help': 'Where do you want to store the pretrained models downloaded from s3'}
    )
    num_beams: Optional[int] = field(
        default=1, metadata={'help': 'Number of beams for beam search. 1 means no beam search.'}
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={'help': 'Whether to stop the beam search when at least ``num_beams`` sentences \
                                         are finished per batch or not'}
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=0, metadata={'help': 'If set to int > 0, all ngrams of that size can only occur once.'}
    )
    length_penalty: Optional[float] = field(
        default=1.0, metadata={'help': 'Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 \
                                        in order to encourage the model to generate shorter sequences, to a value>1.0 \
                                        in order to encourage the model to produce longer sequences'}
    )
    top_k: Optional[int] = field(
        default=50, metadata={'help': 'The number of highest probability vocabulary tokens to keep for top-k-filtering.'}
    )
    top_p: Optional[float] = field(
        default=1.0, metadata={'help': 'If set to float < 1, only the most probable tokens with probabilities \
                                         that add up to :obj:`top_p` or higher are kept for generation.'}
    )
    repetition_penalty: Optional[float] = field(
        default=1.0, metadata={'help': 'The parameter for repetition penalty. 1.0 means no penalty. See `this paper \
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.'}
    )

@dataclass
class DataTrainingArguments:
    '''
    Arguments pertaining to what data we are going to input our model for training and eval.
    '''
    task_name: str = field(
        metadata={'help': 'Task name'},
    )
    data_dir: Optional[str] = field(
        default='./data',
        metadata={'help': 'Path for train and eval dataset(s)'}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: int = field(
        default=10,
        metadata={
            "help": "The maximum total target sequence length. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class T2TDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        '''
        Prepares labels from target_ids, and returns examples with keys as
        expected by the forward method. This is necessary because the Trainer
        directly passes this dict as arguments to the model.
        '''
        input_ids = torch.stack([torch.LongTensor(example.input_ids) for example in batch])
        rumi_input_ids = torch.stack([torch.LongTensor(example.rumi_input_ids) for example in batch])
        rumi_attention_mask= torch.stack([torch.LongTensor(example.rumi_attention_mask) for example in batch])
        labels = torch.stack([torch.LongTensor(example.labels) for example in batch])
        # Do not calculate loss for pad tokens. All labels set to -100 are ignored (masked).
        attention_mask = torch.stack([torch.LongTensor(example.attention_mask) for example in batch])
        decoder_attention_mask = torch.stack([torch.LongTensor(example.decoder_attention_mask) for example in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'decoder_attention_mask': decoder_attention_mask,
            'rumi_input_ids': rumi_input_ids,
            'rumi_attention_mask': rumi_attention_mask
        }

def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0   # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0

def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")

# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)

def main():
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    # Load the arguments from a json file
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.'
        )

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        'Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s',
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info('Training/evaluation parameters: %s\n', training_args)

    logger.info('Model/generation parameters: %s\n', model_args)

    # Set seed
    set_seed(training_args.seed)


    config = T5Config.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                        finetuning_task=data_args.task_name,
                                        cache_dir=model_args.cache_dir,)
    # Load pretrained model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # model = T5ForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path,
    #     config = config,
    #     cache_dir=model_args.cache_dir
    # )
    model = RumiT5.from_pretrained(
        model_args.model_name_or_path,
        config = config,
        cache_dir=model_args.cache_dir
    )
    train_dataset = (
        MultipleChoiceDatasetWithSeq2Seq(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            max_target_length=data_args.max_target_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        MultipleChoiceDatasetWithSeq2Seq(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            max_target_length=data_args.max_target_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    logger.info('Finished loading dataset(s)')

    # def compute_metrics(p: EvalPrediction) -> Dict:
    #     import pdb
    #     pdb.set_trace()
    #     preds = np.argmax(p.predictions, axis=1)
    #     return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=T2TDataCollator()
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # Also re-save the tokenizer to the same directory.
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:

        logger.info('*** Evaluate ***')
        accuracy = []
        predictions = []


        model = trainer.model
        model.eval()
        eval_dataloader = trainer.get_eval_dataloader(eval_dataset)

        result = trainer.evaluate()

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc='Calculate Acc (Unified Generation)'):
                prediction = model.generate(
                    input_ids = batch['input_ids'].to(trainer.args.device),
                    attention_mask = batch['attention_mask'].to(trainer.args.device),
                    max_length = data_args.max_target_length,
                    num_beams=model_args.num_beams,
                    early_stopping=model_args.early_stopping,
                    no_repeat_ngram_size=model_args.no_repeat_ngram_size,
                    length_penalty=model_args.length_penalty,
                    top_k=model_args.top_k,
                    top_p=model_args.top_p,
                    repetition_penalty=model_args.repetition_penalty,
                    rumi_input_ids = batch['rumi_input_ids'].to(trainer.args.device),
                    rumi_attention_mask = batch['rumi_attention_mask'].to(trainer.args.device),
                )

                prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in prediction]
                target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['labels']]

                predictions.extend(prediction)


        assert len(predictions) == len(eval_dataset)

        for i, example in tqdm(enumerate(eval_dataset), desc='Calculate Acc'):

            prediction = predictions[i].replace("\n", "").strip()
            gold_answer = example.target_texts

            input_split = example.input_texts.split("\\n")


            candidates_string = input_split[1].strip().lower()

            regex = re.compile("\([a-e]\)")

            candidates_split = regex.split(candidates_string)
            candidates_split = [x.strip() for x in candidates_split if len(x.strip()) > 0]

            # print(f"{prediction} <-> {candidates_split}")
            scores = [score_string_similarity(x, prediction.lower()) for x in candidates_split]
            max_idx = np.argmax(scores)
            try:
                gold = candidates_split.index(gold_answer.lower())
            except:
                print('Error exists in : candidate [%s], gold_answer [%s]' %(candidates_split, gold_answer))


            if max_idx == gold:
                accuracy.append(1)
            else:
                accuracy.append(0)

        accuracy = 1.0 * sum(accuracy) / len(accuracy)

        output_eval_file = os.path.join(training_args.output_dir, 'eval_results.txt')
        if trainer.is_world_process_zero():
            with open(output_eval_file, 'w') as writer:
                logger.info('***** Eval results *****')
                for key in sorted(result.keys()):
                    logger.info('  %s = %s', key, str(result[key]))
                    writer.write('%s = %s\n' % (key, str(result[key])))
                logger.info('eval_acc = %f\n' % accuracy)
                writer.write('eval_acc = %f\n' % accuracy)

                results.update(result)

    return results


if __name__ == '__main__':
    main()
