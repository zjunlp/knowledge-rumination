import torch
import numpy as np
import os
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from transformers import (TrainingArguments, HfArgumentParser,EvalPrediction, set_seed, Trainer,
                            RobertaConfig,RobertaTokenizer, RobertaForMultipleChoice,
                            DebertaConfig, DebertaTokenizer,
                            BartConfig, BartTokenizer,
                            T5Config, T5Tokenizer)
from model import RumiBART, RumiRoBERTa, RumiT5, DebertaForMultipleChoice
from utils_commonsense import MultipleChoiceDataset, Split, processors
import logging

logger = logging.getLogger(__name__)

# MODEL_CLASSES = {
#     'roberta': (RobertaConfig, RumiRoBERTa, RobertaTokenizer),
#     'bart': (BartConfig, RumiBART, BartTokenizer),
#     't5': (T5Config, RumiT5, T5Tokenizer)
# }
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),
    'deberta': (DebertaConfig, DebertaForMultipleChoice, DebertaTokenizer),
    'bart': (BartConfig, RumiBART, BartTokenizer),
    't5': (T5Config, RumiT5, T5Tokenizer)
}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_type: str= field(
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    # answer_model_path: str = field(
    #     metadata={"help": "Path to pretrained model for the model to answer the question"}
    # )
    # rumi_model_path: str = field(
    #     metadata={"help": "Path to pretrained model for the model to ruminate information"}
    # )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model for the model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    rumi_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for the rumi information."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def train(args):
    model = RumiRoBERTa(args)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)
    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    config = config_class.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                        num_labels=num_labels,
                                        finetuning_task=data_args.task_name,
                                        cache_dir=model_args.cache_dir,)
    tokenizer = tokenizer_class.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    

    train_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            rumi_length=None,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            rumi_length=None,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    test_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            rumi_length=None,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        if training_args.do_predict
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

    if training_args.do_predict:
        logger.info("*** Test ***")
        result = trainer.predict(test_dataset).metrics
        output_eval_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

        for dir in os.listdir(training_args.output_dir):
            if dir.startswith('checkpoint-'):
                logger.info("*** Test Saved Checkpoint ***")
                trainer.model.load_state_dict(
                    torch.load(os.path.join(training_args.output_dir, dir, 'pytorch_model.bin')))
                results = trainer.predict(test_dataset)
                result = results.metrics
                # predictions = results.predictions
                # label_ids = results.label_ids.tolist()
                # prediction_ids = np.argmax(predictions, axis=1).tolist()
                # assert len(label_ids) == len(prediction_ids)
                # with open(os.path.join(training_args.output_dir, 'case_analysis.csv'), 'w', encoding='utf-8') as f:
                #     for prediction_id, label_id in zip(prediction_ids, label_ids):
                #         f.write(str(prediction_id) + ',' + str(label_id) + '\n')
                output_eval_file = os.path.join(training_args.output_dir, "test_results2.txt")
                if trainer.is_world_process_zero():
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Test results *****")
                        for key, value in result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))

    return results



if __name__ == "__main__":
    main()