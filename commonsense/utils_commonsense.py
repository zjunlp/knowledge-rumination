# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """


import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available, DebertaTokenizer
import uuid


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    endings: List[str]
    concept: Optional[str]
    label: Optional[str]
    contexts: Optional[List[str]]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    rumi_ids: Optional[List[List[int]]]
    rumi_attention_mask: Optional[List[List[int]]]
    rumi_info_mask: Optional[List[List[int]]]
    rumi_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]

@dataclass(frozen=True)
class InputFeaturesWithSeq2Seq:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_texts: List[str]
    target_texts: List[str]
    input_ids: List[List[int]]
    rumi_input_ids: Optional[List[List[int]]]
    attention_mask: Optional[List[List[int]]]
    rumi_attention_mask: Optional[List[List[int]]]
    labels: Optional[List[List[int]]]
    decoder_attention_mask: Optional[List[List[int]]]

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MultipleChoiceDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            rumi_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()

            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                ),
            )
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    label_list = processor.get_labels()
                    if mode == Split.dev:
                        examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    else:
                        examples = processor.get_train_examples(data_dir)
                    logger.info("Training examples: %s", len(examples))
                    self.features = convert_examples_to_features(
                        task,
                        examples,
                        label_list,
                        max_seq_length,
                        tokenizer,
                        rumi_length = rumi_length,
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

    class MultipleChoiceDatasetWithSeq2Seq(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeaturesWithSeq2Seq]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            max_target_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()

            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    str(max_target_length),
                    task,
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    label_list = processor.get_labels()
                    if mode == Split.dev:
                        examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    else:
                        examples = processor.get_train_examples(data_dir)
                    logger.info("Training examples: %s", len(examples))
                    self.features = convert_examples_to_features_with_seq2seq(
                        examples,
                        task,
                        max_seq_length,
                        max_target_length,
                        tokenizer,
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeaturesWithSeq2Seq:
            return self.features[i]


if is_tf_available():
    import tensorflow as tf

    class TFMultipleChoiceDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = 128,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()

            logger.info(f"Creating features from dataset file at {data_dir}")
            label_list = processor.get_labels()
            if mode == Split.dev:
                examples = processor.get_dev_examples(data_dir)
            elif mode == Split.test:
                examples = processor.get_test_examples(data_dir)
            else:
                examples = processor.get_train_examples(data_dir)
            logger.info("Training examples: %s", len(examples))

            self.features = convert_examples_to_features(
                task,
                examples,
                label_list,
                max_seq_length,
                tokenizer,
            )

            def gen():
                for (ex_index, ex) in tqdm.tqdm(enumerate(self.features), desc="convert examples to features"):
                    if ex_index % 10000 == 0:
                        logger.info("Writing example %d of %d" % (ex_index, len(examples)))

                    yield (
                        {
                            "example_id": 0,
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        },
                        ex.label,
                    )

            self.dataset = tf.data.Dataset.from_generator(
                gen,
                (
                    {
                        "example_id": tf.int32,
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                        "token_type_ids": tf.int32,
                    },
                    tf.int64,
                ),
                (
                    {
                        "example_id": tf.TensorShape([]),
                        "input_ids": tf.TensorShape([None, None]),
                        "attention_mask": tf.TensorShape([None, None]),
                        "token_type_ids": tf.TensorShape([None, None]),
                    },
                    tf.TensorShape([]),
                ),
            )

        def get_dataset(self):
            self.dataset = self.dataset.apply(tf.data.experimental.assert_cardinality(len(self.features)))

            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class CommonsenseProcessor(DataProcessor):
    """Processor for the Commonsense data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        examples = []
        for line in tqdm.tqdm(lines, desc="read coqa data"):
            data_raw = json.loads(line.strip("\n"))
            if "answerKey" in data_raw:
                truth = ord(data_raw["answerKey"]) - ord("A")
            else:
                truth = None
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            concept = question_choices['question_concept']
            id = data_raw["id"]
            options = question_choices["choices"]
            assert len(options) == 5
            examples.append(
                InputExample(
                    example_id=id,
                    question=question,
                    concept=concept,
                    contexts=["", "", "", "", ""],
                    endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"], options[4]["text"]],
                    label=truth,
                )
            )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))

        return examples

class SocialIQAProcessor(DataProcessor):
    """Processor for the SocialIQA data set from YJ."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev", data_dir)

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test", data_dir)

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def read_labels(self, file):
        labels = []
        with open(file, 'r') as f:
            for line in f:
                labels.append(line.strip())
        return labels

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines
    def _create_examples(self, lines, type, data_dir):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = self.read_labels(os.path.join(data_dir, f'{type}-labels.lst'))

        assert len(labels) == len(lines)

        for i, line in tqdm.tqdm(enumerate(lines), desc="read social_iqa data"):
            data_raw = json.loads(line.strip("\n"))

            label = int(labels[i])
            examples.append(
                InputExample(
                example_id= str(uuid.uuid1),
                question=data_raw['question'],
                concept='',
                contexts=[data_raw['context'], data_raw['context'], data_raw['context']],
                endings=[data_raw['answerA'], data_raw['answerB'], data_raw['answerC']],
                label=label - 1,
                )
            )

        return examples


class PIQAProcessor(DataProcessor):
    """Processor for the PIQA data set from YJ."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "valid.jsonl")), "valid", data_dir)

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test", data_dir)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def read_labels(self, file):
        labels = []
        with open(file, 'r') as f:
            for line in f:
                labels.append(line.strip())
        return labels

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines
    def _create_examples(self, lines, type, data_dir):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = self.read_labels(os.path.join(data_dir, f'{type}-labels.lst'))

        assert len(labels) == len(lines)

        for i, line in tqdm.tqdm(enumerate(lines), desc="read piqa data"):
            data_raw = json.loads(line.strip("\n"))

            label = int(labels[i])
            examples.append(
                InputExample(
                example_id= str(uuid.uuid1),
                question=data_raw['goal'],
                concept='',
                contexts=["", ""],
                endings=[data_raw['sol1'], data_raw['sol2']],
                label=label,
                )
            )

        return examples

class WinograndeProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.jsonl")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.jsonl")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.jsonl")))

    def get_labels(self):
        return [0, 1]
    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines
    def _create_examples(self, records):
        examples = []
        for (i, record) in enumerate(records):
            record = json.loads(record.strip("\n"))

            guid = record['qID']
            sentence = record['sentence']

            name1 = record['option1']
            name2 = record['option2']
            if not 'answer' in record:
                # This is a dummy label for test prediction.
                # test.jsonl doesn't include the `answer`.
                label = 0
            else:
                label = int(record['answer']) - 1

            conj = "_"
            idx = sentence.index(conj)
            context = sentence[:idx]
            option_str = "_ " + sentence[idx + len(conj):].strip()

            option1 = option_str.replace("_", name1)
            option2 = option_str.replace("_", name2)

            mc_example = InputExample(
                example_id=guid,
                question="",
                contexts=[context, context],
                concept="",
                endings=[option1, option2],
                label=label
            )
            examples.append(mc_example)
        for e in examples[:2]:
            logger.info("*** Example ***")
            logger.info("example: %s" % e)
        return examples

class AnliProcessor():
    """Processor for the ANLI data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev", data_dir)

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test", data_dir)
    def get_labels(self):
        """See base class."""
        return [0, 1]
    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines
    def read_labels(self, file):
        labels = []
        with open(file, 'r') as f:
            for line in f:
                labels.append(line.strip())
        return labels
    def _create_examples(self, records, type, data_dir):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = self.read_labels(os.path.join(data_dir, f'{type}-labels.lst'))

        assert len(labels) == len(records)

        for (i, record) in enumerate(records):
            record = json.loads(record.strip("\n"))
            guid = "%s" % (record['story_id'])

            beginning = record['obs1']
            ending = record['obs2']

            option1 = record['hyp1']
            option2 = record['hyp2']
            label = int(labels[i])

            examples.append(
                InputExample(
                    example_id=guid,
                    question=beginning,
                    contexts=[ending, ending],
                    concept="",
                    endings=[option1, option2],
                    label=label - 1,
                )
            )
        return examples

class OBQAProcessor(DataProcessor):
    """Processor for the OpenBook QA (OBQA) data set."""
    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, 'train.jsonl')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, 'dev.jsonl')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, 'test.jsonl')), 'test')

    def get_labels(self):
        return [0, 1, 2, 3]

    def _create_examples(self, lines, set_type):
        examples = []
        for line in tqdm.tqdm(lines, desc="read obqa data"):
            data_raw = json.loads(line.strip("\n"))
            if "answerKey" in data_raw:
                truth = ord(data_raw["answerKey"]) - ord("A")
            else:
                truth = None
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            assert len(options) == 4
            examples.append(
                InputExample(
                    example_id=id,
                    question=question,
                    concept="",
                    contexts=["", "", "", ""],
                    endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                    label=truth,
                )
            )

        if set_type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))

        return examples

class HellaSWAGProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(os.path.join(data_dir, "train.jsonl"))
    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(os.path.join(data_dir, "val.jsonl"))
    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(os.path.join(data_dir, "test.jsonl"))
    def get_labels(self):
        return [0, 1, 2, 3]
    def _create_examples(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = json.loads(line)
                data.append(line)

        examples = [
            InputExample(
                example_id=str(dt["ind"]),
                question=dt["ctx"],
                concept="",
                contexts=['']*4,
                endings=dt["endings"],
                label=dt["label"],
            )
            for idx, dt in enumerate(data)
        ]
        return examples

def convert_examples_to_features(
    task: str,
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    rumi_length: int,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    mask_id = tokenizer._convert_token_to_id(tokenizer.mask_token)
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        # question =  example.question
        # rumi_prompt = question + ' <\s> ' + 'as far as I know,' + ' <mask>' * rumi_length
        rumi_prompts = []
        # print(rumi_prompt)
        # rumi_input = tokenizer(
        #     rumi_prompt,
        #     add_special_tokens=True,
        #     max_length=max_length+rumi_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_overflowing_tokens=True,
        # )
        # rumi_ids = rumi_input["input_ids"]
        # rumi_info_mask = [1 if token_id == mask_id else 0 for token_id in rumi_ids]
        # assert sum(rumi_info_mask) == rumi_length
        # # print(rumi_info_mask)
        # rumi_attention_mask = rumi_input["attention_mask"]
        # rumi_token_type_ids = rumi_input["token_type_ids"]
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            if rumi_length != None:
                mask_token = ' [MASK]' if isinstance(tokenizer, DebertaTokenizer) else ' <mask>'
                if task == 'winogrande':
                    ## TODO
                    pass
                elif task == 'socialiqa':
                    # num_similarities = len(example.similarities)
                    # if num_similarities > 0:
                    #     similarty_orders = np.argsort(example.similarities)[::-1]
                    #     max_num_rumi = min(num_similarities, 3)
                    #     num_rumis.append(max_num_rumi)
                    #     rumi_prompts = [example.context + ' ' + example.question + ' <\s> ' \
                    #                   + (f'About {example.mentions[similarty_orders[i]]} I know') + ' <mask>' * rumi_length \
                    #                     for i in range(max_num_rumi)]
                    #
                    # else:
                    #     num_rumis.append(1)
                    #     rumi_prompts = [example.context + ' ' + example.question + ' <\s> ' \
                    #                   + (f'as far as i know') + ' <mask>' * rumi_length]
                    rumi_prompt = example.question + ' <\s> ' + 'As far as i know' + mask_token * rumi_length
                    rumi_prompts.append(rumi_prompt)

                elif task == 'coqa':
                    rumi_prompt = example.question + ' <\s> ' + (f'About {example.concept} and {ending} I know') + mask_token * rumi_length
                    # rumi_prompt = example.question + ' <\s> ' + 'As far as i know' + ' <mask>' * (rumi_length // 2)
                    # rumi_prompt2 = example.question + ' <\s> ' + (f'About {example.concept} I know') + ' <mask>' * (rumi_length - rumi_length // 2)
                    rumi_prompts.append(rumi_prompt)
                    # rumi_prompts.append(rumi_prompt2)
                elif task == 'obqa':
                    rumi_prompt = example.question + ' <\s> ' + 'As far as i know' + mask_token * rumi_length
                    rumi_prompts.append(rumi_prompt)
                elif task == 'piqa':
                    rumi_prompt = example.question + ' <\s> ' + 'As far as i know' + mask_token * rumi_length
                    rumi_prompts.append(rumi_prompt)
                elif task == 'hellaswag':
                    rumi_prompt = example.question + ' <\s> ' + 'As far as i know' + mask_token * rumi_length
                    rumi_prompts.append(rumi_prompt)
                elif task == 'anli':
                    rumi_prompt = example.question + ' '.join([context, ending]) + ' <\s> ' + 'As far as i know' + mask_token * rumi_length
                    rumi_prompts.append(rumi_prompt)
                else:
                    raise NotImplementedError
            if len(context) == 0:
                text_a = "Q: "+ example.question + " A: " + ending
                inputs = tokenizer(
                    text_a,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=True,
                )
            else:
                text_a = context
                if len(example.question) == 0:
                    text_b = ending
                else:
                    text_b = "Q: "+ example.question + " A: " + ending
                inputs = tokenizer(
                    text_a,
                    text_b,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=True,
                )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)
        if rumi_length != None:
            rumi_input = tokenizer(
                rumi_prompts,
                add_special_tokens=True,
                max_length=max_length+rumi_length+20,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
            )
            rumi_ids = rumi_input["input_ids"]
            rumi_info_mask = [[1 if token_id == mask_id else 0 for token_id in token_ids] for token_ids in rumi_ids]

            assert sum(map(sum,rumi_info_mask)) == rumi_length * len(rumi_info_mask)
            # print(rumi_info_mask)
            rumi_attention_mask = rumi_input["attention_mask"] if "attention_mask" in rumi_input else None
            rumi_token_type_ids = rumi_input['token_type_ids'] if "token_type_ids" in rumi_input else None
        # print(label_map)
        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                rumi_ids=rumi_ids if rumi_length != None else None,
                rumi_attention_mask= rumi_attention_mask if rumi_length != None else None,
                rumi_token_type_ids=rumi_token_type_ids if rumi_length != None else None,
                rumi_info_mask= rumi_info_mask if rumi_length != None else None,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features
def format_example_commonsense_qa(example):
    '''
    Formats the commonsense_qa example as below,

    Input
        question: She was always helping at the senior center, it brought her what? \n
        choices: (A) satisfaction (B) heart (C) feel better (D) pay (E) happiness

    Target
        happiness
    '''
    formated_example = dict()

    lables = ['A', 'B', 'C', 'D', 'E']

    choices = ['(%s) %s' % (i, choice) for i, choice in zip(lables, example.endings)]
    formated_example['input_text'] = '%s \\n %s' % (example.question, ' '.join(choices))
    formated_example['rumi_text'] = example.question + ' As far as I know, '

    # Use the following format if you want the target to be the string answer, rather than the alphabetical choice
    formated_example['target_text'] = '%s' % (example.endings[example.label])
    # example['target_text'] = '%s' % example['answerKey']
    formated_example['example_id'] = example.example_id

    return formated_example


def format_example_social_i_qa(example):
    '''
    TODO: modify the format for social_i_qa
    Formats the social_i_qa example as below,

    Input
        question: How would you describe Sydney? context: Sydney walked past a homeless woman
        asking for change but did not have any money they could give to her. Sydney felt bad
        afterwards. options: A: sympathetic B: like a person who was unable to help C: incredulous
    Target
        A: sympathetic
    '''
    formated_example = dict()

    lables = ['A', 'B', 'C']

    assert len(lables) == len(example.endings) == len(example.contexts)

    choices = ['(%s) %s' % (i, choice) for i, choice in zip(lables, example.endings)]
    formated_example['input_text'] = '%s \\n %s \\n %s' % (example.question, ' '.join(choices), example.contexts[0])
    formated_example['rumi_text'] = example.question + ' As far as I know, '

    # Use the following format if you want the target to be the string answer, rather than the alphabetical choice
    formated_example['target_text'] = '%s' % (example.endings[example.label])
    # example['target_text'] = '%s' % example['answerKey']
    formated_example['example_id'] = example.example_id

    return formated_example
def format_example(example, task: str):
    '''
    Forwards the format_example method to the correct implementation for the dataset.
    '''
    if task == 'coqa':
        return format_example_commonsense_qa(example)
    elif task == 'socialiqa':
        return format_example_social_i_qa(example)
    else:
        raise NotImplementedError(f'{task} not Supported yet.')

def convert_examples_to_features_with_seq2seq(
    examples: List[InputExample],
    task: str,
    max_length: int,
    max_target_length: int,
    tokenizer: PreTrainedTokenizer,
):

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        example = format_example(example, task=task)
        input_text = example['input_text']
        target_text = example['target_text']
        example_id = example['example_id']
        rumi_text = example['rumi_text']
        rumi_inputs = tokenizer(
            rumi_text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )
        inputs = tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )
        targets = tokenizer(
            target_text,
            add_special_tokens=True,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )

        features.append(
            InputFeaturesWithSeq2Seq(
                example_id=example_id,
                input_texts= example['input_text'],
                target_texts=example['target_text'],
                input_ids=inputs['input_ids'],
                rumi_input_ids=rumi_inputs['input_ids'],
                rumi_attention_mask=rumi_inputs['attention_mask'],
                attention_mask=inputs['attention_mask'],
                labels=targets['input_ids'],
                decoder_attention_mask=targets['attention_mask']
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


processors = {"coqa":CommonsenseProcessor, 'socialiqa':SocialIQAProcessor, 'piqa':PIQAProcessor, 'winogrande':WinograndeProcessor, 'anli':AnliProcessor, 'obqa':OBQAProcessor, 'hellaswag':HellaSWAGProcessor}
