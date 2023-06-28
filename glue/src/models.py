"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
from copy import deepcopy
import transformers
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead, RobertaPreTrainedModel, RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput

import logging
logger = logging.getLogger(__name__)

def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repead_idx = [1] * a.dim()
    repead_idx[dim] = n_tile
    a = a.repeat(*(repead_idx))
    order_index = torch.cuda.LongTensor(torch.cat([ init_dim * torch.arange(n_tile,device=a.device)+i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output



class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output

class RumiRobertaConfig(RobertaConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel` or a
    :class:`~transformers.TFRobertaModel`. It is used to instantiate a RoBERTa model according to the specified
    arguments, defining the model architecture.


    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.

    Examples::

        >>> from transformers import RobertaConfig, RobertaModel

        >>> # Initializing a RoBERTa configuration
        >>> configuration = RobertaConfig()

        >>> # Initializing a model from the configuration
        >>> model = RobertaModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "roberta"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.ffn_layer = [23]
        self.inject_ffn = True
        self.inject_concat = False
        self.few_shot_type = "prompt"


class PrefixEncoder(torch.nn.Module):
    def __init__(self,
        num_layers,
        n_embd,
        prefix_seq_len = 5,
        mid_dim = 512,
    ):
        super().__init__()
        '''
        num_layers: 模型层数
        n_embd: 模型的hidden_state的维度
        prefix_seq_len:
        mid_dim:
        '''
        self.n_embd = n_embd
        self.num_layers = num_layers
        self.mid_dim = mid_dim
        self.prefix_seq_len = prefix_seq_len
        self.embedding = torch.nn.Embedding(self.prefix_seq_len, self.n_embd)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(self.n_embd, self.mid_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.mid_dim, self.num_layers * 2 * self.n_embd)
        )

    def forward(self, prefix):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        return past_key_values


class RumiRoBERTa(RobertaPreTrainedModel):
    config_class = RumiRobertaConfig
    def __init__(self, config: RumiRobertaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.few_shot_type = config.few_shot_type
        rumi_config = deepcopy(config)
        rumi_config.ffn_layer = []
        rumi_config.inject_ffn = False
        rumi_config.inject_concat = False
        self.rumi_model = RobertaModel(rumi_config)
        if config.inject_concat:
            self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.projection = None
        if config.inject_ffn:
            print("inject_ffn:",config.ffn_layer)
        elif config.inject_concat:
            print("inject_concat")
        self.answer_model = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)

        # For regression
        self.lb = None
        self.ub = None

        self.return_full_softmax = None

        # param for prefix
        self.num_layers = config.num_hidden_layers
        self.num_head = config.num_attention_heads
        self.n_embd = config.hidden_size
        self.match_n_embd = self.n_embd // self.num_head
        self.prefix_seq_len = 15
        self.prefix_encoder = PrefixEncoder(self.num_layers, self.n_embd, self.prefix_seq_len)
        self.prefix_tokens = torch.arange(self.prefix_seq_len).long()

        self.init_weights()

        self.freeze_parameters()

    def freeze_parameters(self):
        for name, param in self.rumi_model.named_parameters():
            param.requires_grad = False

    def get_prompt(self, batch_size, device):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        self.prefix_encoder.embedding.to(device)
        self.prefix_encoder.trans.to(device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            self.num_layers * 2,
            self.num_head,
            self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mask_pos=None,
            labels=None,
            rumi_ids=None,
            rumi_attention_mask=None,
            rumi_info_mask=None,
            return_dict=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)
        """

        # prefix生成部分代码
        batch_size = rumi_ids.shape[0]
        device = rumi_ids.device
        past_key_values = self.get_prompt(batch_size=batch_size, device=device)
        prefix_attention_mask = torch.ones(batch_size, self.prefix_seq_len).to(device)
        rumi_attention_mask = torch.cat((prefix_attention_mask, rumi_attention_mask), dim=1)


        rumi_outputs = self.rumi_model(
            rumi_ids,
            attention_mask=rumi_attention_mask,
            past_key_values=past_key_values,  # 传入prefix训练参数
        )

        rumi_sequence_output = rumi_outputs[0]

        # rumi_length = torch.sum(rumi_info_mask[0])
        rumi_info_mask = rumi_info_mask.unsqueeze(-1).type_as(rumi_sequence_output).bool()
        # rumi_info_mask = rumi_info_mask.view(-1, rumi_info_mask.size(-1)).unsqueeze(-1).type_as(rumi_sequence_output).bool()
        # print("runi_info_mask_shape:", rumi_info_mask)
        rumi_info = torch.masked_select(rumi_sequence_output, rumi_info_mask)
        # print("runi_info_shape:", rumi_info.shape)
        rumi_info = rumi_info.view(rumi_sequence_output.size(0), -1, rumi_sequence_output.size(-1))
        # print(rumi_info.shape)
        # print(rumi_info.shape)
        # rumi_info = self.projection(rumi_info)
        # print("runi_info_shape:", rumi_info.shape)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.answer_model(
            input_ids,
            attention_mask=attention_mask,
            rumi_info=rumi_info,
        )

        if "prompt" in self.few_shot_type:
                    # Get <mask> token representation
            sequence_output, pooled_output = outputs[:2]
            sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

            # Logits over vocabulary tokens
            prediction_mask_scores = self.lm_head(sequence_mask_output)

            # Exit early and only return mask logits.
            if self.return_full_softmax:
                if labels is not None:
                    return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
                return prediction_mask_scores

            # Return logits for each label
            logits = []
            for label_id in range(len(self.label_word_list)):
                logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            logits = torch.cat(logits, -1)

            # Regression task
            if self.config.num_labels == 1:
                logsoftmax = nn.LogSoftmax(-1)
                logits = logsoftmax(logits) # Log prob of right polarity

            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    # Regression task
                    loss_fct = nn.KLDivLoss(log_target=True)
                    labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                    loss = loss_fct(logits.view(-1, 2), labels)
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            output = (logits,)
            if self.num_labels == 1:
                # Regression output
                output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
            return ((loss,) + output) if loss is not None else output
    
        else:
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)

            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
