from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import warnings
from torch.nn import CrossEntropyLoss
from transformers import (T5Model,T5PreTrainedModel,T5Config,
                          BartModel, BartPretrainedModel,
                          RobertaModel,RobertaConfig,DebertaConfig,
                          BartConfig)
from transformers.modeling_outputs import MultipleChoiceModelOutput,Seq2SeqLMOutput,BaseModelOutput,ModelOutput
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaModel, ContextPooler, StableDropout
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repead_idx = [1] * a.dim()
    repead_idx[dim] = n_tile
    a = a.repeat(*(repead_idx))
    order_index = torch.cuda.LongTensor(torch.cat([ init_dim * torch.arange(n_tile,device=a.device)+i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

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

class DebertaForMultipleChoice(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, 1)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.deberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RumiDebertaConfig(DebertaConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.DebertaModel` or a
    :class:`~transformers.TFDebertaModel`. It is used to instantiate a DeBERTa model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the DeBERTa `microsoft/deberta-base <https://huggingface.co/microsoft/deberta-base>`__
    architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DeBERTa model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.DebertaModel` or
            :class:`~transformers.TFDebertaModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"`, :obj:`"gelu"`, :obj:`"tanh"`, :obj:`"gelu_fast"`,
            :obj:`"mish"`, :obj:`"linear"`, :obj:`"sigmoid"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.DebertaModel` or
            :class:`~transformers.TFDebertaModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        relative_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether use relative position encoding.
        max_relative_positions (:obj:`int`, `optional`, defaults to 1):
            The range of relative positions :obj:`[-max_position_embeddings, max_position_embeddings]`. Use the same
            value as :obj:`max_position_embeddings`.
        pad_token_id (:obj:`int`, `optional`, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (:obj:`List[str]`, `optional`):
            The type of relative position attention, it can be a combination of :obj:`["p2c", "c2p", "p2p"]`, e.g.
            :obj:`["p2c"]`, :obj:`["p2c", "c2p"]`, :obj:`["p2c", "c2p", 'p2p"]`.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """
    model_type = "deberta"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=False,
        max_relative_positions=-1,
        pad_token_id=0,
        position_biased_input=True,
        pos_att_type=None,
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input

        # Backwards compatibility
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act

        '''
        For RumiDeberta configuration
        '''
        self.ffn_layer = []
        self.inject_ffn = False
        self.inject_concat = True




class RumiDeBERTa(DebertaPreTrainedModel):
    config_class = RumiDebertaConfig

    def __init__(self, config: RumiDebertaConfig):
        super().__init__(config)
        rumi_config = deepcopy(config)
        rumi_config.ffn_layer = []
        rumi_config.inject_ffn = False
        rumi_config.inject_concat = False
        self.rumi_model = DebertaModel(rumi_config)
        if config.inject_concat:
            self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.projection = None
        self.answer_model = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, 1)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

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
            rumi_ids=None,
            rumi_attention_mask=None,
            rumi_token_type_ids=None,
            rumi_info_mask=None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # prefix生成部分代码
        # batch_size = rumi_ids.shape[0]
        # rumi_num_choices = rumi_ids.shape[1]
        # device = rumi_ids.device
        # past_key_values = self.get_prompt(batch_size=batch_size * rumi_num_choices, device=device)
        # prefix_attention_mask = torch.ones(batch_size * rumi_num_choices, self.prefix_seq_len).to(device)

        flat_rumi_ids = rumi_ids.view(-1, rumi_ids.size(-1)) if rumi_ids is not None else None
        flat_rumi_token_type_ids = rumi_token_type_ids.view(-1, rumi_token_type_ids.size(-1)) if rumi_token_type_ids is not None else None
        flat_rumi_attention_mask = rumi_attention_mask.view(-1, rumi_attention_mask.size(
            -1)) if rumi_attention_mask is not None else None
        # rumi_attention_mask = torch.cat((prefix_attention_mask, flat_rumi_attention_mask), dim=1)
        rumi_outputs = self.rumi_model(
            flat_rumi_ids,
            attention_mask=flat_rumi_attention_mask,
            token_type_ids=flat_rumi_token_type_ids,
            # past_key_values=past_key_values,  # 传入prefix训练参数
        )
        rumi_sequence_output = rumi_outputs[0]
        # rumi_length = torch.sum(rumi_info_mask[0])
        # rumi_info_mask = rumi_info_mask.unsqueeze(-1).type_as(rumi_sequence_output).bool()
        rumi_info_mask = rumi_info_mask.view(-1, rumi_info_mask.size(-1)).unsqueeze(-1).type_as(
            rumi_sequence_output).bool()
        # print("runi_info_mask_shape:", rumi_info_mask)
        rumi_info = torch.masked_select(rumi_sequence_output, rumi_info_mask)
        # print("runi_info_shape:", rumi_info.shape)
        rumi_info = rumi_info.view(rumi_sequence_output.size(0), -1, rumi_sequence_output.size(-1))
        # print(rumi_info.shape)
        # rumi_info = tile(rumi_info,dim=0,n_tile=input_ids.size(1))
        # print(rumi_info.shape)
        if self.projection is not None:
            rumi_info = self.projection(rumi_info)
        # rumi_info_k = self.projection1(rumi_info)
        # rumi_info_v = self.projection2(rumi_info)
        # rumi_info = (rumi_info_k,rumi_info_v)
        # print("runi_info_shape:", rumi_info.shape)
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.answer_model(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rumi_info=rumi_info,
        )
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
        self.ffn_layer = []
        self.inject_ffn = False
        self.inject_concat = True

class RumiRoBERTa(RobertaPreTrainedModel):
    config_class = RumiRobertaConfig
    def __init__(self, config: RumiRobertaConfig):
        super().__init__(config)
        rumi_config = deepcopy(config)
        rumi_config.ffn_layer = [23]
        rumi_config.inject_ffn = False
        rumi_config.inject_concat = True
        self.rumi_model = RobertaModel(rumi_config)
        if config.inject_concat:
            self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.projection = None
        self.answer_model = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        #param for prefix
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

    def get_rumi_info(
        self,
        rumi_ids=None,
        rumi_attention_mask=None,
        rumi_info_mask=None,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # prefix生成部分代码
        batch_size = rumi_ids.shape[0]
        rumi_num_choices = rumi_ids.shape[1]
        device = rumi_ids.device
        past_key_values = self.get_prompt(batch_size=batch_size * rumi_num_choices, device=device)
        prefix_attention_mask = torch.ones(batch_size * rumi_num_choices, self.prefix_seq_len).to(device)

        flat_rumi_ids = rumi_ids.view(-1, rumi_ids.size(-1)) if rumi_ids is not None else None
        flat_rumi_attention_mask = rumi_attention_mask.view(-1, rumi_attention_mask.size(
            -1)) if rumi_attention_mask is not None else None
        rumi_attention_mask = torch.cat((prefix_attention_mask, flat_rumi_attention_mask), dim=1)
        rumi_outputs = self.rumi_model(
            flat_rumi_ids,
            attention_mask=rumi_attention_mask,
            past_key_values=past_key_values,  # 传入prefix训练参数
        )
        rumi_sequence_output = rumi_outputs[0]
        # rumi_length = torch.sum(rumi_info_mask[0])
        # rumi_info_mask = rumi_info_mask.unsqueeze(-1).type_as(rumi_sequence_output).bool()
        rumi_info_mask = rumi_info_mask.view(-1, rumi_info_mask.size(-1)).unsqueeze(-1).type_as(
            rumi_sequence_output).bool()
        # print("runi_info_mask_shape:", rumi_info_mask)
        rumi_info = torch.masked_select(rumi_sequence_output, rumi_info_mask)
        # print("runi_info_shape:", rumi_info.shape)
        rumi_info = rumi_info.view(rumi_sequence_output.size(0), -1, rumi_sequence_output.size(-1))

        return rumi_info

    def get_avg_rumi_embedding(
        self,
        rumi_ids=None,
        rumi_attention_mask=None,
        rumi_info_mask=None,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # prefix生成部分代码
        batch_size = rumi_ids.shape[0]
        device = rumi_ids.device
        past_key_values = self.get_prompt(batch_size=batch_size, device=device)
        prefix_attention_mask = torch.ones(batch_size, self.prefix_seq_len).to(device)

        flat_rumi_ids = rumi_ids.view(-1, rumi_ids.size(-1)) if rumi_ids is not None else None
        flat_rumi_attention_mask = rumi_attention_mask.view(-1, rumi_attention_mask.size(
            -1)) if rumi_attention_mask is not None else None
        rumi_attention_mask = torch.cat((prefix_attention_mask, flat_rumi_attention_mask), dim=1)
        rumi_outputs = self.rumi_model(
            flat_rumi_ids,
            attention_mask=rumi_attention_mask,
            past_key_values=past_key_values,  # 传入prefix训练参数
        )
        rumi_sequence_output = rumi_outputs[0]
        # rumi_length = torch.sum(rumi_info_mask[0])
        # rumi_info_mask = rumi_info_mask.unsqueeze(-1).type_as(rumi_sequence_output).bool()
        rumi_info_mask = rumi_info_mask.view(-1, rumi_info_mask.size(-1)).unsqueeze(-1).type_as(
            rumi_sequence_output).bool()
        # print("runi_info_mask_shape:", rumi_info_mask)
        rumi_info = torch.masked_select(rumi_sequence_output, rumi_info_mask)
        # print("runi_info_shape:", rumi_info.shape)
        rumi_info = rumi_info.view(rumi_sequence_output.size(0), -1, rumi_sequence_output.size(-1))

        return rumi_info

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
        rumi_ids=None,
        rumi_attention_mask=None,
        rumi_info_mask=None,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # prefix生成部分代码
        batch_size = rumi_ids.shape[0]
        rumi_num_choices = rumi_ids.shape[1]
        device = rumi_ids.device 
        past_key_values = self.get_prompt(batch_size=batch_size * rumi_num_choices, device=device)
        prefix_attention_mask = torch.ones(batch_size * rumi_num_choices, self.prefix_seq_len).to(device)
        
        flat_rumi_ids = rumi_ids.view(-1, rumi_ids.size(-1)) if rumi_ids is not None else None
        flat_rumi_attention_mask = rumi_attention_mask.view(-1, rumi_attention_mask.size(-1)) if rumi_attention_mask is not None else None
        rumi_attention_mask = torch.cat((prefix_attention_mask, flat_rumi_attention_mask), dim=1)
        rumi_outputs = self.rumi_model(
            flat_rumi_ids,
            attention_mask=rumi_attention_mask,
            past_key_values=past_key_values, #传入prefix训练参数
            )
        rumi_sequence_output = rumi_outputs[0]
        # rumi_length = torch.sum(rumi_info_mask[0])
        # rumi_info_mask = rumi_info_mask.unsqueeze(-1).type_as(rumi_sequence_output).bool()
        rumi_info_mask = rumi_info_mask.view(-1, rumi_info_mask.size(-1)).unsqueeze(-1).type_as(rumi_sequence_output).bool()
        # print("runi_info_mask_shape:", rumi_info_mask)
        rumi_info = torch.masked_select(rumi_sequence_output,rumi_info_mask)
        # print("runi_info_shape:", rumi_info.shape)
        rumi_info = rumi_info.view(rumi_sequence_output.size(0), -1, rumi_sequence_output.size(-1))
        # print(rumi_info.shape)
        # rumi_info = tile(rumi_info,dim=0,n_tile=input_ids.size(1))
        # print(rumi_info.shape)
        if self.projection is not None:
            rumi_info=self.projection(rumi_info)
        # rumi_info_k = self.projection1(rumi_info)
        # rumi_info_v = self.projection2(rumi_info)
        # rumi_info = (rumi_info_k,rumi_info_v)
        # print("runi_info_shape:", rumi_info.shape)
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.answer_model(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rumi_info=rumi_info,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RumiBART(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.rumi_model = BartModel(config)
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.answer_model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):

        if is_training:
            decoder_start_token_id = self.config.decoder_start_token_id
            _decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            _decoder_input_ids[..., 1:] = decoder_input_ids[..., :-1].clone()
            _decoder_input_ids[..., 0] = decoder_start_token_id
        else:
            _decoder_input_ids = decoder_input_ids.clone()

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            loss_fct = nn.CrossEntropyLoss(reduce=False)
            losses = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              decoder_input_ids.view(-1))
            loss = torch.sum(losses * decoder_attention_mask.float().view(-1))
            return loss
        return (lm_logits, ) + outputs[1:]


class RumiT5(T5PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.rumi_encoder = T5Stack(encoder_config, self.shared)
        self.encoder = T5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.rumi_decoder = T5Stack(decoder_config, self.shared)
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.init_weights()
        self.freeze_parameters()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
    
    def freeze_parameters(self):
        for name, param in self.rumi_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.rumi_decoder.named_parameters():
            param.requires_grad = False
    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def get_rumi_info(
        self,
        rumi_input_ids=None,
        rumi_attention_mask=None,
        rumi_past_key_values=None,
        use_cache=None,
        return_dict=None
    ):
        rumi_info = None
        rumi_encoder_outputs = self.rumi_encoder(
            input_ids=rumi_input_ids,
            attention_mask=rumi_attention_mask,
        )
        rumi_hidden_states = rumi_encoder_outputs[0]
        rumi_decoder_input_ids = self._prepare_decoder_input_ids_for_generation(
                        rumi_input_ids,
                    )
        cur_len = 0
        while cur_len < 10:
            if rumi_past_key_values is not None:
                rumi_decoder_inputs_embeds = rumi_decoder_inputs_embeds[:, -1:]
                rumi_decoder_input_ids = None
            else:
                rumi_decoder_inputs_embeds = None

            # Decode
            rumi_decoder_outputs = self.rumi_decoder(
                input_ids=rumi_decoder_input_ids,
                # attention_mask=decoder_attention_mask,
                inputs_embeds=rumi_decoder_inputs_embeds,
                past_key_values=rumi_past_key_values,
                encoder_hidden_states=rumi_hidden_states,
                encoder_attention_mask=rumi_attention_mask,
                use_cache=use_cache,
                return_dict=return_dict
            )
            # print(rumi_decoder_outputs[0].shape)
            rumi_decoder_inputs_embeds = rumi_decoder_outputs[0]
            rumi_decoder_inputs_embeds = self.projection(rumi_decoder_inputs_embeds)
            rumi_past_key_values = rumi_decoder_outputs[1]
            if rumi_info is None:
                rumi_info = rumi_decoder_inputs_embeds
            else:
                rumi_info = torch.cat([rumi_info,rumi_decoder_inputs_embeds],dim=1)
            cur_len = cur_len + 1
        return rumi_info
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        rumi_input_ids=None,
        rumi_attention_mask=None,
        rumi_past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        rumi_info = None
        # Encode if needed (training, first prediction pass)
        if rumi_input_ids is not None:
            rumi_info = self.get_rumi_info(
                rumi_input_ids=rumi_input_ids,
                rumi_attention_mask=rumi_attention_mask,
                rumi_past_key_values=rumi_past_key_values,
                use_cache=use_cache,
                return_dict=return_dict
            )
            rumi_info_mask = torch.ones((rumi_info.shape[0],rumi_info.shape[1]),device=input_ids.device)
            attention_mask = torch.cat([rumi_info_mask,attention_mask],dim=1)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                rumi_info=rumi_info,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0]
        # print(hidden_states.shape)
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        
       
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        # retrieve encoder hidden states
        rumi_info = self.get_rumi_info(rumi_input_ids=model_kwargs['rumi_input_ids'],rumi_attention_mask=model_kwargs['rumi_attention_mask'])
        model_kwargs.pop('rumi_input_ids')
        model_kwargs.pop('rumi_attention_mask')
        attention_mask = model_kwargs['attention_mask']
        rumi_info_mask = torch.ones((rumi_info.shape[0],rumi_info.shape[1]),device=input_ids.device)
        attention_mask = torch.cat([rumi_info_mask,attention_mask],dim=1)
        model_kwargs['rumi_info'] = rumi_info
        model_kwargs['attention_mask'] = attention_mask
        # model_kwargs.update('attention_mask',attention_mask)
        encoder = self.get_encoder()
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
        }
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        # print(model_kwargs.keys())
        return model_kwargs

