#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn
from transformers.models.bert import BertConfig, BertModel
from transformers.optimization import AdamW
from transformers.models.bert import BertTokenizer
from transformers.models.roberta import RobertaTokenizer

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM

from dpr.models.biencoder import BiEncoder
from dpr.utils.data_utils import Tensorizer
from .reader import Reader

logger = logging.getLogger(__name__)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    question_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        gradient_checkpointing=cfg.encoder.gradient_checkpointing,
        **kwargs
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    fix_ctx_encoder = cfg.fix_ctx_encoder if hasattr(cfg, "fix_ctx_encoder") else False

    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )
    # logger.info('optimizer use learning_rate:{}'.format(cfg.train.learning_rate))
    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, reader, optimizer


def get_bert_tensorizer(cfg, tokenizer=None):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    if not tokenizer:
        tokenizer = get_bert_tokenizer(
            pretrained_model_cfg
        )
        if cfg.special_tokens:
            _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizer(tokenizer, sequence_length)


def _add_special_tokens(tokenizer, special_tokens):
    logger.info("Adding special tokens %s", special_tokens)
    special_tokens_num = len(special_tokens)
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code
    assert special_tokens_num < 50
    unused_ids = [
        tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)
    ]
    logger.info("Utilizing the following unused token ids %s", unused_ids)

    for idx, id in enumerate(unused_ids):
        del tokenizer.vocab["[unused{}]".format(idx)]
        tokenizer.vocab[special_tokens[idx]] = id
        tokenizer.ids_to_tokens[id] = special_tokens[idx]

    tokenizer._additional_special_tokens = list(special_tokens)
    logger.info(
        "Added special tokenizer.additional_special_tokens %s",
        tokenizer.additional_special_tokens,
    )
    logger.info("Tokenizer's all_special_tokens %s", tokenizer.all_special_tokens)


def get_roberta_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_roberta_tokenizer(
            args.pretrained_model_cfg
        )
    return RobertaTensorizer(tokenizer, args.sequence_length)


def get_optimizer(
        model: nn.Module,
        learning_rate: float = 1e-5,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    try:
        # return BertTokenizer.from_pretrained(
        return AutoTokenizer.from_pretrained(
            pretrained_cfg_name, local_files_only=True
        )
    except ValueError:
        # return BertTokenizer.from_pretrained(
        return AutoTokenizer.from_pretrained(
            "/mnt/netapp7/ohadr/prompt_sel/DPR/bert-base-uncased"
        )


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(
        pretrained_cfg_name
    )


# class HFBertEncoder(BertModel):
class HFBertEncoder(nn.Module):
    def __init__(self, config, project_dim: int = 0):
        # BertModel.__init__(self, config)
        # AutoModel.__init__(self, config)
        super(HFBertEncoder, self).__init__()
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        logger.info('in HFBertEncoder.__init__: project_dim={}'.format(project_dim))
        # self.init_weights()
        self.model = None
        self.config = None
        self.cfg_name = None

    def __call__(
            self,
            input_ids: T,
            token_type_ids: T,
            attention_mask: T,
            representation_token_pos=0,
    ):
        return self.forward(input_ids, token_type_ids, attention_mask, representation_token_pos)


    @classmethod
    def init_encoder(
            cls,
            cfg_name: str,
            projection_dim: int = 0,
            dropout: float = 0.1,
            pretrained: bool = True,
            gradient_checkpointing: bool = False,
            **kwargs
    ):
        # ) -> BertModel:
        cfg = AutoConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased", local_files_only=True)
        cfg.gradient_checkpointing = gradient_checkpointing
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            logger.info('in hf_models.py, from_pretrained {}, project_dim={}'.format(cfg_name, projection_dim))
            if gradient_checkpointing:
                logger.info("Using gradient checkpointing for encoder")
                # return AutoModel.from_pretrained(
                #     cfg_name, config=cfg, **kwargs
                # )
                encoder = cls.from_pretrained(
                    cfg_name, config=cfg, project_dim=projection_dim, **kwargs
                )
                # encoder.gradient_checkpointing_enable()
                return encoder
            else:
                # encoder = cls.from_pretrained(cfg_name, config=cfg, **kwargs)
                encoder = HFBertEncoder(cfg, project_dim=projection_dim)
                encoder.model = AutoModel.from_pretrained(cfg_name, config=cfg, local_files_only=True, **kwargs)
                encoder.model = encoder.model
                encoder.config = cfg
                encoder.cfg_name = cfg_name
                # encoder.model.forward = encoder.forward
                return encoder
                # if cfg_name == "bert-base-uncased":
                #     logger.info(cls.from_pretrained(
                #        cls, cfg_name, config=cfg, project_dim=projection_dim, **kwargs
                #     ))
                #     return cls.from_pretrained(
                #         cfg_name, config=cfg, project_dim=projection_dim, **kwargs
                #     )
                # else:
                #
                #     encoder = AutoModel.from_pretrained(
                #         cfg_name, config=cfg, **kwargs
                #     )
                #     logger.info(encoder)
                #     return encoder
        else:
            logger.info('in hf_models.py, not using pretrained, project_dim={}'.format(cfg_name, projection_dim))
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,

    ) -> Tuple[T, ...]:
        # print('before input_ids', torch.cuda.current_device(), input_ids.device)
        # print('before token_type_ids', torch.cuda.current_device(), token_type_ids.device)
        # print('before attention_mask', torch.cuda.current_device(), attention_mask.device)
        # print('before model', torch.cuda.current_device(), self.model.device)
        # # input_ids.to("cuda:"+str(torch.cuda.current_device()))
        # # token_type_ids.to("cuda:"+str(torch.cuda.current_device()))
        # # attention_mask.to("cuda:"+str(torch.cuda.current_device()))
        # # self.model = self.model.to("cuda:"+str(torch.cuda.current_device()))
        # print('after input_ids', torch.cuda.current_device(), input_ids.device)
        # print('after token_type_ids', torch.cuda.current_device(), token_type_ids.device)
        # print('after attention_mask', torch.cuda.current_device(), attention_mask.device)
        # print('after model', torch.cuda.current_device(), self.model.device)

        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = self.model.forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
        else:
            if self.cfg_name not in ["microsoft/deberta-v3-base"]:
                hidden_states = None
                sequence_output, pooled_output = self.model.forward(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
            else:
                hidden_states = None
                sequence_output = self.model.forward(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )[0]

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert (
                    representation_token_pos.size(0) == bsz
            ), "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack(
                [
                    sequence_output[i, representation_token_pos[i, 1], :]
                    for i in range(bsz)
                ]
            )

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BertTensorizer(Tensorizer):
    def __init__(
            # self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True
            self, tokenizer: AutoTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
            self,
            text: str,
            title: str = None,
            add_special_tokens: bool = True,
            apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                    seq_len - len(token_ids)
            )
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(
            tokenizer, max_length, pad_to_max=pad_to_max
        )
