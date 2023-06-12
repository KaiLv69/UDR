from datasets import load_dataset
from typing import Any, Dict, Iterable
from transformers import AutoTokenizer
import torch
import pandas as pd
import numpy as np
import json
import random
import os
import pandas as pd
from datasets import Dataset
import json
from collections import defaultdict
import re
from src.dataset_readers.scorer_tasks import ScorerTask
import logging

logger = logging.getLogger(__name__)


class ScorerDatasetReader(torch.utils.data.Dataset):

    def __init__(self, example_file, model_name, task_name, setup_type, ds_size=None, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.tokenizer.pad_token = "<|endoftext|>"
        # self.tokenizer.padding_side = "left"
        self.tokenizer.padding_side = "right"
        self.task = ScorerTask.from_name(task_name)(example_file, ds_size)
        self.kwargs = kwargs
        self.setup_type = setup_type
        assert self.setup_type in ["qa", "q"]

        def get_instance(entry):

            # examples = entry.pop("examples") if "examples" in entry else entry.pop('near_examples')
            examples = entry.pop("ctxs")
            for exp in examples:
                # print(exp)  # {'id': 807}
                # print(self.task.training_dataset[exp['id']])  # (807, {'id': None, "document", "e_k", "a_k"})
                exp.update(self.task.training_dataset[exp['id']][1])
                for key, val in entry.items():
                    exp[f"test_{key}"] = val
            yield from examples

        def get_dataset(data):
            for entry in data:
                yield from get_instance(entry)

        # overfit for debug
        if "debug" in example_file.split("/")[-1]:
            df = pd.DataFrame(list(get_dataset(self.task.data)))
        else:
            # if task_name in ['kp20k', 'dwiki']:
            #     data_num = 50000
            #     df_path = f"/remote-home/klv/exps/rtv_icl/data/score_{task_name}_{setup_type}_{data_num}.csv"
            #     if os.path.exists(df_path):
            #         df = pd.read_csv(df_path)
            #     else:
            #         random.seed(42)
            #         data_sample = random.sample(self.task.data, data_num)
            #         df = pd.DataFrame(list(get_dataset(data_sample)))
            #         df.to_csv(df_path)
            # else:
            #     df = pd.DataFrame(list(get_dataset(self.task.data)))
            df = pd.DataFrame(list(get_dataset(self.task.data)))
        self.dataset = Dataset.from_pandas(df)
        logger.info('dataset size:{}'.format(len(self.dataset)))

    def shard(self, accelerator):
        self.dataset = self.dataset.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)

    def __getitem__(self, index):
        return self.text_to_instance(self.dataset[index], index=index)

    def __len__(self):
        return len(self.dataset)

    def text_to_instance(self, entry: Dict[str, Any], index=-1):
        question, answer, test_question, test_answer = self.task.get_fields(entry)
        if self.setup_type == "qa":
            if hasattr(self.task, 'postfix'):
                enc_text = f"{question}\t{answer}\n{test_question}\t{self.task.postfix}{test_answer}"
            else:
                enc_text = f"{question}\t{answer}\n{test_question}\t{test_answer}"
            tokenized_example = self.tokenizer.encode_plus(enc_text, truncation=True, add_special_tokens=False,
                                                           return_tensors='pt')
            tokenized_labels = self.tokenizer.encode_plus(test_answer, truncation=True, add_special_tokens=False,
                                                          return_tensors='pt')
            # tokenized_example = self.tokenizer.encode_plus(enc_text, truncation=True, add_special_tokens=False,
            #                                                return_tensors='pt')
            # tokenized_labels = self.tokenizer.encode_plus(test_answer, truncation=True, add_s`pecial_tokens=False,
            #                                               return_tensors='pt')
        elif self.setup_type == "q":
            enc_text = f"{question}\t{test_question}"
            tokenized_example = self.tokenizer.encode_plus(enc_text, truncation=False, add_special_tokens=False,
                                                           return_tensors='pt')
            tokenized_labels = self.tokenizer.encode_plus(test_question, truncation=False, add_special_tokens=False,
                                                          return_tensors='pt')
            # tokenized_example = self.tokenizer.encode_plus(enc_text, truncation=True, add_special_tokens=False,
            #                                                return_tensors='pt')
            # tokenized_labels = self.tokenizer.encode_plus(test_question, truncation=True, add_special_tokens=False,
            #                                               return_tensors='pt')
        else:
            raise NotImplementedError
        # 这里只考虑mask前面q的部分的loss，不考虑batch里的pad，batch里的pad应该会自动补0，待确认
        input_ids = tokenized_example.input_ids.squeeze(0)
        labels = tokenized_labels.attention_mask.squeeze(0)

        try:
            pad_mask = torch.nn.functional.pad(labels, (input_ids.shape[-1] - labels.shape[-1] - 1, 0), value=0)
        except Exception as e:
            print(e)
            print(labels)
            print(input_ids)
            print(input_ids.shape[-1])
            print(labels.shape[-1])
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pad_mask': pad_mask,
            "metadata": entry
        }
