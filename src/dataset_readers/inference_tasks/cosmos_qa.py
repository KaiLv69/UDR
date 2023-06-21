import os

from datasets import load_from_disk, Dataset, concatenate_datasets
from src.utils.dataset_utils import load_train_dataset

import json
from src.utils.tokenizer_utils import get_length
from src.utils.cls_utils import get_multi_choice_labels


def set_length(example, idx,**kwargs):
    tokenizer = kwargs['tokenizer']
    q_field = example['question']
    a_field = example['label']
    prompt_qa = f"{q_field}\tAnswer: {a_field}"
    example['prompt_qa'] = prompt_qa
    example['prompt_len'] = get_length(tokenizer,prompt_qa)
    return example


class CosmosQaInferenceTask:
    name = "cosmos_qa"

    def __init__(self, prompt_file, tokenizer, ds_size=None):
        self.prompt_file = prompt_file
        with open(self.prompt_file) as f:
            self.prompts = json.load(f)
        self.prompts = get_multi_choice_labels(self.prompts, 'cosmos_qa', 'validation')
        dataset = load_dataset("KaiLv/UDR_CosmosQA")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size, listify=False)
        self.hf_dataset = self.hf_dataset.map(set_length, with_indices=True, fn_kwargs={'tokenizer': tokenizer})
        self.training_dataset = list(self.hf_dataset)
        self.postfix = 'Answer: '

    @classmethod
    def postproccess(cls, string):
        return string

    def get_fields(self, entry):
        question = entry['question']
        answer = entry['label'] if "label" in entry else entry['answers'][0]
        idx_list = [p['id'] for p in entry['ctxs']]
        prompts = self.hf_dataset.select([p['id'] for p in entry['ctxs']])
        return question, answer, prompts['prompt_qa'], prompts['prompt_len'], entry['test_label']
