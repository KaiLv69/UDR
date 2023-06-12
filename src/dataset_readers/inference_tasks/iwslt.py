import os

from datasets import load_from_disk, Dataset, concatenate_datasets
from src.utils.dataset_utils import load_train_dataset

import json
from src.utils.tokenizer_utils import get_length


def set_length(example, idx,**kwargs):
    tokenizer = kwargs['tokenizer']
    question_prefix = "German: "
    answer_prefix = "English: "
    q_field = question_prefix + example['translation.de']
    a_field = answer_prefix + example['translation.en']
    prompt_qa = f"{q_field}\t{a_field}"
    example['prompt_qa'] = prompt_qa
    example['prompt_len'] = get_length(tokenizer,prompt_qa)
    return example


class IWSLTInferenceTask:
    name = "iwslt"

    def __init__(self, prompt_file, tokenizer, ds_size=None):
        self.prompt_file = prompt_file
        with open(self.prompt_file) as f:
            self.prompts = json.load(f)
        if os.path.exists("/remote-home/klv/exps/rtv_icl/data/iwslt"):
            dataset = load_from_disk("/remote-home/klv/exps/rtv_icl/data/iwslt")
        else:
            dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/iwslt")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size, listify=False)
        self.hf_dataset = self.hf_dataset.map(set_length, with_indices=True, fn_kwargs={'tokenizer': tokenizer})
        self.training_dataset = list(self.hf_dataset)
        self.postfix = 'English: '

    @classmethod
    def postproccess(cls, string):
        return string

    def get_fields(self, entry):
        question = entry['translation.de'] if "translation.de" in entry else entry['question']
        answer = entry['translation.en'] if "translation.en" in entry else entry['answers'][0]
        idx_list = [p['id'] for p in entry['ctxs']]
        prompts = self.hf_dataset.select([p['id'] for p in entry['ctxs']])
        return "German: " + question, answer, prompts['prompt_qa'], prompts['prompt_len'], idx_list
