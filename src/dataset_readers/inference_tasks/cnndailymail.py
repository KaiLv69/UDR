from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from src.utils.dataset_utils import load_train_dataset

import json, os
from src.utils.tokenizer_utils import get_length


def set_length(example, idx,**kwargs):
    tokenizer = kwargs['tokenizer']
    question_prefix = ""
    answer_prefix = "TL;DR: "
    q_field = question_prefix + example['article']
    a_field = answer_prefix + example['highlights']
    prompt_qa = f"{q_field}\t{a_field}"
    example['prompt_qa'] = prompt_qa
    example['prompt_len'] = get_length(tokenizer,prompt_qa)
    return example


class CNNDailyMailInferenceTask:
    name = "cnndailymail"

    def __init__(self, prompt_file, tokenizer, ds_size=None):
        self.prompt_file = prompt_file
        with open(self.prompt_file) as f:
            self.prompts = json.load(f)
        dataset = load_dataset("KaiLv/UDR_CNNDailyMail")
        self.hf_dataset = load_train_dataset(dataset, size=ds_size, listify=False)
        self.hf_dataset = self.hf_dataset.map(set_length, with_indices=True, fn_kwargs={'tokenizer': tokenizer})
        self.training_dataset = list(self.hf_dataset)
        self.postfix = 'TL;DR: '

    @classmethod
    def postproccess(cls, string):
        return string

    def get_fields(self, entry):
        question = entry['article'] if "article" in entry else entry['question']
        answer = entry['highlights'] if "highlights" in entry else entry['answers'][0]
        idx_list = [p['id'] for p in entry['ctxs']]
        prompts = self.hf_dataset.select([p['id'] for p in entry['ctxs']])
        return "" + question, answer, prompts['prompt_qa'], prompts['prompt_len'], idx_list
