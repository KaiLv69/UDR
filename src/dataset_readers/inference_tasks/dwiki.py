from datasets import load_dataset
from src.utils.dataset_utils import load_train_dataset

import json
from src.utils.tokenizer_utils import get_length


def set_length(example, idx,**kwargs):
    question_prefix = "Simplify the text: "
    answer_prefix = "Simplified text: "
    tokenizer = kwargs['tokenizer']
    q_field = example['src']
    q_field = question_prefix + q_field
    a_field = example['tgt']
    a_field = answer_prefix + a_field
    prompt_qa = f"{q_field}\t{a_field}"
    example['prompt_qa'] = prompt_qa
    example['prompt_len'] = get_length(tokenizer,prompt_qa)
    return example


class DwikiInferenceTask:
    name = "dwiki"

    def __init__(self, prompt_file, tokenizer, ds_size=None):
        self.prompt_file = prompt_file
        with open(self.prompt_file) as f:
            self.prompts = json.load(f)
        base_dir = "/remote-home/klv/exps/rtv_icl/Document-level-text-simplification/Dataset/"
        dataset = load_dataset("json", data_files={
            'train': base_dir + "train.json",
            'valid': base_dir + "valid.json",
            "test": base_dir + "test.json"
        })

        def process_dwiki(example):
            example['src'] = example['src'].strip()
            example['tgt'] = example['tgt'].strip()
            example['src_len'] = len(example['src'].split(" "))
            example['tgt_len'] = len(example['tgt'].split(" "))
            return example

        dataset = dataset.map(process_dwiki)
        dataset = dataset.filter(lambda x: x['src_len'] + x['tgt_len'] < 1000)
        self.hf_dataset = load_train_dataset(dataset, size=ds_size, listify=False)
        self.hf_dataset = self.hf_dataset.map(set_length, with_indices=True, fn_kwargs={'tokenizer': tokenizer})
        self.training_dataset = list(self.hf_dataset)
        self.prefix = ""
        self.postfix = 'Simplified text: '

    @classmethod
    def postproccess(cls, string):
        return string

    def get_fields(self, entry):
        question_prefix = "Simplify the text: "
        question = entry['src'] if "src" in entry else entry['question']
        question = question_prefix + question
        answer = entry['tgt'] if "tgt" in entry else entry['answers'][0]
        idx_list = [p['id'] for p in entry['ctxs']]
        prompts = self.hf_dataset.select([p['id'] for p in entry['ctxs']])
        return question, answer, prompts['prompt_qa'], prompts['prompt_len'], idx_list
