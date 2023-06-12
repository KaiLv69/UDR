from datasets import load_dataset, Dataset, concatenate_datasets
from src.utils.dataset_utils import load_train_dataset

import json
from src.utils.tokenizer_utils import get_length


def set_length(example, idx,**kwargs):
    tokenizer = kwargs['tokenizer']
    q_field = example['document']
    # a_field = example['extractive_keyphrases']
    a_field = example['abstractive_keyphrases']
    prompt_qa = f"{q_field}\t{a_field}"
    example['prompt_qa'] = prompt_qa
    example['prompt_len'] = get_length(tokenizer,prompt_qa)
    return example


class KP20kInferenceTask:
    name = "kp20k"

    def __init__(self, prompt_file, tokenizer, ds_size=None):
        self.prompt_file = prompt_file
        with open(self.prompt_file) as f:
            self.prompts = json.load(f)
        dataset = load_dataset("midas/kp20k", "generation")
        # 去掉extractive_keyphrases为空的数据 去掉包含<eos>的数据
        dataset = dataset.filter(lambda x: len(x['extractive_keyphrases']) > 0 and len(x['document']) + 4 * len(x['extractive_keyphrases']) < 1000)

        def process_kp20k(example):
            example['extractive_keyphrases'] = str(example['extractive_keyphrases']).replace("'", '"')
            example['abstractive_keyphrases'] = str(example['abstractive_keyphrases']).replace("'", '"')
            example['document'] = " ".join(example['document'])
            return example

        dataset = dataset.map(process_kp20k, num_proc=8)
        # add idx column
        dataset = dataset.remove_columns("id")
        for split in ['train', 'validation', 'test']:
            ds_id = Dataset.from_dict({"idx": list(range(len(dataset[split])))})
            dataset[split] = concatenate_datasets([dataset[split], ds_id], axis=1)

        self.hf_dataset = load_train_dataset(dataset, size=ds_size, listify=False)
        self.hf_dataset = self.hf_dataset.map(set_length, with_indices=True, fn_kwargs={'tokenizer': tokenizer})
        self.training_dataset = list(self.hf_dataset)
        self.prefix = ""
        self.postfix = '["'

    @classmethod
    def postproccess(cls, string):
        return string

    def get_fields(self, entry):
        question = entry['document'] if "document" in entry else entry['question']
        # answer = entry['abstractive_keyphrases'] if "abstractive_keyphrases" in entry else entry['answers'][0]
        answer = entry['extractive_keyphrases'] if "extractive_keyphrases" in entry else entry['answers'][0]
        idx_list = [p['id'] for p in entry['ctxs']]
        prompts = self.hf_dataset.select([p['id'] for p in entry['ctxs']])
        return question, answer, prompts['prompt_qa'], prompts['prompt_len'], idx_list
