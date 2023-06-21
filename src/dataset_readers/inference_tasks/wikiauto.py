from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from src.utils.dataset_utils import load_train_dataset

import json
from src.utils.tokenizer_utils import get_length


def set_length(example, idx,**kwargs):
    tokenizer = kwargs['tokenizer']
    question_prefix = "Simplify the text: "
    answer_prefix = "Simplified text: "
    q_field = question_prefix + example['source']
    a_field = answer_prefix + example['target']
    prompt_qa = f"{q_field}\t{a_field}"
    example['prompt_qa'] = prompt_qa
    example['prompt_len'] = get_length(tokenizer,prompt_qa)
    return example


class WikiautoInferenceTask:
    name = "wikiauto"

    def __init__(self, prompt_file, tokenizer, ds_size=None):
        self.prompt_file = prompt_file
        with open(self.prompt_file) as f:
            self.prompts = json.load(f)
        # dataset = load_dataset('GEM/wiki_auto_asset_turk', 'train')
        # dataset = dataset.filter(lambda x: len(x['target']) < 1000)
        # # add idx column
        # for split in ['train', 'validation', 'test_asset', 'test_turk', "test_wiki"]:
        #     ds_id = Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        #     dataset[split] = concatenate_datasets([dataset[split], ds_id], axis=1)
        dataset = load_dataset("KaiLv/UDR_WikiAuto")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size, listify=False)
        self.hf_dataset = self.hf_dataset.map(set_length, with_indices=True, fn_kwargs={'tokenizer': tokenizer})
        self.training_dataset = list(self.hf_dataset)
        self.postfix = 'Simplified text: '

    @classmethod
    def postproccess(cls, string):
        return string

    def get_fields(self, entry):
        question = entry['source'] if "source" in entry else entry['question']
        answer = entry['target'] if "target" in entry else entry['answers'][0]
        idx_list = [p['id'] for p in entry['ctxs']]
        prompts = self.hf_dataset.select([p['id'] for p in entry['ctxs']])
        return "Simplify the text: " + question, answer, prompts['prompt_qa'], prompts['prompt_len'], idx_list
