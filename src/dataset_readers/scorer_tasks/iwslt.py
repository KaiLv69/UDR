import os

from datasets import load_from_disk, Dataset, concatenate_datasets
import re
import json
from src.utils.dataset_utils import load_train_dataset


class IWSLTScorerTask:
    name = "iwslt"
    question_field = "translation.de"
    # question_field="question"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        if os.path.exists("/remote-home/klv/exps/rtv_icl/data/iwslt"):
            dataset = load_from_disk("/remote-home/klv/exps/rtv_icl/data/iwslt")
        else:
            dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/iwslt")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = "English :"

    def get_fields(self, entry, index=-1):
        question_prefix = "German: "
        answer_prefix = "English: "
        if 'translation.de' in entry:
            question = question_prefix + entry['translation.de']
            test_question = question_prefix + entry['test_translation.de']
        else:
            question = question_prefix + entry['question']
            test_question = question_prefix + entry['question']
        if 'translation.en' in entry:
            decomposition = answer_prefix + entry['translation.en']
            test_decomposition = entry['test_translation.en']
        else:
            decomposition = answer_prefix + entry['answers'][0]
            test_decomposition = entry['answers'][0]
        return question, decomposition, test_question, test_decomposition
