import os

from datasets import load_from_disk, Dataset, concatenate_datasets
import re
import json
from src.utils.dataset_utils import load_train_dataset


class WMTDeEnScorerTask:
    name = "wmt_de_en"
    question_field = "question"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        if os.path.exists("/remote-home/klv/exps/rtv_icl/data/wmt_de_en"):
            dataset = load_from_disk("/remote-home/klv/exps/rtv_icl/data/wmt_de_en")
        else:
            dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/wmt_de_en")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = "English:"

    def get_fields(self, entry, index=-1):
        question_prefix = "German: "
        answer_prefix = "English: "
        question = question_prefix + entry['question']
        test_question = question_prefix + entry['question']
        if 'target' in entry:
            decomposition = answer_prefix + entry['target']
            test_decomposition = entry['test_target']
        else:
            decomposition = answer_prefix + entry['answers'][0]
            test_decomposition = entry['answers'][0]
        return question, decomposition, test_question, test_decomposition
