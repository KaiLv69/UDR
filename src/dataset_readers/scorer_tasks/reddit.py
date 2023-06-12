from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import re
import json, os
from src.utils.dataset_utils import load_train_dataset


class RedditScorerTask:
    name = "reddit"
    question_field = "question"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
            dataset = load_from_disk("/remote-home/klv/exps/rtv_icl/data/reddit")
        else:
            dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/reddit")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = "TL;DR: "

    def get_fields(self, entry, index=-1):
        question_prefix = ""
        answer_prefix = "TL;DR: "
        test_question = question_prefix + entry['test_question']
        question = question_prefix + entry['question']
        decomposition = answer_prefix + entry['target']
        test_decomposition = entry['test_target']
        return question, decomposition, test_question, test_decomposition
