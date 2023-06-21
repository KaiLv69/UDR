from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import re, os
import json
from src.utils.dataset_utils import load_train_dataset


class DartScorerTask:
    name = "dart"
    question_field = "question"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        dataset = load_dataset("KaiLv/UDR_DART")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = "Sentence: "

    def get_fields(self, entry, index=-1):
        question_prefix = "Table: "
        answer_prefix = "Sentence: "
        test_question = question_prefix + entry['test_question']
        question = question_prefix + entry['question']
        decomposition = answer_prefix + entry['target']
        test_decomposition = entry['test_target']
        return question, decomposition, test_question, test_decomposition
