from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import re
import json, os
from src.utils.dataset_utils import load_train_dataset
from Channel_LM_Prompting.util import get_one_prompt


class SNLIScorerTask:
    name = "snli"
    question_field = "sentence"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        dataset = load_dataset("KaiLv/UDR_SNLI")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = ""

    def get_fields(self, entry, index=-1):
        question_prefix = ""
        answer_prefix = ""
        test_question = question_prefix + entry['test_sentence']
        question = question_prefix + entry['sentence']
        decomposition = get_one_prompt('snli', 0, entry['label'])
        test_decomposition = get_one_prompt('snli', 0, entry['test_label'])
        return question, decomposition, test_question, test_decomposition
