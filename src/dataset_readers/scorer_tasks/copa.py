from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import re
import json, os
from src.utils.dataset_utils import load_train_dataset
from Channel_LM_Prompting.util import get_one_prompt


class CopaScorerTask:
    name = "copa"
    question_field = "question"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
            dataset = load_from_disk("/remote-home/klv/exps/rtv_icl/data/copa")
        else:
            dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/copa")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = "Answer: "

    def get_fields(self, entry, index=-1):
        question_prefix = ""
        answer_prefix = "Answer: "
        test_question = question_prefix + entry['test_question']
        question = question_prefix + entry['question']
        decomposition = answer_prefix + entry['label']
        test_decomposition = entry['test_label']
        return question, decomposition, test_question, test_decomposition
