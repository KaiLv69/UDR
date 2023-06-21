from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import re
import json
from src.utils.dataset_utils import load_train_dataset


class CommonGenScorerTask:
    name = "common_gen"
    question_field = "joined_concepts"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        dataset = load_dataset("KaiLv/UDR_CommonGen")

        # if 'q' in example_file.split('/')[-1]:
        #     self.hf_dataset = dataset['train_dedup']
        # elif 'a' in example_file.split('/')[-1]:
        self.hf_dataset = load_train_dataset(dataset, size=ds_size)

        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = "Generated sentence: "

    def get_fields(self, entry, index=-1):
        question_prefix = "Generate a sentence using these concepts: "
        answer_prefix = "Generated sentence: "
        test_question = question_prefix + entry['test_joined_concepts']
        question = question_prefix + entry['joined_concepts']
        decomposition = answer_prefix + entry['target']
        test_decomposition = entry['test_target']
        return question, decomposition, test_question, test_decomposition
