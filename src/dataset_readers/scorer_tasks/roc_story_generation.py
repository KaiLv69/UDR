from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import re
import json, os
from src.utils.dataset_utils import load_train_dataset


class RocStoryGenerationScorerTask:
    name = "roc_story_generation"
    question_field = "question"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        current_path = os.getcwd()
        base_path = current_path.split("UDR")[0] + "UDR"
        dataset = load_from_disk(os.path.join(base_path, "data/roc_story_generation"))

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = "Rest of the story: "

    def get_fields(self, entry, index=-1):
        question_prefix = "Beginning of the story: "
        answer_prefix = "Rest of the story: "
        test_question = question_prefix + entry['test_question']
        question = question_prefix + entry['question']
        decomposition = answer_prefix + entry['target']
        test_decomposition = entry['test_target']
        return question, decomposition, test_question, test_decomposition
