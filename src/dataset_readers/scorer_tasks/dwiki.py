from datasets import load_dataset
import re
import json
from src.utils.dataset_utils import load_train_dataset


class DwikiScorerTask:
    name = "dwiki"
    question_field = "src"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        base_dir = "/remote-home/klv/exps/rtv_icl/Document-level-text-simplification/Dataset/"
        dataset = load_dataset("json", data_files={
            'train': base_dir + "train.json",
            'valid': base_dir + "valid.json",
            "test": base_dir + "test.json"
        })

        def process_dwiki(example):
            example['src'] = example['src'].strip()
            example['tgt'] = example['tgt'].strip()
            example['src_len'] = len(example['src'].split(" "))
            example['tgt_len'] = len(example['tgt'].split(" "))
            return example

        dataset = dataset.map(process_dwiki)
        dataset = dataset.filter(lambda x: x['src_len'] + x['tgt_len'] < 1000)
        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)[:10]
        self.postfix = "Simplified text: "

    def get_fields(self, entry, index=-1):
        question_prefix = "Simplify the text: "
        answer_prefix = "Simplified text: "
        answer_suffix = ""
        test_question = question_prefix + entry['test_src']
        question = question_prefix + entry['src']
        decomposition = answer_prefix + entry['tgt'] + answer_suffix
        test_decomposition = entry['test_tgt'] + answer_suffix
        return question, decomposition, test_question, test_decomposition
