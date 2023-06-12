import re
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import json, os
from src.utils.app import App
from nltk.tokenize import word_tokenize
from src.utils.dataset_utils import load_train_dataset

field_getter = App()


@field_getter.add("q")
def get_question(entry):
    # 与mtop等不同，kp20k的question是一个list，不需要norm
    return GoBM25Task.norm(entry['question'])


@field_getter.add("qa")
def get_qa(entry):
    return GoBM25Task.norm(f"{entry['question']} {entry['target']}")


@field_getter.add("a")
def get_decomp(entry):
    return GoBM25Task.norm(entry['target'])


class GoBM25Task:
    name = 'go'

    def __init__(self, dataset_split, setup_type, ds_size=None):
        self.setup_type = setup_type
        self.get_field = field_getter.functions[self.setup_type]
        self.dataset_split = dataset_split
        if os.path.exists("/remote-home/klv/exps/rtv_icl/data/go"):
            dataset = load_from_disk("/remote-home/klv/exps/rtv_icl/data/go")
        else:
            dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/go")
        self.train_dataset = load_train_dataset(dataset, size=ds_size)
        print(dataset)
        self.train_dataset = load_train_dataset(dataset, size=ds_size)
        if self.dataset_split == "train":
            self.dataset = self.train_dataset
        else:
            self.dataset = list(dataset[self.dataset_split])
        self.corpus = None
        self.instruction = "Represent the code example for retrieving duplicate examples; Input: "

    def get_corpus(self):
        if self.corpus is None:
            self.corpus = [self.get_field(entry) for entry in self.train_dataset]
        return self.corpus

    @classmethod
    def norm(cls, text):
        # 输出一个list
        return word_tokenize(text)
