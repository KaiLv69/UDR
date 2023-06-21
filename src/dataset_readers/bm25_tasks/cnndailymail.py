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
    return CNNDailyMailBM25Task.norm(entry['article'])


@field_getter.add("qa")
def get_qa(entry):
    return CNNDailyMailBM25Task.norm(f"{entry['article']} {entry['highlights']}")


@field_getter.add("a")
def get_decomp(entry):
    return CNNDailyMailBM25Task.norm(entry['highlights'])


class CNNDailyMailBM25Task:
    name = 'cnndailymail'

    def __init__(self, dataset_split, setup_type, ds_size=None):
        self.setup_type = setup_type
        self.get_field = field_getter.functions[self.setup_type]
        self.dataset_split = dataset_split
        dataset = load_dataset("KaiLv/UDR_CNNDailyMail")
        self.train_dataset = load_train_dataset(dataset, size=ds_size)
        if self.dataset_split == "train":
            self.dataset = self.train_dataset
        else:
            self.dataset = list(dataset[self.dataset_split])
        self.corpus = None
        self.instruction = "Represent the news example for retrieving duplicate examples; Input: "

    def get_corpus(self):
        if self.corpus is None:
            self.corpus = [self.get_field(entry) for entry in self.train_dataset]
        return self.corpus

    @classmethod
    def norm(cls, text):
        # 输出一个list
        return word_tokenize(text)
