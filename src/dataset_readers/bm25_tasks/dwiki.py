import re
from datasets import load_dataset
import json
from src.utils.app import App
from nltk.tokenize import word_tokenize
from src.utils.dataset_utils import load_train_dataset

field_getter = App()


@field_getter.add("q")
def get_question(entry):
    # 与mtop等不同，kp20k的question是一个list，不需要norm
    return DwikiBM25Task.norm(entry['src'])


@field_getter.add("qa")
def get_qa(entry):
    return DwikiBM25Task.norm(f"{entry['src']} {entry['tgt']}")


@field_getter.add("a")
def get_decomp(entry):
    return DwikiBM25Task.norm(entry['tgt'])


class DwikiBM25Task:
    name = 'dwiki'

    def __init__(self, dataset_split, setup_type, ds_size=None):
        self.setup_type = setup_type
        self.get_field = field_getter.functions[self.setup_type]
        self.dataset_split = dataset_split
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
        # overfit for debug
        dataset['debug'] = dataset['train'].select(range(5000))
        self.train_dataset = load_train_dataset(dataset, size=ds_size)
        if self.dataset_split == "train":
            self.dataset = self.train_dataset
        else:
            self.dataset = list(dataset[self.dataset_split])
        self.corpus = None

    def get_corpus(self):
        if self.corpus is None:
            self.corpus = [self.get_field(entry) for entry in self.train_dataset]
        return self.corpus

    @classmethod
    def norm(cls, text):
        # 输出一个list
        return word_tokenize(text)
