import re
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import json
from src.utils.app import App
from nltk.tokenize import word_tokenize
from src.utils.dataset_utils import load_train_dataset

field_getter = App()


@field_getter.add("q")
def get_question(entry):
    # 与mtop等不同，kp20k的question是一个list，不需要norm
    return KP20kBM25Task.norm(entry['document'])


@field_getter.add("qa")
def get_qa(entry):
    # return KP20kBM25Task.norm(f"{entry['document']} {entry['abstractive_keyphrases']}")
    return KP20kBM25Task.norm(f"{entry['document']} {entry['extractive_keyphrases']}")


@field_getter.add("a")
def get_decomp(entry):
    # return KP20kBM25Task.norm(entry['abstractive_keyphrases'])
    return KP20kBM25Task.norm(entry['extractive_keyphrases'])


class KP20kBM25Task:
    name = 'kp20k'

    def __init__(self, dataset_split, setup_type, ds_size=None):
        self.setup_type = setup_type
        self.get_field = field_getter.functions[self.setup_type]
        self.dataset_split = dataset_split
        # dataset = load_dataset("midas/kp20k", "generation")
        # # 去掉extractive_keyphrases为空的数据 去掉包含<eos>的数据
        # dataset = dataset.filter(lambda x: len(x['extractive_keyphrases']) > 0 and len(x['document']) + 4 * len(x['extractive_keyphrases']) < 1000)
        #
        # def process_kp20k(example):
        #     example['extractive_keyphrases'] = str(example['extractive_keyphrases']).replace("'", '"')
        #     example['abstractive_keyphrases'] = str(example['abstractive_keyphrases']).replace("'", '"')
        #     example['document'] = " ".join(example['document'])
        #     return example
        #
        # dataset = dataset.map(process_kp20k, num_proc=8)
        # # add idx column
        # dataset = dataset.remove_columns("id")
        # for split in ['train', 'validation', 'test']:
        #     ds_id = Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        #     dataset[split] = concatenate_datasets([dataset[split], ds_id], axis=1)
        # # overfit for debug
        # dataset['debug'] = dataset['train'].select(range(5000))
        dataset = load_from_disk("/remote-home/klv/exps/rtv_icl/data/kp20k")

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
