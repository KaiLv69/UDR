import collections
import csv
import json
import logging
import os.path
import pickle
from typing import Dict


import hydra
import jsonlines
import torch
from omegaconf import DictConfig
from dpr.utils.data_utils import App
from datasets import load_dataset, load_from_disk
import datasets

from dpr.utils.data_utils import load_train_dataset, get_one_prompt

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
    normalize_question,
    get_dpr_files,
    read_nq_tables_jsonl,
    split_tables_to_chunks,
)

logger = logging.getLogger(__name__)
QASample = collections.namedtuple("QuerySample", ["query", "id", "answers"])
TableChunk = collections.namedtuple("TableChunk", ["text", "title", "table_id"])



class RetrieverData(torch.utils.data.Dataset):
    def __init__(self, file: str):
        """
        :param file: - real file name or the resource name as they are defined in download_data.py
        """
        self.file = file
        self.data_files = []

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        assert (
            len(self.data_files) == 1
        ), "RetrieverData source currently works with single files only. Files specified: {}".format(
            self.data_files
        )
        self.file = self.data_files[0]


class QASrc(RetrieverData):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file)
        self.data = None
        self.selector = hydra.utils.instantiate(selector) if selector else None
        self.special_query_token = special_query_token
        self.query_special_suffix = query_special_suffix

    def __getitem__(self, index) -> QASample:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _process_question(self, question: str):
        # as of now, always normalize query
        question = normalize_question(question)
        if self.query_special_suffix and not question.endswith(
            self.query_special_suffix
        ):
            question += self.query_special_suffix
        return question


class CsvQASrc(QASrc):
    def __init__(
        self,
        file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_col = question_col
        self.answers_col = answers_col
        self.id_col = id_col

    def load_data(self):
        super().load_data()
        data = []
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[self.question_col]
                answers = eval(row[self.answers_col])
                id = None
                if self.id_col >= 0:
                    id = row[self.id_col]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data




class JsonlQASrc(QASrc):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        question_attr: str = "question",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_attr = question_attr
        self.answers_attr = answers_attr
        self.id_attr = id_attr

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                answers = jline[self.answers_attr] if self.answers_attr in jline else []
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class KiltCsvQASrc(CsvQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            question_col,
            answers_col,
            id_col,
            selector,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file


class KiltJsonlQASrc(JsonlQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_attr: str = "input",
        answers_attr: str = "answer",
        id_attr: str = "id",
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            selector,
            question_attr,
            answers_attr,
            id_attr,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                out = jline["output"]
                answers = [o["answer"] for o in out if "answer" in o]
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class TTS_ASR_QASrc(QASrc):
    def __init__(self, file: str, trans_file: str):
        super().__init__(file)
        self.trans_file = trans_file

    def load_data(self):
        super().load_data()
        orig_data_dict = {}
        with open(self.file, "r") as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            id = 0
            for row in reader:
                question = row[0]
                answers = eval(row[1])
                orig_data_dict[id] = (question, answers)
                id += 1
        data = []
        with open(self.trans_file, "r") as tfile:
            reader = csv.reader(tfile, delimiter="\t")
            for r in reader:
                row_str = r[0]
                idx = row_str.index("(None-")
                q_id = int(row_str[idx + len("(None-") : -1])
                orig_data = orig_data_dict[q_id]
                answers = orig_data[1]
                q = row_str[:idx].strip().lower()
                data.append(QASample(q, idx, answers))
        self.data = data


class CsvCtxSrc(RetrieverData):
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                if row[self.id_col] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col]
                passage = row[self.text_col]
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row[self.title_col])

dataset_dict = App()
@dataset_dict.add("break")
def get_break():
    return load_dataset("break_data","QDMR")

@dataset_dict.add("mtop")
def get_mtop():
    return load_dataset("iohadrubin/mtop",name="mtop")


@dataset_dict.add("smcalflow")
def get_smcalflow():
    dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/smcalflow")
    # dataset = load_dataset("iohadrubin/smcalflow")
    return dataset
@dataset_dict.add("kp20k")
def get_kp20k():
    dataset = load_dataset("midas/kp20k", "generation")
    dataset = dataset.filter(lambda x: len(x['extractive_keyphrases']) > 0 and len(x['document']) + 4 * len(
        x['extractive_keyphrases']) < 1000)

    def process_kp20k(example):
        example['extractive_keyphrases'] = str(example['extractive_keyphrases']).replace("'", '"')
        example['abstractive_keyphrases'] = str(example['abstractive_keyphrases']).replace("'", '"')
        example['document'] = " ".join(example['document'])
        return example
    dataset = dataset.map(process_kp20k, num_proc=8)
    # add idx column
    dataset = dataset.remove_columns("id")
    for split in ['train', 'validation', 'test']:
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([dataset[split], ds_id], axis=1)
    # overfit for debug
    dataset['debug'] = dataset['train'].select(range(5000))
    return dataset


@dataset_dict.add("dwiki")
def get_dwiki():
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
    return dataset


@dataset_dict.add("wikiauto")
def get_wikiauto():
    # dataset = load_dataset('GEM/wiki_auto_asset_turk', 'train')
    # dataset = dataset.filter(lambda x: len(x['target']) < 1000)
    # # add idx column
    # for split in ['train', 'validation', 'test_asset', 'test_turk', 'test_wiki']:
    #     ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
    #     dataset[split] = datasets.concatenate_datasets([dataset[split], ds_id], axis=1)
    # dataset['debug'] = dataset['train'].select(range(5000))
    dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/wikiauto")
    return dataset

@dataset_dict.add("iwslt")
def get_iwslt():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/iwslt"):
        dataset = load_from_disk("/remote-home/klv/exps/rtv_icl/data/iwslt")
    else:
        dataset = datasets.load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/iwslt")
    return dataset

@dataset_dict.add("squadv2")
def get_squadv2():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/squadv2"):
        dataset = load_from_disk("/remote-home/klv/exps/rtv_icl/data/squadv2")
    else:
        dataset = datasets.load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/squadv2")
    return dataset

@dataset_dict.add("opusparcus")
def get_opusparcus():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/opusparcus"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/opusparcus")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/opusparcus")
    return dataset

@dataset_dict.add("common_gen")
def get_common_gen():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/common_gen"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/common_gen")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/common_gen")
    return dataset


@dataset_dict.add("xsum")
def get_xsum():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/xsum"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/xsum")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/xsum")
    return dataset

@dataset_dict.add("spider")
def get_spider():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/spider"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/spider")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/spider")
    return dataset

@dataset_dict.add("iwslt_en_fr")
def get_iwslt_en_fr():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/iwslt_en_fr"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/iwslt_en_fr")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/iwslt_en_fr")
    return dataset

@dataset_dict.add("iwslt_en_de")
def get_iwslt_en_de():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/iwslt_en_de"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/iwslt_en_de")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/iwslt_en_de")
    return dataset

@dataset_dict.add("wmt_en_de")
def get_wmt_en_de():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/wmt_en_de"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/wmt_en_de")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/wmt_en_de")
    return dataset

@dataset_dict.add("roc_ending_generation")
def get_roc_ending_generation():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/roc_ending_generation"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/roc_ending_generation")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/roc_ending_generation")
    return dataset


@dataset_dict.add("roc_story_generation")
def get_roc_story_generation():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/roc_story_generation"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/roc_story_generation")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/roc_story_generation")
    return dataset


@dataset_dict.add("wmt_de_en")
def get_wmt_de_en():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/wmt_de_en"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/wmt_de_en")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/wmt_de_en")
    return dataset

@dataset_dict.add("e2e")
def get_e2e():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/e2e"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/e2e")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/e2e")
    return dataset


@dataset_dict.add("dart")
def get_dart():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/dart"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/dart")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/dart")
    return dataset

@dataset_dict.add("totto")
def get_totto():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/totto"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/totto")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/totto")
    return dataset

@dataset_dict.add("cnndailymail")
def get_cnndailymail():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/cnndailymail"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/cnndailymail")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/cnndailymail")
    return dataset

@dataset_dict.add("python")
def get_python():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/python"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/python")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/python")
    return dataset

@dataset_dict.add("go")
def get_go():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/go"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/go")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/go")
    return dataset

@dataset_dict.add("php")
def get_php():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/php"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/php")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/php")
    return dataset


@dataset_dict.add("trec")
def get_trec():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/trec"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/trec")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/trec")
    return dataset

@dataset_dict.add("sst2")
def get_sst2():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/sst2"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/sst2")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/sst2")
    return dataset

@dataset_dict.add("imdb")
def get_imdb():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/imdb"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/imdb")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/imdb")
    return dataset

@dataset_dict.add("tweet_sentiment_extraction")
def get_tweet_sentiment_extraction():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/tweet_sentiment_extraction"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/tweet_sentiment_extraction")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/tweet_sentiment_extraction")
    return dataset

@dataset_dict.add("financial_phrasebank")
def get_financial_phrasebank():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/financial_phrasebank"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/financial_phrasebank")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/financial_phrasebank")
    return dataset

@dataset_dict.add("emotion")
def get_emotion():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/emotion"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/emotion")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/emotion")
    return dataset

@dataset_dict.add("mnli")
def get_mnli():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/mnli"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/mnli")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/mnli")
    return dataset

@dataset_dict.add("cola")
def get_cola():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/cola"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/cola")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/cola")
    return dataset

@dataset_dict.add("qnli")
def get_qnli():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/qnli"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/qnli")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/qnli")
    return dataset

@dataset_dict.add("mrpc")
def get_mrpc():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/mrpc"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/mrpc")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/mrpc")
    return dataset

@dataset_dict.add("boolq")
def get_boolq():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/boolq"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/boolq")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/boolq")
    return dataset


@dataset_dict.add("qqp")
def get_qqp():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/qqp"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/qqp")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/qqp")
    return dataset

@dataset_dict.add("wnli")
def get_wnli():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/wnli"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/wnli")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/wnli")
    return dataset

@dataset_dict.add("snli")
def get_snli():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/snli"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/snli")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/snli")
    return dataset

@dataset_dict.add("rte")
def get_rte():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/rte"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/rte")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/rte")
    return dataset


@dataset_dict.add("cr")
def get_cr():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/cr"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/cr")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/cr")
    return dataset

@dataset_dict.add("subj")
def get_subj():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/subj"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/subj")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/subj")
    return dataset

@dataset_dict.add("mr")
def get_mr():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/mr"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/mr")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/mr")
    return dataset

@dataset_dict.add("yelp_full")
def get_yelp_full():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/yelp_full"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/yelp_full")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/yelp_full")
    return dataset

@dataset_dict.add("amazon")
def get_amazon():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/amazon"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/amazon")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/amazon")
    return dataset

@dataset_dict.add("agnews")
def get_agnews():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/agnews"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/agnews")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/agnews")
    return dataset

@dataset_dict.add("amazon_scenario")
def get_amazon_scenario():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/amazon_scenario"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/amazon_scenario")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/amazon_scenario")
    return dataset

@dataset_dict.add("mtop_domain")
def get_mtop_domain():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/mtop_domain"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/mtop_domain")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/mtop_domain")
    return dataset

@dataset_dict.add("dbpedia")
def get_dbpedia():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/dbpedia"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/dbpedia")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/dbpedia")
    return dataset

@dataset_dict.add("bank77")
def get_bank77():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/bank77"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/bank77")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/bank77")
    return dataset

@dataset_dict.add("yahoo")
def get_yahoo():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/yahoo"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/yahoo")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/yahoo")
    return dataset

@dataset_dict.add("sst5")
def get_sst5():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/sst5"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/sst5")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/sst5")
    return dataset

@dataset_dict.add("commonsense_qa")
def get_commonsense_qa():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/commonsense_qa"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/commonsense_qa")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/commonsense_qa")
    return dataset

@dataset_dict.add("cosmos_qa")
def get_cosmos_qa():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/cosmos_qa"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/cosmos_qa")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/cosmos_qa")
    return dataset

@dataset_dict.add("copa")
def get_copa():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/copa"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/copa")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/copa")
    return dataset

@dataset_dict.add("balanced_copa")
def get_balanced_copa():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/balanced_copa"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/balanced_copa")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/balanced_copa")
    return dataset

@dataset_dict.add("arc_easy")
def get_arc_easy():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/arc_easy"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/arc_easy")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/arc_easy")
    return dataset

@dataset_dict.add("social_i_qa")
def get_social_i_qa():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/social_i_qa"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/social_i_qa")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/social_i_qa")
    return dataset

@dataset_dict.add("piqa")
def get_piqa():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/piqa"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/piqa")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/piqa")
    return dataset

@dataset_dict.add("cs_explan")
def get_cs_explan():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/cs_explan"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/cs_explan")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/cs_explan")
    return dataset

@dataset_dict.add("cs_valid")
def get_cs_valid():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/cs_valid"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/cs_valid")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/cs_valid")
    return dataset

@dataset_dict.add("hellaswag")
def get_hellaswag():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/hellaswag"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/hellaswag")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/hellaswag")
    return dataset

@dataset_dict.add("openbookqa")
def get_openbookqa():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/openbookqa"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/openbookqa")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/openbookqa")
    return dataset


@dataset_dict.add("race")
def get_race():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/race"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/race")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/race")
    return dataset


@dataset_dict.add("java")
def get_java():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/java"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/java")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/java")
    return dataset

@dataset_dict.add("javascript")
def get_javascript():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/javascript"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/javascript")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/javascript")
    return dataset


@dataset_dict.add("ruby")
def get_ruby():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/ruby"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/ruby")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/ruby")
    return dataset

@dataset_dict.add("reddit")
def get_reddit():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/reddit"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/reddit")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/reddit")
    return dataset

@dataset_dict.add("multinews")
def get_multinews():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/multinews"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/multinews")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/multinews")
    return dataset


@dataset_dict.add("pubmed")
def get_pubmed():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/pubmed"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/pubmed")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/pubmed")
    return dataset

@dataset_dict.add("wikihow")
def get_wikihow():
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data/wikihow"):
        dataset = datasets.load_from_disk("/remote-home/klv/exps/rtv_icl/data/wikihow")
    else:
        dataset = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/wikihow")
    return dataset


fields_dict = {
    "break":{"question_attr":"question_text","answers_attr":"decomposition"},
    "mtop":{"question_attr":"question","answers_attr":"logical_form"},
    "smcalflow":{"question_attr":"user_utterance","answers_attr":"lispress"},
    "kp20k":{"question_attr":"document","answers_attr":"extractive_keyphrases"},
    # "kp20k": {"question_attr": "document", "answers_attr": "abstractive_keyphrases"},
    "dwiki":{"question_attr":"src","answers_attr":"tgt"},
    "wikiauto":{"question_attr":"source","answers_attr":"target"},
    "iwslt":{"question_attr":"translation.de","answers_attr":"translation.en"},
    "squadv2":{"question_attr":"input","answers_attr":"target"},
    "opusparcus":{"question_attr":"input","answers_attr":"target"},
    "common_gen":{"question_attr":"joined_concepts","answers_attr":"target"},
    "xsum": {"question_attr": "document", "answers_attr": "summary"},
    "spider": {"question_attr": "question", "answers_attr": "query"},
    "iwslt_en_fr": {"question_attr": "question", "answers_attr": "target"},
    "iwslt_en_de": {"question_attr": "question", "answers_attr": "target"},
    "wmt_en_de": {"question_attr": "question", "answers_attr": "target"},
    "wmt_de_en": {"question_attr": "question", "answers_attr": "target"},
    "e2e": {"question_attr": "question", "answers_attr": "target"},
    "dart": {"question_attr": "question", "answers_attr": "target"},
    "totto": {"question_attr": "question", "answers_attr": "target"},
    "cnndailymail": {"question_attr": "article", "answers_attr": "highlights"},
    "python": {"question_attr": "question", "answers_attr": "target"},
    "go": {"question_attr": "question", "answers_attr": "target"},
    "php": {"question_attr": "question", "answers_attr": "target"},
    "java": {"question_attr": "question", "answers_attr": "target"},
    "javascript": {"question_attr": "question", "answers_attr": "target"},
    "ruby": {"question_attr": "question", "answers_attr": "target"},
    "reddit": {"question_attr": "question", "answers_attr": "target"},
    "multinews": {"question_attr": "question", "answers_attr": "target"},
    "wikihow": {"question_attr": "question", "answers_attr": "target"},
    "pubmed": {"question_attr": "question", "answers_attr": "target"},
    "roc_ending_generation": {"question_attr": "question", "answers_attr": "target"},
    "roc_story_generation": {"question_attr": "question", "answers_attr": "target"},
    "trec": {"question_attr": "sentence", "answers_attr": "label"},
    "sst2": {"question_attr": "sentence", "answers_attr": "label"},
    "imdb": {"question_attr": "sentence", "answers_attr": "label"},
    "tweet_sentiment_extraction": {"question_attr": "sentence", "answers_attr": "label"},
    "financial_phrasebank": {"question_attr": "sentence", "answers_attr": "label"},
    "emotion": {"question_attr": "sentence", "answers_attr": "label"},
    "mnli": {"question_attr": "sentence", "answers_attr": "label"},
    "cola": {"question_attr": "sentence", "answers_attr": "label"},
    "qnli": {"question_attr": "sentence", "answers_attr": "label"},
    "mrpc": {"question_attr": "sentence", "answers_attr": "label"},
    "boolq": {"question_attr": "sentence", "answers_attr": "label"},
    "qqp": {"question_attr": "sentence", "answers_attr": "label"},
    "wnli": {"question_attr": "sentence", "answers_attr": "label"},
    "snli": {"question_attr": "sentence", "answers_attr": "label"},
    "rte": {"question_attr": "sentence", "answers_attr": "label"},
    "sst5": {"question_attr": "sentence", "answers_attr": "label"},
    "cr": {"question_attr": "sentence", "answers_attr": "label"},
    "mr": {"question_attr": "sentence", "answers_attr": "label"},
    "subj": {"question_attr": "sentence", "answers_attr": "label"},
    "yelp_full": {"question_attr": "sentence", "answers_attr": "label"},
    "amazon": {"question_attr": "sentence", "answers_attr": "label"},
    "agnews": {"question_attr": "sentence", "answers_attr": "label"},
    "amazon_scenario": {"question_attr": "sentence", "answers_attr": "label"},
    "mtop_domain": {"question_attr": "sentence", "answers_attr": "label"},
    "dbpedia": {"question_attr": "sentence", "answers_attr": "label"},
    "bank77": {"question_attr": "sentence", "answers_attr": "label"},
    "yahoo": {"question_attr": "sentence", "answers_attr": "label"},
    "commonsense_qa": {"question_attr": "question", "answers_attr": "label"},
    "cs_explan": {"question_attr": "question", "answers_attr": "label"},
    "cosmos_qa": {"question_attr": "question", "answers_attr": "label"},
    "copa": {"question_attr": "question", "answers_attr": "label"},
    "balanced_copa": {"question_attr": "question", "answers_attr": "label"},
    "arc_easy": {"question_attr": "question", "answers_attr": "label"},
    "social_i_qa": {"question_attr": "question", "answers_attr": "label"},
    "piqa": {"question_attr": "question", "answers_attr": "label"},
    "race": {"question_attr": "question", "answers_attr": "label"},
    "cs_valid": {"question_attr": "question", "answers_attr": "label"},
    "hellaswag": {"question_attr": "question", "answers_attr": "label"},
    "openbookqa": {"question_attr": "question", "answers_attr": "label"},

}




class EPRQASrc(QASrc):
    def __init__(
        self,
        dataset_split,
        task_name,
        ds_size=None,
        file =  "",
        selector: DictConfig = None,
        question_attr: str = "question",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.task_name = task_name
        
        self.dataset_split = dataset_split
        # assert  self.dataset_split in ["train","validation","test","debug","test_asset","test_turk","test_wiki"]
        self.dataset = dataset_dict.functions[self.task_name]()
        if self.dataset_split=="train":
            self.data = load_train_dataset(self.dataset,size=ds_size)
        else:
            self.data = list(self.dataset[self.dataset_split])
        if ds_size is not None:
            assert len(self.data) == ds_size 
        
        self.question_attr = fields_dict[self.task_name]["question_attr"]
        self.answers_attr = fields_dict[self.task_name]["answers_attr"]
        self.id_attr = id_attr

    def load_data(self):
        # super().load_data()
        data = []
        # with jsonlines.open(self.file, mode="r") as jsonl_reader:
        for id, jline in enumerate(self.data):
            question = jline[self.question_attr]
            answers = [str(jline[self.answers_attr])]
            # id = None
            # if self.id_attr in jline:
                # id = jline[self.id_attr]
            data.append(QASample(self._process_question(question), id, answers))
        self.data = data




def reformat(text):
    return " ".join([f"{i+1}#) {x.strip()}" for i,x in enumerate(text.split(";"))])

app = App()
@app.add("break_q")
def get_break_question(entry):
    if "question" in entry:
        question = entry['question']
    else:
        question = entry['question_text']
    return "Parse the sentence into logical form: " + question

@app.add("break_qa")
def get_break_question_decomp(entry):
    if "question" in entry:
        question = entry['question']
    else:
        question = entry['question_text']
    return f"Parse the sentence into logical form: {question}\t{reformat(entry['decomposition'])}"

@app.add("break_a")
def get_break_decomp(entry):
    return reformat(entry['decomposition'])

@app.add("mtop_q")
def get_mtop_question(entry):
    return "Parse the sentence into logical form: " + entry['question']


@app.add("mtop_qa")
def get_mtop_question_decomp(entry):
    return f"Parse the sentence into logical form: {entry['question']}\t{entry['logical_form']}"

@app.add("mtop_a")
def get_mtop_decomp(entry):
    return entry['logical_form']


@app.add("smcalflow_q")
def get_smcalflow_question(entry):
    return "Parse the sentence into logical form: " + entry['user_utterance']

@app.add("smcalflow_qa")
def get_smcalflow_question_decomp(entry):
    return f"Parse the sentence into logical form: {entry['user_utterance']}\t{entry['lispress']}"

@app.add("smcalflow_a")
def get_smcalflow_decomp(entry):
    return entry['lispress']


@app.add("kp20k_q")
def get_kp20k_question(entry):
    return entry['document']

@app.add("kp20k_qa")
def get_kp20k_question_decomp(entry):
    return f"{entry['document']}\t{entry['abstractive_keyphrases']}"
    # return f"{entry['document']}\t{entry['extractive_keyphrases']}"

@app.add("kp20k_a")
def get_kp20k_decomp(entry):
    # return entry['extractive_keyphrases']
    return entry['abstractive_keyphrases']

@app.add("dwiki_q")
def get_dwiki_question(entry):
    return entry['src']

@app.add("dwiki_qa")
def get_dwiki_question_decomp(entry):
    return f"{entry['src']}\t{entry['tgt']}"

@app.add("dwiki_a")
def get_dwiki_decomp(entry):
    return entry['tgt']

@app.add("wikiauto_q")
def get_wikiauto_question(entry):
    return "Simplify the text: " + entry['source']

@app.add("wikiauto_qa")
def get_wikiauto_question_decomp(entry):
    return f"Simplify the text: {entry['source']}\tSimplified text: {entry['target']}"

@app.add("wikiauto_a")
def get_wikiauto_decomp(entry):
    return entry['target']

@app.add("iwslt_q")
def get_iwslt_question(entry):
    return entry['translation.de']

@app.add("iwslt_qa")
def get_iwslt_question_decomp(entry):
    return f"German: {entry['translation.de']}\tEnglish: {entry['translation.en']}"

@app.add("iwslt_a")
def get_iwslt_decomp(entry):
    return entry['translation.en']

@app.add("squadv2_q")
def get_squadv2_question(entry):
    return entry['input']

@app.add("squadv2_qa")
def get_squadv2_question_decomp(entry):
    return f"{entry['input']}\tThe question corresponding to this answer is as follows. {entry['target']}"

@app.add("squadv2_a")
def get_squadv2_decomp(entry):
    return entry['target']

@app.add("opusparcus_q")
def get_opusparcus_question(entry):
    return entry['input']

@app.add("opusparcus_qa")
def get_opusparcus_question_decomp(entry):
    return f"Paraphrase the text: {entry['input']}\tParaphrase: {entry['target']}"

@app.add("opusparcus_a")
def get_opusparcus_decomp(entry):
    return entry['target']

@app.add("common_gen_q")
def get_common_gen_question(entry):
    return "Generate a sentence using these concepts: " + entry['joined_concepts']

@app.add("common_gen_qa")
def get_common_gen_question_decomp(entry):
    return f"Generate a sentence using these concepts: {entry['joined_concepts']}\tGenerated sentence: {entry['target']}"

@app.add("common_gen_a")
def get_common_gen_decomp(entry):
    return entry['target']

@app.add("xsum_q")
def get_xsum_question(entry):
    return entry['document']

@app.add("xsum_qa")
def get_xsum_question_decomp(entry):
    return f"{entry['document']}\tTL;DR: {entry['summary']}"

@app.add("xsum_a")
def get_xsum_decomp(entry):
    return entry['summary']


@app.add("spider_q")
def get_spider_question(entry):
    return entry['question']

@app.add("spider_qa")
def get_spider_question_decomp(entry):
    return f"{entry['question']}\t{entry['query']}"

@app.add("spider_a")
def get_spider_decomp(entry):
    return entry['query']

@app.add("iwslt_en_fr_q")
def get_iwslt_en_fr_question(entry):
    return entry['question']

@app.add("iwslt_en_fr_qa")
def get_iwslt_en_fr_question_decomp(entry):
    return f"English: {entry['question']}\tFrench: {entry['target']}"

@app.add("iwslt_en_fr_a")
def get_iwslt_en_fr_decomp(entry):
    return entry['target']

@app.add("iwslt_en_de_q")
def get_iwslt_en_de_question(entry):
    return entry['question']

@app.add("iwslt_en_de_qa")
def get_iwslt_en_de_question_decomp(entry):
    return f"English: {entry['question']}\tGerman: {entry['target']}"

@app.add("iwslt_en_de_a")
def get_iwslt_en_de_decomp(entry):
    return entry['target']

@app.add("wmt_en_de_q")
def get_wmt_en_de_question(entry):
    return entry['question']

@app.add("wmt_en_de_qa")
def get_wmt_en_de_question_decomp(entry):
    return f"English: {entry['question']}\tGerman: {entry['target']}"

@app.add("wmt_en_de_a")
def get_wmt_en_de_decomp(entry):
    return entry['target']

@app.add("wmt_de_en_q")
def get_wmt_de_en_question(entry):
    return entry['question']

@app.add("wmt_de_en_qa")
def get_wmt_de_en_question_decomp(entry):
    return f"German: {entry['question']}\tEnglish: {entry['target']}"

@app.add("wmt_de_en_a")
def get_wmt_de_en_decomp(entry):
    return entry['target']

@app.add("roc_ending_generation_q")
def get_roc_ending_generation_question(entry):
    return "An unfinished story: " + entry['question']

@app.add("roc_ending_generation_qa")
def get_roc_ending_generation_question_decomp(entry):
    return f"An unfinished story: {entry['question']}\tEnd of the story: {entry['target']}"

@app.add("roc_ending_generation_a")
def get_roc_ending_generation_decomp(entry):
    return entry['target']

@app.add("roc_story_generation_q")
def get_roc_story_generation_question(entry):
    return "Beginning of the story: " + entry['question']

@app.add("roc_story_generation_qa")
def get_roc_story_generation_question_decomp(entry):
    return f"Beginning of the story: {entry['question']}\tRest of the story: {entry['target']}"

@app.add("roc_story_generation_a")
def get_roc_story_generation_decomp(entry):
    return entry['target']

@app.add("e2e_q")
def get_e2e_question(entry):
    return "Describe the table in natural language. Table: " + entry['question']

@app.add("e2e_qa")
def get_e2e_question_decomp(entry):
    return f"Describe the table in natural language. Table: {entry['question']}\tSentence: {entry['target']}"

@app.add("e2e_a")
def get_e2e_decomp(entry):
    return entry['target']

@app.add("dart_q")
def get_dart_question(entry):
    return "Describe the table in natural language. Table: " + entry['question']

@app.add("dart_qa")
def get_dart_question_decomp(entry):
    return f"Describe the table in natural language. Table: {entry['question']}\tSentence: {entry['target']}"

@app.add("dart_a")
def get_dart_decomp(entry):
    return entry['target']

@app.add("totto_q")
def get_totto_question(entry):
    return entry['question']

@app.add("totto_qa")
def get_totto_question_decomp(entry):
    return f"Table: {entry['question']}\tSentence: {entry['target']}"

@app.add("totto_a")
def get_totto_decomp(entry):
    return entry['target']

@app.add("cnndailymail_q")
def get_cnndailymail_question(entry):
    return "Summarize the text: " + entry['article']

@app.add("cnndailymail_qa")
def get_cnndailymail_question_decomp(entry):
    return f"Summarize the text: {entry['article']}\tTL;DR: {entry['highlights']}"

@app.add("cnndailymail_a")
def get_cnndailymail_decomp(entry):
    return entry['highlights']

@app.add("python_q")
def get_python_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("python_qa")
def get_python_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("python_a")
def get_python_decomp(entry):
    return entry['target']

@app.add("go_q")
def get_go_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("go_qa")
def get_go_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("go_a")
def get_go_decomp(entry):
    return entry['target']

@app.add("php_q")
def get_php_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("php_qa")
def get_php_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("php_a")
def get_php_decomp(entry):
    return entry['target']

@app.add("trec_q")
def get_trec_question(entry):
    return "Topic of the question: " + entry['sentence']

@app.add("trec_qa")
def get_trec_question_decomp(entry):
    a = get_one_prompt('trec', 3, entry['label'])
    return f"Topic of the question: {entry['sentence']}\t{a}"

@app.add("trec_a")
def get_trec_decomp(entry):
    return get_one_prompt('trec', 3, entry['label'])

@app.add("sst2_q")
def get_sst2_question(entry):
    return "Sentiment of the sentence: " + entry['sentence']

@app.add("sst2_qa")
def get_sst2_question_decomp(entry):
    a = get_one_prompt('sst2', 1, entry['label'])
    return f"Sentiment of the sentence: {entry['sentence']}\t{a}"

@app.add("sst2_a")
def get_sst2_decomp(entry):
    return get_one_prompt('sst2', 1, entry['label'])

@app.add("imdb_q")
def get_imdb_question(entry):
    return "Sentiment of the sentence: " + entry['sentence']

@app.add("imdb_qa")
def get_imdb_question_decomp(entry):
    a = get_one_prompt('imdb', 1, entry['label'])
    return f"Sentiment of the sentence: {entry['sentence']}\t{a}"

@app.add("imdb_a")
def get_imdb_decomp(entry):
    return get_one_prompt('imdb', 1, entry['label'])

@app.add("tweet_sentiment_extraction_q")
def get_tweet_sentiment_extraction_question(entry):
    return "Sentiment of the sentence: " + entry['sentence']

@app.add("tweet_sentiment_extraction_qa")
def get_tweet_sentiment_extraction_question_decomp(entry):
    a = get_one_prompt('tweet_sentiment_extraction', 1, entry['label'])
    return f"Sentiment of the sentence: {entry['sentence']}\t{a}"

@app.add("tweet_sentiment_extraction_a")
def get_tweet_sentiment_extraction_decomp(entry):
    return get_one_prompt('tweet_sentiment_extraction', 1, entry['label'])

@app.add("financial_phrasebank_q")
def get_financial_phrasebank_question(entry):
    return "Sentiment of the sentence: " + entry['sentence']

@app.add("financial_phrasebank_qa")
def get_financial_phrasebank_question_decomp(entry):
    a = get_one_prompt('financial_phrasebank', 1, entry['label'])
    return f"Sentiment of the sentence: {entry['sentence']}\t{a}"

@app.add("financial_phrasebank_a")
def get_financial_phrasebank_decomp(entry):
    return get_one_prompt('financial_phrasebank', 1, entry['label'])

@app.add("emotion_q")
def get_emotion_question(entry):
    return "Sentiment of the sentence: " + entry['sentence']

@app.add("emotion_qa")
def get_emotion_question_decomp(entry):
    a = get_one_prompt('emotion', 1, entry['label'])
    return f"Sentiment of the sentence: {entry['sentence']}\t{a}"

@app.add("emotion_a")
def get_emotion_decomp(entry):
    return get_one_prompt('emotion', 1, entry['label'])

@app.add("mnli_q")
def get_mnli_question(entry):
    return "Recognizing textual entailment between these 2 texts. " + entry['sentence']

@app.add("mnli_qa")
def get_mnli_question_decomp(entry):
    a = get_one_prompt('mnli', 0, entry['label'])
    return f"Recognizing textual entailment between these 2 texts. {entry['sentence']}\t{a}"

@app.add("mnli_a")
def get_mnli_decomp(entry):
    return get_one_prompt('mnli', 0, entry['label'])


@app.add("cola_q")
def get_cola_question(entry):
    return "The grammaticality of this sentence: " + entry['sentence']

@app.add("cola_qa")
def get_cola_question_decomp(entry):
    a = get_one_prompt('cola', 1, entry['label'])
    return f"The grammaticality of this sentence: {entry['sentence']}\t{a}"

@app.add("cola_a")
def get_cola_decomp(entry):
    return get_one_prompt('cola', 1, entry['label'])

@app.add("qnli_q")
def get_qnli_question(entry):
    return "Recognizing textual entailment between these 2 texts. " + entry['sentence']

@app.add("qnli_qa")
def get_qnli_question_decomp(entry):
    a = get_one_prompt('qnli', 0, entry['label'])
    return f"Recognizing textual entailment between these 2 texts. {entry['sentence']}\t{a}"

@app.add("qnli_a")
def get_qnli_decomp(entry):
    return get_one_prompt('qnli', 0, entry['label'])


@app.add("mrpc_q")
def get_mrpc_question(entry):
    return "Recognizing textual entailment between these 2 texts. " + entry['sentence']

@app.add("mrpc_qa")
def get_mrpc_question_decomp(entry):
    a = get_one_prompt('mrpc', 0, entry['label'])
    return f"Recognizing textual entailment between these 2 texts. {entry['sentence']}\t{a}"

@app.add("mrpc_a")
def get_mrpc_decomp(entry):
    return get_one_prompt('mrpc', 0, entry['label'])

@app.add("boolq_q")
def get_boolq_question(entry):
    return "Answer the question based on the text. " + entry['sentence']

@app.add("boolq_qa")
def get_boolq_question_decomp(entry):
    a = get_one_prompt('boolq', 0, entry['label'])
    return f"Answer the question based on the text. {entry['sentence']}\t{a}"

@app.add("boolq_a")
def get_boolq_decomp(entry):
    return get_one_prompt('boolq', 0, entry['label'])

@app.add("qqp_q")
def get_qqp_question(entry):
    return "Recognizing textual entailment between these 2 texts. " + entry['sentence']

@app.add("qqp_qa")
def get_qqp_question_decomp(entry):
    a = get_one_prompt('qqp', 0, entry['label'])
    return f"Recognizing textual entailment between these 2 texts. {entry['sentence']}\t{a}"

@app.add("qqp_a")
def get_qqp_decomp(entry):
    return get_one_prompt('qqp', 0, entry['label'])

@app.add("wnli_q")
def get_wnli_question(entry):
    return "Recognizing textual entailment between these 2 texts. " + entry['sentence']

@app.add("wnli_qa")
def get_wnli_question_decomp(entry):
    a = get_one_prompt('wnli', 0, entry['label'])
    return f"Recognizing textual entailment between these 2 texts. {entry['sentence']}\t{a}"

@app.add("wnli_a")
def get_wnli_decomp(entry):
    return get_one_prompt('wnli', 0, entry['label'])

@app.add("snli_q")
def get_snli_question(entry):
    return "Recognizing textual entailment between these 2 texts. " + entry['sentence']

@app.add("snli_qa")
def get_snli_question_decomp(entry):
    a = get_one_prompt('snli', 0, entry['label'])
    return f"Recognizing textual entailment between these 2 texts. {entry['sentence']}\t{a}"

@app.add("snli_a")
def get_snli_decomp(entry):
    return get_one_prompt('snli', 0, entry['label'])


@app.add("rte_q")
def get_rte_question(entry):
    return "Recognizing textual entailment between these 2 texts. " + entry['sentence']

@app.add("rte_qa")
def get_rte_question_decomp(entry):
    a = get_one_prompt('rte', 0, entry['label'])
    return f"Recognizing textual entailment between these 2 texts. {entry['sentence']}\t{a}"

@app.add("rte_a")
def get_rte_decomp(entry):
    return get_one_prompt('rte', 0, entry['label'])


@app.add("cr_q")
def get_cr_question(entry):
    return "Sentiment of the sentence: " + entry['sentence']

@app.add("cr_qa")
def get_cr_question_decomp(entry):
    a = get_one_prompt('cr', 1, entry['label'])
    return f"Sentiment of the sentence: {entry['sentence']}\t{a}"

@app.add("cr_a")
def get_cr_decomp(entry):
    return get_one_prompt('cr', 1, entry['label'])

@app.add("mr_q")
def get_mr_question(entry):
    return "Sentiment of the sentence: " + entry['sentence']

@app.add("mr_qa")
def get_mr_question_decomp(entry):
    a = get_one_prompt('mr', 1, entry['label'])
    return f"Sentiment of the sentence: {entry['sentence']}\t{a}"

@app.add("mr_a")
def get_mr_decomp(entry):
    return get_one_prompt('mr', 1, entry['label'])


@app.add("yelp_full_q")
def get_yelp_full_question(entry):
    return "Sentiment of the sentence: " + entry['sentence']

@app.add("yelp_full_qa")
def get_yelp_full_question_decomp(entry):
    a = get_one_prompt('yelp_full', 1, entry['label'])
    return f"Sentiment of the sentence: {entry['sentence']}\t{a}"

@app.add("yelp_full_a")
def get_yelp_full_decomp(entry):
    return get_one_prompt('yelp_full', 1, entry['label'])

@app.add("amazon_q")
def get_amazon_question(entry):
    return "Sentiment of the sentence: " + entry['sentence']

@app.add("amazon_qa")
def get_amazon_question_decomp(entry):
    a = get_one_prompt('amazon', 1, entry['label'])
    return f"Sentiment of the sentence: {entry['sentence']}\t{a}"

@app.add("amazon_a")
def get_amazon_decomp(entry):
    return get_one_prompt('amazon', 1, entry['label'])

@app.add("agnews_q")
def get_agnews_question(entry):
    return "Topic of the text: " + entry['sentence']

@app.add("agnews_qa")
def get_agnews_question_decomp(entry):
    a = get_one_prompt('agnews', 0, entry['label'])
    return f"Topic of the text: {entry['sentence']}\t{a}"

@app.add("agnews_a")
def get_agnews_decomp(entry):
    return get_one_prompt('agnews', 0, entry['label'])

@app.add("amazon_scenario_q")
def get_amazon_scenario_question(entry):
    return "Topic of the text: " + entry['sentence']

@app.add("amazon_scenario_qa")
def get_amazon_scenario_question_decomp(entry):
    a = get_one_prompt('amazon_scenario', 0, entry['label'])
    return f"Topic of the text: {entry['sentence']}\t{a}"

@app.add("amazon_scenario_a")
def get_amazon_scenario_decomp(entry):
    return get_one_prompt('amazon_scenario', 0, entry['label'])

@app.add("mtop_domain_q")
def get_mtop_domain_question(entry):
    return "Topic of the text: " + entry['sentence']

@app.add("mtop_domain_qa")
def get_mtop_domain_question_decomp(entry):
    a = get_one_prompt('mtop_domain', 0, entry['label'])
    return f"Topic of the text: {entry['sentence']}\t{a}"

@app.add("mtop_domain_a")
def get_mtop_domain_decomp(entry):
    return get_one_prompt('mtop_domain', 0, entry['label'])

@app.add("dbpedia_q")
def get_dbpedia_question(entry):
    return "Topic of the text: " + entry['sentence']

@app.add("dbpedia_qa")
def get_dbpedia_question_decomp(entry):
    a = get_one_prompt('dbpedia', 0, entry['label'])
    return f"Topic of the text: {entry['sentence']}\t{a}"

@app.add("dbpedia_a")
def get_dbpedia_decomp(entry):
    return get_one_prompt('dbpedia', 0, entry['label'])

@app.add("bank77_q")
def get_bank77_question(entry):
    return "Topic of the text: " + entry['sentence']

@app.add("bank77_qa")
def get_bank77_question_decomp(entry):
    a = get_one_prompt('bank77', 0, entry['label'])
    return f"Topic of the text: {entry['sentence']}\t{a}"

@app.add("bank77_a")
def get_bank77_decomp(entry):
    return get_one_prompt('bank77', 0, entry['label'])

@app.add("yahoo_q")
def get_yahoo_question(entry):
    return "Topic of the text: " + entry['sentence']

@app.add("yahoo_qa")
def get_yahoo_question_decomp(entry):
    a = get_one_prompt('yahoo', 0, entry['label'])
    return f"Topic of the text: {entry['sentence']}\t{a}"

@app.add("yahoo_a")
def get_yahoo_decomp(entry):
    return get_one_prompt('yahoo', 0, entry['label'])

@app.add("subj_q")
def get_subj_question(entry):
    return "Subjectivity status of the sentence: " + entry['sentence']

@app.add("subj_qa")
def get_subj_question_decomp(entry):
    a = get_one_prompt('subj', 2, entry['label'])
    return f"Subjectivity status of the sentence: {entry['sentence']}\t{a}"

@app.add("subj_a")
def get_subj_decomp(entry):
    return get_one_prompt('subj', 2, entry['label'])


@app.add("sst5_q")
def get_sst5_question(entry):
    return "Sentiment of the sentence: " + entry['sentence']

@app.add("sst5_qa")
def get_sst5_question_decomp(entry):
    a = get_one_prompt('sst5', 1, entry['label'])
    return f"Sentiment of the sentence: {entry['sentence']}\t{a}"

@app.add("sst5_a")
def get_sst5_decomp(entry):
    return get_one_prompt('sst5', 1, entry['label'])


@app.add("commonsense_qa_q")
def get_commonsense_qa_question(entry):
    return entry['question']

@app.add("commonsense_qa_qa")
def get_commonsense_qa_question_decomp(entry):
    return f"{entry['question']}\tAnswer: {entry['label']}"

@app.add("commonsense_qa_a")
def get_commonsense_qa_decomp(entry):
    return entry['label']

@app.add("cs_explan_q")
def get_cs_explan_question(entry):
    return entry['question']

@app.add("cs_explan_qa")
def get_cs_explan_question_decomp(entry):
    return f"{entry['question']}\tAnswer: {entry['label']}"

@app.add("cs_explan_a")
def get_cs_explan_decomp(entry):
    return entry['label']

@app.add("cs_valid_q")
def get_cs_valid_question(entry):
    return entry['question']

@app.add("cs_valid_qa")
def get_cs_valid_question_decomp(entry):
    return f"{entry['question']}\tAnswer: {entry['label']}"

@app.add("cs_valid_a")
def get_cs_valid_decomp(entry):
    return entry['label']

@app.add("hellaswag_q")
def get_hellaswag_question(entry):
    return entry['question']

@app.add("hellaswag_qa")
def get_hellaswag_question_decomp(entry):
    return f"{entry['question']} {entry['label']}"

@app.add("hellaswag_a")
def get_hellaswag_decomp(entry):
    return entry['label']

@app.add("openbookqa_q")
def get_openbookqa_question(entry):
    return entry['question']

@app.add("openbookqa_qa")
def get_openbookqa_question_decomp(entry):
    return f"{entry['question']} {entry['label']}"

@app.add("openbookqa_a")
def get_openbookqa_decomp(entry):
    return entry['label']

@app.add("race_q")
def get_race_question(entry):
    return entry['question']

@app.add("race_qa")
def get_race_question_decomp(entry):
    return f"{entry['question']}\tAnswer: {entry['label']}"

@app.add("race_a")
def get_race_decomp(entry):
    return entry['label']

@app.add("cosmos_qa_q")
def get_cosmos_qa_question(entry):
    return "Answer the question based on the text. " + entry['question']

@app.add("cosmos_qa_qa")
def get_cosmos_qa_question_decomp(entry):
    return f"Answer the question based on the text. {entry['question']}\tAnswer: {entry['label']}"

@app.add("cosmos_qa_a")
def get_cosmos_qa_decomp(entry):
    return entry['label']

@app.add("copa_q")
def get_copa_question(entry):
    return "Answer the question based on the text. " + entry['question']

@app.add("copa_qa")
def get_copa_question_decomp(entry):
    return f"Answer the question based on the text. {entry['question']}\tAnswer: {entry['label']}"

@app.add("copa_a")
def get_copa_decomp(entry):
    return entry['label']

@app.add("balanced_copa_q")
def get_balanced_copa_question(entry):
    return entry['question']

@app.add("balanced_copa_qa")
def get_balanced_copa_question_decomp(entry):
    return f"{entry['question']}\tAnswer: {entry['label']}"

@app.add("balanced_copa_a")
def get_balanced_copa_decomp(entry):
    return entry['label']

@app.add("arc_easy_q")
def get_arc_easy_question(entry):
    return entry['question']

@app.add("arc_easy_qa")
def get_arc_easy_question_decomp(entry):
    return f"{entry['question']}\tAnswer: {entry['label']}"

@app.add("arc_easy_a")
def get_arc_easy_decomp(entry):
    return entry['label']


@app.add("piqa_q")
def get_piqa_question(entry):
    return entry['question']

@app.add("piqa_qa")
def get_piqa_question_decomp(entry):
    return f"{entry['question']}\tAnswer: {entry['label']}"

@app.add("piqa_a")
def get_piqa_decomp(entry):
    return entry['label']

@app.add("social_i_qa_q")
def get_social_i_qa_question(entry):
    return entry['question']

@app.add("social_i_qa_qa")
def get_social_i_qa_question_decomp(entry):
    return f"{entry['question']}\tAnswer: {entry['label']}"

@app.add("social_i_qa_a")
def get_social_i_qa_decomp(entry):
    return entry['label']


@app.add("java_q")
def get_java_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("java_qa")
def get_java_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("java_a")
def get_java_decomp(entry):
    return entry['target']

@app.add("javascript_q")
def get_javascript_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("javascript_qa")
def get_javascript_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("javascript_a")
def get_javascript_decomp(entry):
    return entry['target']

@app.add("ruby_q")
def get_ruby_question(entry):
    return "Comment on the code. Code: " + entry['question']

@app.add("ruby_qa")
def get_ruby_question_decomp(entry):
    return f"Comment on the code. Code: {entry['question']}\tComment: {entry['target']}"

@app.add("ruby_a")
def get_ruby_decomp(entry):
    return entry['target']

@app.add("reddit_q")
def get_reddit_question(entry):
    return "Summarize the text: " + entry['question']

@app.add("reddit_qa")
def get_reddit_question_decomp(entry):
    return f"Summarize the text: {entry['question']}\tTL;DR: {entry['target']}"

@app.add("reddit_a")
def get_reddit_decomp(entry):
    return entry['target']


@app.add("multinews_q")
def get_multinews_question(entry):
    return entry['question']

@app.add("multinews_qa")
def get_multinews_question_decomp(entry):
    return f"{entry['question']}\tTL;DR: {entry['target']}"

@app.add("multinews_a")
def get_multinews_decomp(entry):
    return entry['target']


@app.add("pubmed_q")
def get_pubmed_question(entry):
    return "Summarize the text: " + entry['question']

@app.add("pubmed_qa")
def get_pubmed_question_decomp(entry):
    return f"Summarize the text: {entry['question']}\tTL;DR: {entry['target']}"

@app.add("pubmed_a")
def get_pubmed_decomp(entry):
    return entry['target']

@app.add("wikihow_q")
def get_wikihow_question(entry):
    return entry['question']

@app.add("wikihow_qa")
def get_wikihow_question_decomp(entry):
    return f"{entry['question']}\tTL;DR: {entry['target']}"

@app.add("wikihow_a")
def get_wikihow_decomp(entry):
    return entry['target']


class EPRCtxSrc(RetrieverData):
    def __init__(
        self,
        task_name,
        setup_type,
        ds_size=None,
        file = "",
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.setup_type = setup_type
        assert self.setup_type in ["q","qa","a"]
        self.file = file
        self.task_name = task_name
        self.dataset = dataset_dict.functions[self.task_name]()
        self.get_field = app.functions[f"{self.task_name}_{self.setup_type}"]
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.data = load_train_dataset(self.dataset,size=ds_size)
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        # with open(self.file) as ifile:
        #     reader = json.load(ifile)
        for sample_id,entry in enumerate(self.data):
            passage = self.get_field(entry)
            if self.normalize:
                passage = normalize_passage(passage)
            ctxs[sample_id] = BiEncoderPassage(passage, "")

class JsonCtxSrc(RetrieverData):
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        with open(self.file) as ifile:
            reader = json.load(ifile)
            for row in reader:
                sample_id = row["id"]
                passage = row['text']
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row['title'])


class KiltCsvCtxSrc(CsvCtxSrc):
    def __init__(
        self,
        file: str,
        mapping_file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(
            file, id_col, text_col, title_col, id_prefix, normalize=normalize
        )
        self.mapping_file = mapping_file

    def convert_to_kilt(self, kilt_gold_file, dpr_output, kilt_out_file):
        logger.info("Converting to KILT format file: %s", dpr_output)

        with open(dpr_output, "rt") as fin:
            dpr_output = json.load(fin)

        with jsonlines.open(kilt_gold_file, "r") as reader:
            kilt_gold_file = list(reader)
        assert len(kilt_gold_file) == len(dpr_output)
        map_path = self.mapping_file
        with open(map_path, "rb") as fin:
            mapping = pickle.load(fin)

        with jsonlines.open(kilt_out_file, mode="w") as writer:
            for dpr_entry, kilt_gold_entry in zip(dpr_output, kilt_gold_file):
                assert dpr_entry["question"] == kilt_gold_entry["input"]
                provenance = []
                for ctx in dpr_entry["ctxs"]:
                    wikipedia_id, end_paragraph_id = mapping[int(ctx["id"])]
                    provenance.append(
                        {
                            "wikipedia_id": wikipedia_id,
                            "end_paragraph_id": end_paragraph_id,
                        }
                    )
                kilt_entry = {
                    "id": kilt_gold_entry["id"],
                    "input": dpr_entry["question"],
                    "output": [{"provenance": provenance}],
                }
                writer.write(kilt_entry)

        logger.info("Saved KILT formatted results to: %s", kilt_out_file)


class JsonlTablesCtxSrc(object):
    def __init__(
        self,
        file: str,
        tables_chunk_sz: int = 100,
        split_type: str = "type1",
        id_prefix: str = None,
    ):
        self.tables_chunk_sz = tables_chunk_sz
        self.split_type = split_type
        self.file = file
        self.id_prefix = id_prefix

    def load_data_to(self, ctxs: Dict):
        docs = {}
        logger.info("Parsing Tables data from: %s", self.file)
        tables_dict = read_nq_tables_jsonl(self.file)
        table_chunks = split_tables_to_chunks(
            tables_dict, self.tables_chunk_sz, split_type=self.split_type
        )
        for chunk in table_chunks:
            sample_id = self.id_prefix + str(chunk[0])
            docs[sample_id] = TableChunk(chunk[1], chunk[2], chunk[3])
        logger.info("Loaded %d tables chunks", len(docs))
        ctxs.update(docs)