import hydra
import hydra.utils as hu

from hydra.core.hydra_config import HydraConfig
import tqdm
import numpy as np
import json
from rank_bm25 import BM25Okapi
# from src.utils.app import App
from src.dataset_readers.bm25_tasks import BM25Task
from dataclasses import dataclass
import multiprocessing
from ElasticSearchBM25 import ElasticSearchBM25
from datasets import load_from_disk


# global_context = {}


# def __post_init__(self):
#     self.converter = QDMRToQDMRStepTokensConverter()
#     self.matcher = LogicalFromStructuralMatcher()
#     self.scorer = NormalizedGraphMatchScorer()

class BM25Finder:
    def __init__(self, cfg) -> None:
        self.output_path = cfg.output_path
        self.task_name = cfg.task_name
        # assert cfg.dataset_split in ["train", "validation", "test", 'debug', "test_asset", "test_turk", "test_wiki"]
        self.is_train = cfg.dataset_split in ["train", "debug"]
        self.L = cfg.L
        self.setup_type = cfg.setup_type
        assert self.setup_type in ["q", "qa", "a"]
        self.task = BM25Task.from_name(cfg.task_name)(cfg.dataset_split,
                                                      cfg.setup_type,
                                                      ds_size=None if "ds_size" not in cfg else cfg.ds_size)
        print("started creating the corpus")
        # self.corpus = self.task.get_corpus()
        self.corpus = {}
        task_corpus = self.task.get_corpus()
        for i in tqdm.tqdm(range(len(task_corpus))):
            self.corpus[str(i)] = " ".join(task_corpus[i])
        # self.bm25 = ElasticSearchBM25(self.corpus, index_name=self.task_name)
        self.bm25 = ElasticSearchBM25(self.corpus,
                                      # host="http://localhost",
                                      # host="127.0.0.1",
                                      # port_http="9200",
                                      # port_tcp="9300",
                                      index_name=self.task_name,
                                      reindexing=cfg.reindexing
                                      )
        # self.bm25 = BM25Okapi(self.corpus)
        print("finished creating the corpus")


def search(tokenized_query, is_train, idx, L, score):
    # bm25 = global_context['bm25']

    bm25 = bm25_global
    # scores = bm25.get_scores(tokenized_query)
    # near_ids = list(np.argsort(scores)[::-1][:L])
    # near_ids = near_ids[1:] if is_train else near_ids
    query = " ".join(tokenized_query)[:1024]
    # if len(query) == 0 or len(query) >= 1024:
    #     return [], idx
    if score:
        rank, scores = bm25.query(query, topk=L, return_scores=True)
        near_ids = [int(x) for x in rank.keys()]
        near_ids = near_ids[1:] if is_train else near_ids
        # scores = scores[1:] if is_train else scores
        # print('---')
        # print(scores)
        # print('***')
        ctxs = [{'id': int(k), 'score': v} for k,v in scores.items()]
        if is_train:
            ctxs = ctxs[1:]
        # ctxs = [{"id": int(near_id), "score": scores[near_id]} for near_id in near_ids]
        return ctxs, idx
    else:
        rank = bm25.query(query, topk=L)
        near_ids = [int(x) for x in rank.keys()]
        near_ids = near_ids[1:] if is_train else near_ids
        return [{"id": int(a)} for a in near_ids], idx


def _search(args):
    tokenized_query, is_train, idx, L, score = args
    try:
        result = search(tokenized_query, is_train, idx, L, score)
        return result
    except Exception as e:
        print(e)
        # print(args)
        return [], idx


class GlobalState:
    def __init__(self, bm25) -> None:
        self.bm25 = bm25


def find(cfg):
    knn_finder = BM25Finder(cfg)
    tokenized_queries = [knn_finder.task.get_field(entry)
                         for entry in knn_finder.task.dataset]

    # global_context['bm25'] = knn_finder.bm25

    def set_global_object(bm25):
        global bm25_global
        bm25_global = bm25

    # for debug
    if "debug" in cfg.output_path.split("/")[-1]:
        knn_finder.is_train = True

    # here
    pool = multiprocessing.Pool(processes=None, initializer=set_global_object, initargs=(knn_finder.bm25,))
    # set_global_object(knn_finder.bm25)
    # cntx_pre = [[tokenized_query, knn_finder.is_train, idx, knn_finder.L, cfg.score] for idx, tokenized_query in
    #             zip(idxs, tokenized_queries)]
    cntx_pre = [[tokenized_query, knn_finder.is_train, idx, knn_finder.L, cfg.score] for idx, tokenized_query in
                enumerate(tokenized_queries)]

    data_list = list(knn_finder.task.dataset)
    cntx_post = []
    with tqdm.tqdm(total=len(cntx_pre)) as pbar:
        # for e in cntx_pre:
        #     cntx_post.append(_search(e))
        #     pbar.update(1)
        # here
        for i, res in enumerate(pool.imap_unordered(_search, cntx_pre)):
            pbar.update()
            cntx_post.append(res)
    for ctx, idx in cntx_post:
        data_list[idx]['ctxs'] = ctx
    # if knn_finder.is_train and "sc" not in cfg.output_path.split("/")[-1]:
    if knn_finder.is_train:
        data_list = [entry for entry in data_list if len(entry['ctxs']) in [49, 50, 20]]
    # for cbr
    if "cbr" in cfg.output_path.split("/")[-1]:
        data_list=data_list[:8000]
    return data_list


# python find_bm25.py output_path=$PWD/data/test_bm25_1.json dataset_split=validation setup_type=qa task_name=break
@hydra.main(config_path="configs", config_name="bm25_finder")
def main(cfg):
    print(cfg)

    data_list = find(cfg)
    # print(data_list)
    with open(cfg.output_path, "w") as f:
        json.dump(data_list, f)


if __name__ == "__main__":
    main()
