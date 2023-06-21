"""
把dense_retriever.py输出的文件格式转换为scorer.py需要的格式
input_file中与input_scored_file中的idx对应的数据会被放到out_dep_file中
不重复的放到out_file中送给scorer.py打分

打分结束后，把out_file和out_dep_file合并并排序，得到最终的结果
"""
import argparse
import copy
import json
import os.path

from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk


def convert_kp20k():
    print("input_file: ", args.input_file)
    print("input_scored_file: ", args.input_scored_file)
    print("output_file: ", args.output_file)
    print("output_dep_file: ", args.output_dep_file)

    with open(args.input_scored_file, "r") as data_file:
        input_scored_data = json.load(data_file)

    with open(args.input_file, "r") as data_file:
        input_data = json.load(data_file)
    print("len of input file: ", len(input_data))
    print("len of input scored file", len(input_scored_data))
    dataset_path = {
        "agnews": "KaiLv/UDR_AGNews",
        "amazon": "KaiLv/UDR_Amazon",
        "break": "KaiLv/UDR_BREAK",
        "cnndailymail": "KaiLv/UDR_CNNDailyMail",
        "cola": "KaiLv/UDR_COLA",
        "common_gen": "KaiLv/UDR_CommonGen",
        "copa": "KaiLv/UDR_COPA",
        "cosmos_qa": "KaiLv/UDR_CosmosQA",
        "cr": "KaiLv/UDR_CR",
        "cs_explan": "KaiLv/UDR_ComE",
        "cs_valid": "KaiLv/UDR_ComV",
        "dart": "KaiLv/UDR_DART",
        "dbpedia": "KaiLv/UDR_DBPedia",
        "e2e": "KaiLv/UDR_E2E",
        'go': "KaiLv/UDR_Go",
        'java': "KaiLv/UDR_Java",
        'mnli': "KaiLv/UDR_MNLI",
        'mr': "KaiLv/UDR_MR",
        'mtop': 'KaiLv/UDR_MTOP',
        'php': "KaiLv/UDR_PHP",
        'pubmed': "KaiLv/UDR_PubMed",
        'python': "KaiLv/UDR_Python",
        'reddit': "KaiLv/UDR_Reddit",
        'roc_ending_generation': "KaiLv/UDR_RocEnding",
        'roc_story_generation': "KaiLv/UDR_RocStory",
        'rte': "KaiLv/UDR_RTE",
        'smcalflow': "KaiLv/UDR_SMCalFlow",
        'snli': "KaiLv/UDR_SNLI",
        'sst2': "KaiLv/UDR_SST-2",
        'sst5': "KaiLv/UDR_SST-5",
        'subj': "KaiLv/UDR_Subj",
        'trec': "KaiLv/UDR_TREC",
        'wikiauto': "KaiLv/UDR_WikiAuto",
        'yahoo': "KaiLv/UDR_Yahoo",
        'yelp': "KaiLv/UDR_Yelp",
    }
    dataset = load_dataset(dataset_path[args.dataset])

    train_set = dataset[args.split]

    out_data = []
    out_dep_data = []
    print("len of train set: ", len(train_set))
    idx_attr = "idx"
    idx2scored = {e[idx_attr]: e['ctxs'] for e in input_scored_data}
    # for i in tqdm(range(len(input_scored_data))):
    for i in tqdm(range(len(input_data))):
        idx = train_set[i][idx_attr]
        out_data.append(copy.deepcopy(train_set[i]))
        out_dep_data.append(copy.deepcopy(train_set[i]))
        ctx = [{'id': e['id']} for e in input_data[i]['ctxs'] if e['id'] != idx]

        # scored里去重
        if idx in idx2scored:
            ctx_in_scored = idx2scored[idx]
        else:
            ctx_in_scored = []
        # list不能hash
        if args.dataset in ["wikiauto"]:
            refs = {}
            for entry in ctx_in_scored:
                refs[entry['id']] = entry['references']
                entry['references'] = None
        elif args.dataset == "squadv2":
            refs = {}
            for entry in ctx_in_scored:
                refs[entry['id']] = entry['references']
                entry['references'] = None
                entry['answers'] = None
        elif args.dataset == "common_gen":
            refs = {}
            for entry in ctx_in_scored:
                refs[entry['id']] = entry['references']
                entry['references'] = None
                entry['concepts'] = None

        ctx_in_scored = [dict(t) for t in set([tuple(d.items()) for d in ctx_in_scored])]
        ctx_in_scored.sort(key=lambda x: x['score'])
        # 还原
        if args.dataset == "kp20k":
            for entry in ctx_in_scored:
                entry['abstractive_keyphrases'] = eval(entry['abstractive_keyphrases'])
        elif args.dataset in ["wikiauto", 'squadv2', 'common_gen']:
            for entry in ctx_in_scored:
                entry['references'] = refs[entry['id']]
        if args.split == "train":
            id_in_scored = [x['id'] for x in ctx_in_scored if x['id'] != i]
            id_in_rtv = [x['id'] for x in input_data[i]['ctxs'] if x['id'] != i]
        else:
            id_in_scored = [x['id'] for x in ctx_in_scored if x['id'] != idx]
            id_in_rtv = [x['id'] for x in input_data[i]['ctxs'] if x['id'] != idx]
        out_data[i]['ctxs'] = [e for e in ctx if e['id'] not in id_in_scored]
        if args.num == 'all':
            out_dep_data[i]['ctxs'] = [e for e in ctx_in_scored]
        elif args.num == "static":
            out_dep_data[i]['ctxs'] = [e for e in ctx_in_scored if e['id'] in id_in_rtv or e['score'] <= ctx_in_scored[4]['score']]
        else:
            raise NotImplementedError

    print("len of out file: ", len(out_data))
    print("len of out dep file: ", len(out_dep_data))
    with open(args.output_file, "w") as data_file:
        json.dump(out_data, data_file)
    with open(args.output_dep_file, "w") as data_file:
        json.dump(out_dep_data, data_file)

def convert():
    print("input_file: ", args.input_file)
    print("input_scored_file: ", args.input_scored_file)
    print("output_file: ", args.output_file)
    print("output_dep_file: ", args.output_dep_file)

    with open(args.input_scored_file, "r") as data_file:
        input_scored_data = json.load(data_file)

    with open(args.input_file, "r") as data_file:
        input_data = json.load(data_file)
    print("len of input file: ", len(input_data))
    print("len of input scored file", len(input_scored_data))

    if args.dataset == "mtop":
        dataset = load_dataset("iohadrubin/mtop")
        idx = "idx"
        question = "question"
    elif args.dataset == "break":
        dataset = load_dataset("break_data", "QDMR")
        idx = "question_id"
        question = "question_text"
    elif args.dataset == "smcalflow":
        dataset = load_dataset("KaiLv/UDR_SMCalFlow")
        # dataset = load_dataset("iohadrubin/smcalflow")
        idx = "idx"
        question = "user_utterance"

    train_set = dataset[args.split]

    out_data = []
    out_dep_data = []
    offset = 0
    for i in tqdm(range(len(train_set))):
        if train_set[i][question].replace("'", "").replace("’", "") != input_data[i]['question'].replace("'", "").replace("’", ""):
            print("different question id: ", i)
            print(train_set[i][question])
            print(input_data[i]['question'])
        out_data.append(copy.deepcopy(train_set[i]))
        out_dep_data.append(copy.deepcopy(train_set[i]))

        ctx = [{'id': e['id']} for e in input_data[i]['ctxs'] if e['id'] != i]

        if train_set[i][idx] != input_scored_data[i-offset][idx]:
            out_data[i]['ctxs'] = ctx
            out_dep_data[i]['ctxs'] = []
            offset += 1
        else:
            # 去重
            ctx_in_scored = input_scored_data[i - offset]['ctxs']
            ctx_in_scored = [dict(t) for t in set([tuple(d.items()) for d in ctx_in_scored])]
            ctx_in_scored.sort(key=lambda x: x['score'])

            id_in_scored = [x['id'] for x in ctx_in_scored if x['id'] != i]
            id_in_rtv = [x['id'] for x in input_data[i]['ctxs'] if x['id'] != i]
            out_data[i]['ctxs'] = [e for e in ctx if e['id'] not in id_in_scored]
            if args.num == 'all':
                out_dep_data[i]['ctxs'] = [e for e in ctx_in_scored]
            elif args.num == "static":
                out_dep_data[i]['ctxs'] = [e for e in ctx_in_scored if
                                           e['id'] in id_in_rtv or e['score'] <= ctx_in_scored[4]['score']]
            else:
                raise NotImplementedError
            # out_dep_data[i]['ctxs'] = [e for e in ctx_in_scored if e['id'] in id_in_rtv or e['score'] <= ctx_in_scored[4]['score']]

    print("len of out file: ", len(out_data))
    print("len of out dep file: ", len(out_dep_data))
    with open(args.output_file, "w") as data_file:
        json.dump(out_data, data_file)
    with open(args.output_dep_file, "w") as data_file:
        json.dump(out_dep_data, data_file)


def merge():
    print("scored_file1: ", args.scored_file1)
    print("scored_file2: ", args.scored_file2)
    print("output_file: ", args.output_file)

    with open(args.scored_file1, "r") as data_file:
        scored_data1 = json.load(data_file)
    with open(args.scored_file2, "r") as data_file:
        scored_data2 = json.load(data_file)

    print(len(scored_data1), len(scored_data2))
    idx = 'idx'
    if args.dataset == "mtop":
        idx = "idx"
    elif args.dataset == "break":
        idx = "question_id"
    elif args.dataset == "kp20k":
        idx = "idx"
    elif args.dataset == "dwiki":
        idx = "idx"
    elif args.dataset == "wikiauto":
        idx = "idx"
    elif args.dataset == "iwslt":
        idx = "idx"
    idx2sc2 = {e[idx]: e['ctxs'] for e in scored_data2}
    for i in tqdm(scored_data1):  # len(scored_data2) < len(scored_data1)
        idx_i = i[idx]
        if idx_i in idx2sc2:
            i['ctxs'] += idx2sc2[idx_i]

        if args.dataset == "wikiauto":
            refs = {}
            for entry in i['ctxs']:
                refs[entry['id']] = entry['references']
                entry['references'] = None
        elif args.dataset == "common_gen":
            refs = {}
            for entry in i['ctxs']:
                refs[entry['id']] = entry['references']
                entry['references'] = None
                entry['concepts'] = None
        i['ctxs'] = [dict(t) for t in set([tuple(d.items()) for d in i['ctxs']])]
        if args.dataset in ["wikiauto", 'squadv2', 'common_gen']:
            for entry in i['ctxs']:
                entry['references'] = refs[entry['id']]

        i['ctxs'].sort(key=lambda x: x['score'])

    out_data = [e for e in scored_data1 if e['ctxs']]
    print("len of merged data: ", len(out_data))

    with open(args.output_file, "w") as data_file:
        json.dump(out_data, data_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, default="convert")
    parser.add_argument("--dataset", type=str, default="mtop")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num", type=str, default="static")
    parser.add_argument("--input_file", type=str, default="/remote-home/klv/exps/rtv_icl/v0/EPR/exps/exp_2022_9_19_mtop_batch_128_epoch_30/data/epr_result_data_validation")
    parser.add_argument("--input_scored_file", type=str, default="/remote-home/klv/exps/rtv_icl/v0/EPR/exps/exp_2022_9_19_mtop_batch_128_epoch_30/data/epr_result_data_validation")
    parser.add_argument("--output_file", type=str, default="/remote-home/klv/exps/rtv_icl/v0/EPR/data/mtop_scoredqa_conv.json")
    parser.add_argument("--output_dep_file", type=str, default="/remote-home/klv/exps/rtv_icl/v0/EPR/data/mtop_scoredqa_conv.json")
    parser.add_argument("--scored_file1", type=str, default="/remote-home/klv/exps/rtv_icl/v0/EPR/data/mtop_scoredqa_conv.json")
    parser.add_argument("--scored_file2", type=str, default="/remote-home/klv/exps/rtv_icl/v0/EPR/data/mtop_scoredqa_conv.json")
    args = parser.parse_args()

    if args.func == "convert":
        if args.dataset not in ["mtop", "break"]:
            convert_kp20k()
        else:
            convert()
    elif args.func == "merge":
        merge()
