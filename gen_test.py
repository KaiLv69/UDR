import os
from src.utils.app import App
import json
import pandas as pd
from src.utils.eval_many import eval_many, eval_many_mtop,eval_many_smcalflow
from src.utils.cache_util import BufferedJsonWriter,BufferedJsonReader
import requests
import re
import numpy as np
import httpx
import asyncio
import torch
import fitlog

def renorm(text):
    text = text.split("\n")[0]
    text = re.sub("[\d]+\#\) ",";", text)
    return text

import argparse

def dwiki_bleu(file_name):
    from tqdm import tqdm
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.tokenize import word_tokenize
    import re, string
    ref_list = []
    hyp_list = []
    with open(file_name) as f:
        data = json.load(f)
    punctuation = '[%s]+' % re.escape(string.punctuation)
    for line in tqdm(data):
        ref = line['tgt'] if 'tgt' in line else line['answers'][0]
        hyp = line['generated'].split("<|endoftext|>")[0].strip()
        # 去除标点
        ref = re.sub(punctuation, '', ref)
        hyp = re.sub(punctuation, '', hyp)
        ref_list.append([word_tokenize(ref)])
        hyp_list.append(word_tokenize(hyp))
    print('bleu score:{}'.format(corpus_bleu(ref_list, hyp_list)))


def wikiauto_bleu(file_name):
    from tqdm import tqdm
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.tokenize import word_tokenize
    import re, string
    ref_list = []
    hyp_list = []
    with open(file_name) as f:
        data = json.load(f)
    punctuation = '[%s‘]+' % re.escape(string.punctuation)

    if args.dataset == 'wikiauto' and args.split not in ['debug', 'test_wiki']:
        from datasets import load_dataset
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
        ref_dict = {}
        for e in dataset[args.split]:
            k= e['target']
            k = re.sub(punctuation, '', k)
            ref_dict[k] = e['references']

    if args.dataset in ['common_gen', 'opusparcus', 'squadv2', 'e2e', 'dart', 'totto']:
        from datasets import load_dataset
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
        ref_dict = {}
        for e in dataset[args.split]:
            k= e['target']
            k = re.sub(punctuation, '', k)
            ref_dict[k] = e['references']
    q = []

    for line in tqdm(data):
        if args.dataset == 'wikiauto':
            q.append(line['source'] if 'source' in line else line['question'])
            if args.split in ['debug', 'test_wiki']:
                ref = line['target'] if 'target' in line else line['answers'][0]
                # ref = re.sub(punctuation, '', ref)
                hyp = line['generated'].split("<|endoftext|>")[0].strip()
                # 去除标点
                # hyp = re.sub(punctuation, '', hyp)
                # ref_list.append([word_tokenize(ref)])
                # hyp_list.append(word_tokenize(hyp))
                ref_list.append(ref)
                hyp_list.append(hyp)
            else:
                k = line['target'] if 'target' in line else line['answers'][0]
                k = re.sub(punctuation, '', k)
                ref = ref_dict[k]
                # ref = [re.sub(punctuation, '', r) for r in ref]
                # ref_list.append([word_tokenize(r) for r in ref])
                ref_list.append([r for r in ref])
                hyp = line['generated'].split("<|endoftext|>")[0].strip()
                # 去除标点
                # hyp = re.sub(punctuation, '', hyp)
                # hyp_list.append(word_tokenize(hyp))
                hyp_list.append(hyp)
        elif args.dataset in ['xsum']:
            ref = line['summary'] if 'summary' in line else line['answers'][0]
            # ref = re.sub(punctuation, '', ref)
            hyp = line['generated'].split("<|endoftext|>")[0].strip()
            if len(hyp) == 0 or hyp == '.':
                hyp = "a"
            # 去除标点
            # hyp = re.sub(punctuation, '', hyp)
            ref_list.append(ref)
            hyp_list.append(hyp)
        elif args.dataset in ['cnndailymail']:
            ref = line['highlights'] if 'highlights' in line else line['answers'][0]
            # ref = re.sub(punctuation, '', ref)
            hyp = line['generated'].split("<|endoftext|>")[0].strip()
            if len(hyp) == 0 or hyp == '.':
                hyp = "a"
            if len(ref) == 0 or hyp == '.':
                ref = "a"
            # 去除标点
            # hyp = re.sub(punctuation, '', hyp)
            ref_list.append(ref)
            hyp_list.append(hyp)
        elif args.dataset in ['reddit', 'wikihow', 'pubmed', 'multinews']:
            ref = line['target'] if 'target' in line else line['answers'][0]
            # ref = re.sub(punctuation, '', ref)
            hyp = line['generated'].split("<|endoftext|>")[0].strip()
            if len(hyp) == 0 or hyp == '.':
                hyp = "a"
            if len(ref) == 0 or ref == '.':
                ref = "a"
            # 去除标点
            # hyp = re.sub(punctuation, '', hyp)
            ref_list.append(ref)
            hyp_list.append(hyp)
        # elif args.dataset in ['opusparcus']:
        #     k = line['target'] if 'target' in line else line['answers'][0]
        #     k = re.sub(punctuation, '', k)
        #     ref = ref_dict[k][0]
        #     hyp = line['generated'].split("<|endoftext|>")[0].strip()
        #     if len(hyp) == 0 or hyp == '.':
        #         hyp = "a"
        #     if len(ref) == 0 or ref == '.':
        #         ref = "a"
        #     ref_list.append(ref)
        #     hyp_list.append(hyp)
        # elif args.dataset in ['common_gen']:
        #     k = line['target'] if 'target' in line else line['answers'][0]
        #     k = re.sub(punctuation, '', k)
        #     ref = ref_dict[k]
        #     hyp = line['generated'].split("<|endoftext|>")[0].strip()
        #     ref_list.append(ref)
        #     hyp_list.append(hyp)
        elif args.dataset in ['common_gen', 'squadv2', "opusparcus"]:
            k = line['target'] if 'target' in line else line['answers'][0]
            k = re.sub(punctuation, '', k)
            ref = ref_dict[k]
            # ref = [re.sub(punctuation, '', r) for r in ref]
            ref_list.append([word_tokenize(r) for r in ref])
            hyp = line['generated'].split("<|endoftext|>")[0].strip()
            if len(hyp) == 0 or hyp == '.':
                hyp = "a"
            # 去除标点
            # hyp = re.sub(punctuation, '', hyp)
            hyp_list.append(word_tokenize(hyp))
        elif args.dataset == 'iwslt':
            ref = line['translation.en'] if 'translation.en' in line else line['answers'][0]
            hyp = line['generated'].split("<|endoftext|>")[0].strip()
            # 去除标点
            ref = re.sub(punctuation, '', ref)
            hyp = re.sub(punctuation, '', hyp)
            ref_list.append([word_tokenize(ref)])
            hyp_list.append(word_tokenize(hyp))
        elif args.dataset in ['iwslt_en_fr', 'iwslt_en_de', "wmt_en_de", 'python', 'ruby', 'java', 'javascript', 'go', "php", 'wmt_de_en', 'roc_ending_generation', 'roc_story_generation']:
            ref = line['target'] if 'target' in line else line['answers'][0]
            hyp = line['generated'].split("<|endoftext|>")[0].strip()
            # 去除标点
            # ref = re.sub(punctuation, '', ref)
            # hyp = re.sub(punctuation, '', hyp)
            ref_list.append([word_tokenize(ref)])
            hyp_list.append(word_tokenize(hyp))
        elif args.dataset in ['e2e', 'dart', 'totto']:
            k = line['target'] if 'target' in line else line['answers'][0]
            k = re.sub(punctuation, '', k)
            ref = ref_dict[k]
            ref = ref.split('\n')
            # ref = [re.sub(punctuation, '', r) for r in ref]
            ref_list.append([word_tokenize(r) for r in ref])
            hyp = line['generated'].split("<|endoftext|>")[0].strip()
            # 去除标点
            # hyp = re.sub(punctuation, '', hyp)
            hyp_list.append(word_tokenize(hyp))
    if args.dataset == "wikiauto":
        from src.utils import sari
        def sari_score(src, pred, ref):
            scores = []
            for a, b, c in zip(src, pred, ref):
                scores += [sari.SARIsent(a, b, c)]
                # for EASSE metrics
                # c = [[k] for k in c]
                # scores += [corpus_sari([a], [b], c)]
            return scores
        sari = sari_score(q, hyp_list, ref_list)
        sari_avg = round(sum(sari)/ len(sari), 4)
        print("SARI: ", sari_avg)
        fitlog.add_best_metric({args.split: {'sari': sari_avg}})
        return
    if args.dataset in ['common_gen']:
        from rouge import Rouge
        bleu = round(corpus_bleu(ref_list, hyp_list, [1/3, 1/3, 1/3, 0]), 4)
        fitlog.add_best_metric({args.split: {'bleu3': bleu}})
        print('bleu3 score:{}'.format(bleu))
        bleu = round(corpus_bleu(ref_list, hyp_list), 4)
        fitlog.add_best_metric({args.split: {'bleu': bleu}})
        print('bleu score:{}'.format(bleu))
        return
    if args.dataset not in ['xsum', 'cnndailymail', 'reddit', 'wikihow', 'pubmed', 'multinews']:
        if args.dataset in ['python', 'java', 'ruby', 'javascript', "php", "go", 'roc_ending_generation', 'roc_story_generation', 'opusparcus']:
            bleu = round(corpus_bleu(ref_list, hyp_list, [1, 0, 0, 0]), 4)
            fitlog.add_best_metric({args.split: {'bleu1': bleu}})
            print('bleu1 score:{}'.format(bleu))
            bleu = round(corpus_bleu(ref_list, hyp_list, [0.5, 0.5, 0, 0]), 4)
            fitlog.add_best_metric({args.split: {'bleu2': bleu}})
            print('bleu2 score:{}'.format(bleu))
        else:
            bleu = round(corpus_bleu(ref_list, hyp_list), 4)
            fitlog.add_best_metric({args.split: {'bleu': bleu}})
            print('bleu score:{}'.format(bleu))
    else:
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(hyp_list, ref_list, avg=True)
        fitlog.add_best_metric({args.split: {'rouge-1': scores['rouge-1']['f'], 'rouge-2': scores['rouge-2']['f'],
                                             'rouge-l': scores['rouge-l']['f']}})
        print('rouge-1 score:{}'.format(scores['rouge-1']['f']))
        print('rouge-2 score:{}'.format(scores['rouge-2']['f']))
        print('rouge-l score:{}'.format(scores['rouge-l']['f']))



def kp20k_f1(file_name, split):
    from evaluate_prediction import main, ARGS
    import re
    from tqdm import tqdm
    with open(file_name, 'r') as f:
        data = json.load(f)
    src_list = []
    trg_list = []
    pred_list = []
    opt = ARGS()
    for i in tqdm(data):
        # src_list.append(" ".join(i['document']))
        try:
            d = i['document'] if "document" in i else i['question']
            src_list.append(d)
            if split == 'abstract':
                if "abstractive_keyphrases" in i:
                    i['abstractive_keyphrases'] = eval(i['abstractive_keyphrases'])
                    trg_list.append(";".join(i['abstractive_keyphrases']))
                    # trg_list.append(";".join(i['extractive_keyphrases']+i['abstractive_keyphrases']))
                else:
                    i['abstractive_keyphrases'] = eval(i['answers'][0])
                    trg_list.append(";".join(i['abstractive_keyphrases']))
            else:
                if "extractive_keyphrases" in i:
                    i['extractive_keyphrases'] = eval(i['extractive_keyphrases'])
                    trg_list.append(";".join(i['extractive_keyphrases']))
                    # trg_list.append(";".join(i['extractive_keyphrases']+i['abstractive_keyphrases']))
                else:
                    i['extractive_keyphrases'] = eval(i['answers'][0])
                    trg_list.append(";".join(i['extractive_keyphrases']))


            pred = '"' + i['generated']
            reg = re.compile(r'"(.*?)"')
            pred = re.findall(reg, pred)
            if len(pred) == 0:
                print(i['generated'])
            pred_list.append(";".join(pred))
        except Exception as e:
            print(e)

    result_dict = main(opt,src_list,trg_list,pred_list)
    for k,v in result_dict.items():
        if '@M' in k and 'macro' in k:
            print('{} : {}'.format(k,v))

def add_mtop_acc(file_name,id_list=None):
    correct_list = []
    with open(file_name) as f:
        line_list = []
        data = json.load(f)

        for line in data:
            if id_list is not None and line['id'] not in id_list:
                continue
            lf = line['logical_form'] if 'logical_form' in line else line['answers'][0]
            line_list.append((line['generated'].split("<|endoftext|>")[0].strip(),lf))
    pred,gold = list(zip(*line_list))
    res_list = eval_many_mtop(pred,gold)
    # print('result')
    # print(res_list)
    for entry,acc in zip(data,res_list):
        entry['acc'] =acc

    # print(res_list)
    res_list_int = list(map(int,res_list))
    # print()
    acc_result = sum(res_list_int)/len(res_list_int)
    print('file_name:{}'.format(file_name))
    print('prediction acc:{}'.format(acc_result))
    fitlog.add_best_metric({args.split: {'acc': acc_result}})
    return data


def add_spider_acc(file_name,id_list=None):
    from datasets import load_from_disk
    import spider.evaluation
    from tqdm import tqdm
    with open(file_name, 'r') as f:
        input_data = json.load(f)
    ds = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/spider")

    db_dict = {}
    for e in ds['validation']:
        db_dict[e['query']] = e['db_id']

    pred_list = []
    gold_list = []
    num = 0
    for e in tqdm(input_data):
        p = e['generated'].split("<|endoftext|>")[0].strip()
        g = e['query'] if 'query' in e else e['answers'][0]
        db_id = db_dict[g]
        # g = g.replace(" ,", ",")
        if p == g:
            num += 1
        g = [g, db_id]
        if len(p) == 0:
            p = "a"
        p = [p, db_id]
        pred_list.append(p)
        gold_list.append(g)
    ret = spider.evaluation.evaluate_in_memory(pred_list, gold_list)
    fitlog.add_best_metric({args.split: {'acc': ret['total_scores']['all']['exact']}})

def add_break_acc(path, id_list=None):
    with BufferedJsonReader(path) as f:
        df = pd.DataFrame(f.read())
    data = df.to_dict("records")
    question_field = "question" if "question" in data[0] else 'question_text'
    zipped_data = []
    for entry in data:
        if id_list is not None and entry['id'] in id_list:
            continue
        generated = renorm(entry['generated'].split("\n")[0].split("<|endoftext|>")[0]).strip()
        decomposition = entry['decomposition'] if "decomposition" in entry else entry['answers'][0]

        zipped_data.append([entry[question_field], generated, decomposition])

    questions, pred, gold = list(zip(*zipped_data))
    acc_results = eval_many(questions, pred, gold)
    acc_results_int = list(map(int,acc_results))
    acc_result = sum(acc_results_int)/len(acc_results_int)
    print('result:')
    print(acc_result)
    fitlog.add_best_metric({args.split: {'acc': acc_result}})

    for entry, acc in zip(data, acc_results):
        entry['acc'] = acc
    return data


def add_smcalflow_acc(file_name,id_list=None):
    correct_list = []
    with open(file_name) as f:
        data = json.load(f)
        line_list = []
        for line in data:
            if id_list is not None and line['id'] not in id_list:
                continue
            lf = line['lispress'] if 'lispress' in line else line['answers'][0]
            line_list.append((line['generated'].split("<|endoftext|>")[0].strip(),lf))
    pred,gold = list(zip(*line_list))
    res_list = eval_many_smcalflow(pred,gold)

    acc_results_int = list(map(int,res_list))
    acc_result = sum(acc_results_int)/len(acc_results_int)
    print('result:')
    print(acc_result)
    fitlog.add_best_metric({args.split: {'acc': acc_result}})

    for entry,acc in zip(data,res_list):
        entry['acc'] =acc
    return data

from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import evaluate
from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import format_qdmr
def eval_break_test(file_name,id_list=None):
    with BufferedJsonReader(file_name) as f:
        df = pd.DataFrame(f.read())
    data = df.to_dict("records")
    question_field = "question" if "question" in data[0] else 'question_text'
    zipped_data = []
    for entry in data:
        if id_list is not None and entry['id'] in id_list:
            continue
        generated = renorm(entry['generated'].split("\n")[0].split("<|endoftext|>")[0]).strip()
        decomposition = entry['decomposition'] if "decomposition" in entry else entry['answers'][0]

        zipped_data.append([entry[question_field], generated, decomposition, entry['question_id']])

    questions, predictions, golds, question_ids = list(zip(*zipped_data))

    predictions = [format_qdmr(pred.replace("  "," ")) for pred in predictions]
    golds = [format_qdmr(gold) for gold in golds]

    res = evaluate(question_ids=question_ids,
                   questions=questions,
                   golds=golds,
                   decompositions=predictions,
                   metadata=None,
                   output_path_base=None,
                   num_processes=8)

    print(res)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',type=str)
    parser.add_argument('--split',type=str,default="test")
    parser.add_argument('--fp',)
    parser.add_argument('--exp_name', type=str)
    # parser.add_argument('--lr', type=float)
    # parser.add_argument('--loss_type', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--plm', type=str)
    parser.add_argument('--iter_scored_num', type=str)
    parser.add_argument('--iter_num', type=str)
    parser.add_argument('--epoch_num', type=str, default="10")
    parser.add_argument('--prompt_num', type=str)
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--beilv', type=str)


    args = parser.parse_args()

    fitlog.set_log_dir("udr_fitlog/metric_logs/")  # 设定日志存储的目录
    fitlog.add_hyper(args)  # 通过这种方式记录ArgumentParser的参数

    # tmp_fp = args.fp
    if args.fp==None:
        tmp_fp = 'data/bm25_{}_result_{}.json'.format(args.dataset,args.split)
    else:
        tmp_fp = args.fp

    # tmp_fp = 'data/bm25_mtop_result_test.json'
    if args.dataset == 'mtop':
        add_mtop_acc(tmp_fp)
    elif args.dataset == 'break':
        if args.split == 'validation':
            add_break_acc(tmp_fp)
        elif args.split == 'test':
            eval_break_test(tmp_fp)
    elif args.dataset == "dwiki":
        dwiki_bleu(tmp_fp)
    elif args.dataset == 'kp20k':
        kp20k_f1(tmp_fp, args.split)
    elif args.dataset in ["wikiauto", 'xsum', 'cnndailymail', 'reddit', 'wikihow', 'pubmed', 'multinews']:
        wikiauto_bleu(tmp_fp)
    elif args.dataset in ["iwslt", "iwslt_en_fr", "iwslt_en_de", "wmt_en_de", 'python', 'ruby', 'java', 'javascript', "php", "go", 'wmt_de_en']:
        wikiauto_bleu(tmp_fp)
    elif args.dataset == "common_gen":
        wikiauto_bleu(tmp_fp)
    elif args.dataset in ["opusparcus", 'roc_ending_generation', 'roc_story_generation']:
        wikiauto_bleu(tmp_fp)
    elif args.dataset == "squadv2":
        wikiauto_bleu(tmp_fp)
    elif args.dataset in ["e2e", 'dart', 'totto']:
        wikiauto_bleu(tmp_fp)
    elif args.dataset == 'smcalflow':
        add_smcalflow_acc(tmp_fp)
    elif args.dataset == 'spider':
        add_spider_acc(tmp_fp)
    else:
        raise NotImplementedError

    fitlog.finish()  # finish the logging

