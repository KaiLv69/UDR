import argparse
import copy
import json
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import pandas as pd


def convert():
    print("input_file: ", args.input_file)
    print("output_file: ", args.output_file)

    df = pd.read_json(args.input_file)
    df['answers'] = df['answers'].map(lambda x: x[0])
    df['ctxs'] = df['ctxs'].map(lambda x: x[:20])
    # with open(args.input_file, "r") as data_file:
    #     input_data = json.load(data_file)
    print("len of input file: ", len(df))
    if args.dataset == "kp20k":
        df = df.rename(columns={'question': 'document', 'answers': 'extractive_keyphrases'})
    elif args.dataset == "wikiauto":
        df = df.rename(columns={'question': 'source', 'answers': 'target'})
    elif args.dataset == "iwslt":
        df = df.rename(columns={'question': 'translation.de', 'answers': 'translation.en'})
    elif args.dataset in ["iwslt_en_fr", "iwslt_en_de"]:
        df = df.rename(columns={'answers': 'target'})
    elif args.dataset == 'mtop':
        df = df.rename(columns={'answers': 'logical_form'})
    elif args.dataset in ['opusparcus', 'python', 'ruby']:
        df = df.rename(columns={'question': 'input', 'answers': 'target'})
    elif args.dataset == 'xsum':
        df = df.rename(columns={'question': 'document', 'answers': 'summary'})
    elif args.dataset == 'cnndailymail':
        df = df.rename(columns={'question': 'article', 'answers': 'highlights'})
    elif args.dataset == 'spider':
        df = df.rename(columns={'answers': 'query'})
    # print(df.to_json(orient='records', lines='orient'))
    df.to_json(args.output_file, orient='records', force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mtop")
    parser.add_argument("--input_file", type=str,
                        default="/remote-home/klv/exps/rtv_icl/v0/EPR/exps/exp_2022_9_19_mtop_batch_128_epoch_30/data/epr_result_data_validation")
    parser.add_argument("--output_file", type=str,
                        default="/remote-home/klv/exps/rtv_icl/v0/EPR/data/mtop_scoredqa_conv.json")
    args = parser.parse_args()
    convert()
