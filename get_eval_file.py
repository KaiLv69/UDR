import json
from argparse import ArgumentParser
from datasets import load_from_disk
from tqdm import tqdm
import spider.evaluation

def get_pred_gold_file():
    with open(args.input_file, 'r') as f:
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
    print(num/len(input_data))
    ret = spider.evaluation.evaluate_in_memory(pred_list, gold_list)
    print(ret['total_scores']['all']['exact'])




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--pred_file", type=str)
    parser.add_argument("--gold_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    args.input_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/bm25_spider_1124/data/inference_bm25_spider_validation_q.json"
    args.pred_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/bm25_spider_1124/data/pred_bm25_spider_validation_q.txt"
    args.gold_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/bm25_spider_1124/data/gold_bm25_spider_validation_q.txt"
    args.output_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/bm25_spider_1124/data/eval_bm25_spider_validation_q.txt"

    args.input_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_spider_1125_bert/data/epr_inference_spider_validation"
    args.pred_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_spider_1125_bert/data/pred_bm25_spider_validation_q.txt"
    args.gold_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_spider_1125_bert/data/gold_bm25_spider_validation_q.txt"
    args.output_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_spider_1125_bert/data/eval_bm25_spider_validation_q.txt"

    args.input_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/rk_iter_spider_1125_bert/data/epr_result_prediction_spider_validation_29"
    args.pred_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_spider_1125_bert/data/pred_epr_spider_validation_q_29.txt"
    args.gold_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_spider_1125_bert/data/gold_epr_spider_validation_q_29.txt"
    args.output_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_spider_1125_bert/data/eval_epr_spider_validation_q_29.txt"

    args.input_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/rk_iter_spider_1125_bert/data/epr_result_prediction_spider_validation_10_9"
    args.pred_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_spider_1125_bert/data/pred_epr_spider_validation_q_10_9.txt"
    args.gold_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_spider_1125_bert/data/gold_epr_spider_validation_q_10_9.txt"
    args.output_file = "/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_spider_1125_bert/data/eval_epr_spider_validation_q_10_9.txt"

    get_pred_gold_file()

    db_dir = "/nvme/xnli/lk_code/exps/rtv_icl/v5/spider/spider_ds/database"
    table = "/nvme/xnli/lk_code/exps/rtv_icl/v5/spider/spider_ds/tables.json"
    # spider.evaluation.main(args.gold_file, args.pred_file, db_dir, table, "all", args.output_file)