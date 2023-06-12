import argparse
import fitlog
import json


def get_scores():
    with open(args.fp, 'r') as f:
        data = json.load(f)

    sc_res = []
    for topk in [1, 2, 3, 4, 5, 10, 15, 20]:
        topk_scs = []
        for e in data:
            if len(e['ctxs']) < topk:
                continue
            for i in range(topk):
                topk_scs.append(e['ctxs'][i]['score'])
        avg_sc = round(sum(topk_scs) / len(topk_scs), 4)
        sc_res.append(avg_sc)
        print('top{} score'.format(topk), avg_sc)
        fitlog.add_best_metric({args.split: {'top{} score'.format(topk): avg_sc}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',type=str)
    parser.add_argument('--split',type=str,default="test")
    parser.add_argument('--fp',)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--iter_num', type=str)
    parser.add_argument('--epoch_num', type=str)
    # parser.add_argument('--prompt_num', type=str)
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--beilv', type=str)

    args = parser.parse_args()
    fitlog.set_log_dir("upr_fitlog/score_logs/")  # 设定日志存储的目录
    fitlog.add_hyper(args)  # 通过这种方式记录ArgumentParser的参数
    get_scores()
    fitlog.finish()  # finish the logging
