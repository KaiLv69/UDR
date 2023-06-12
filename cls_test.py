import fitlog, argparse, json
from tqdm import tqdm
from Channel_LM_Prompting.util import get_label_from_template


def get_labels_cls(data):
    post_data = {}
    for d in tqdm(data):
        idx = d['question'] if 'question' in d else d['sentence']
        if idx in post_data:
            if d['loss'] < post_data[idx]['loss']:
                post_data[idx]['loss'] = d['loss']
                assert d['true_label'] == post_data[idx]['true_label']
                if 'sentence' in d:
                    assert d['sentence'] == post_data[idx]['sentence']
                else:
                    assert d['question'] == post_data[idx]['sentence']
                post_data[idx]['test_label'] = get_label_from_template(args.dataset, d['test_label'])
        else:
            post_data[idx] = {'loss': d['loss'],
                              'true_label': d['true_label'],
                              'test_label': get_label_from_template(args.dataset, d['test_label']),
                              'sentence': d['sentence'] if 'sentence' in d else d['question']}

    true_labels, test_labels = [], []
    for k, v in post_data.items():
        true_labels.append(v['true_label'])
        test_labels.append(v['test_label'])
    return true_labels, test_labels


def get_labels_multi_choice(data):
    post_data = {}
    for d in tqdm(data):
        idx = d['question']
        if idx in post_data:
            if d['loss'] < post_data[idx]['loss']:
                post_data[idx]['loss'] = d['loss']
                assert d['true_label'] == post_data[idx]['true_label']
                assert d['question'] == post_data[idx]['question']
                post_data[idx]['test_label'] = d['test_label']
        else:
            post_data[idx] = {'loss': d['loss'],
                              'true_label': d['true_label'],
                              'test_label': d['test_label'],
                              'question': d['question']}

    true_labels, test_labels = [], []
    for k, v in post_data.items():
        true_labels.append(v['true_label'])
        test_labels.append(v['test_label'])
    return true_labels, test_labels


def cal_acc(true_labels, test_labels):
    assert len(true_labels) == len(test_labels)
    correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] == test_labels[i]:
            correct += 1
    return round(correct / len(true_labels), 4)


def test():
    with open(args.fp) as f:
        data = json.load(f)
    if args.dataset in ['commonsense_qa', 'cs_explan', 'cosmos_qa', 'social_i_qa', 'piqa', 'race', 'cs_valid',
                        'hellaswag', 'openbookqa', 'arc_easy', 'copa', 'balanced_copa']:
        true_labels, test_labels = get_labels_multi_choice(data)
    else:
        true_labels, test_labels = get_labels_cls(data)
    acc_result = cal_acc(true_labels, test_labels)
    print(acc_result)
    fitlog.add_best_metric({args.split: {'acc': acc_result}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--fp', )
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--plm', type=str)
    parser.add_argument('--iter_scored_num', type=str)
    parser.add_argument('--iter_num', type=str)
    parser.add_argument('--epoch_num', type=str)
    parser.add_argument('--prompt_num', type=str)
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--beilv', type=str)

    args = parser.parse_args()

    fitlog.set_log_dir("upr_fitlog/metric_logs/")  # 设定日志存储的目录
    fitlog.add_hyper(args)  # 通过这种方式记录ArgumentParser的参数

    test()

    fitlog.finish()  # finish the logging
