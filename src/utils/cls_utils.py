from Channel_LM_Prompting.util import get_prompts, get_label_from_template, get_one_prompt
from copy import deepcopy
from tqdm import tqdm


def get_test_labels(data, task, idx):
    ret = []
    test_labels = get_prompts(task, idx)
    for e in data:
        for l in test_labels:
            tmp = deepcopy(e)
            tmp['test_label'] = l
            ret.append(tmp)
    return ret

def change_prompt_template(data, task, idx):
    for e in tqdm(data):
        for ctx in e['ctxs']:
            spt = ctx['text'].split('\t')
            q = "\t".join(spt[:-1])
            a = spt[-1]
            label = get_label_from_template(task, a)
            new_a = get_one_prompt(task, idx, label)
            ctx['text'] = q + '\t' + new_a
    return data


def get_multi_choice_labels(data, task, split):
    ret = []
    from datasets import load_from_disk
    ds = load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/" + task)
    q_to_choices = {}
    for e in ds[split]:
        q_to_choices[e['question'].replace("’", "").replace("'", "")] = e['choices']
    for e in tqdm(data):
        if 'choices' not in e:
            k = e['question'].replace("’", "").replace("'", "")  # ’
            e['choices'] = q_to_choices[k]
        e['choices'] = e['choices'].split('\n')
        for choice in e['choices']:
            tmp = deepcopy(e)
            tmp['test_label'] = choice
            ret.append(tmp)
    return ret
