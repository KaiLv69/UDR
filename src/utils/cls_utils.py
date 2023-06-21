import os
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
    from datasets import load_dataset
    ds = load_dataset(dataset_path[task])
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
