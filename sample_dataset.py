import os
import random
from tqdm import tqdm
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from nltk import word_tokenize


def sample_common_gen():
    dataset = datasets.load_dataset('GEM/common_gen')
    id_set = set([])
    select_list = []
    for i, e in enumerate(dataset['train']):
        if e['concept_set_id'] not in id_set:
            id_set.add(e['concept_set_id'])
            select_list.append(i)

    def process_fn(example):
        example['joined_concepts'] = ", ".join(example['concepts'])
        return example

    dataset = dataset.map(process_fn, num_proc=8)
    dataset['train_dedup'] = dataset['train'].select(select_list)

    # add column idx
    for split in ['train', 'validation', 'test', 'train_dedup']:
        ds_id = Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = concatenate_datasets([ds_id, dataset[split]], axis=1)
    # overfit for debug
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)

    if os.path.exists('/nvme/xnli/lk_code/exps/rtv_icl/data'):
        base_dir = '/nvme/xnli/lk_code/exps/rtv_icl/data/common_gen'
    else:
        base_dir = '/remote-home/klv/exps/rtv_icl/data/common_gen'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(dataset)
    dataset.save_to_disk(base_dir)

def sample_roc_ending_generation():
    ds = load_dataset("adamlin/roc_story")
    print(ds)

    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    def process_roc(example):
        example['question'] = example['sentence1'] + ' ' + example['sentence2'] + ' ' + example['sentence3'] + ' ' + example['sentence4']
        example['target'] = example['sentence5']

        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example
    ds = ds.map(process_roc, num_proc=16)
    ds = ds.filter(lambda x: x['len_question'] + x['len_target'] < 900, num_proc=16)

    ds = ds.remove_columns(['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5', 'storytitle+endoftext', 'story'])


    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add idx column
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists('/nvme/xnli/lk_code/exps/rtv_icl/data'):
        base_dir = '/nvme/xnli/lk_code/exps/rtv_icl/data/roc_ending_generation'
    else:
        base_dir = '/remote-home/klv/exps/rtv_icl/data/roc_ending_generation'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def sample_roc_story_generation():
    ds = load_dataset("adamlin/roc_story")
    print(ds)

    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    def process_roc(example):
        example['question'] = example['sentence1']
        example['target'] = example['sentence2'] + ' ' + example['sentence3'] + ' ' + example['sentence4'] + ' ' + example['sentence5']

        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example
    ds = ds.map(process_roc, num_proc=16)
    ds = ds.filter(lambda x: x['len_question'] + x['len_target'] < 900, num_proc=16)

    ds = ds.remove_columns(['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5', 'storytitle+endoftext', 'story'])

    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add idx column
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists('/nvme/xnli/lk_code/exps/rtv_icl/data'):
        base_dir = '/nvme/xnli/lk_code/exps/rtv_icl/data/roc_story_generation'
    else:
        base_dir = '/remote-home/klv/exps/rtv_icl/data/roc_story_generation'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def sample_squadv2():
    dataset = datasets.load_dataset('GEM/squad_v2')
    print(dataset)
    dataset = dataset.filter(lambda x: len(x['answers']['text']) > 0, num_proc=8)
    dataset = dataset.remove_columns(['question', 'id'])

    def process_squadv2(example):
        answer = example['answers']['text'][0]
        example['input'] = 'Generate the question corresponding to the answer. The text is as follows. ' + example[
            'context'] + 'The answer is as follows. ' + answer
        return example

    dataset = dataset.map(process_squadv2, num_proc=8)
    # add column idx
    for split in ['train', 'validation', 'test']:
        ds_id = Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = concatenate_datasets([ds_id, dataset[split]], axis=1)
    # overfit for debug
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)

    data_dir = "/remote-home/klv/exps/rtv_icl/data"
    if not os.path.exists(data_dir):
        data_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data"

    base_dir = data_dir + "/squadv2"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(dataset)
    dataset.save_to_disk(base_dir)


def sample_spider():
    dataset = datasets.load_dataset('spider')
    print(dataset)
    dataset = dataset.remove_columns(['query_toks', 'question_toks', 'query_toks_no_value'])

    # add column idx
    for split in ['train', 'validation']:
        ds_id = Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = concatenate_datasets([ds_id, dataset[split]], axis=1)
    # overfit for debug
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)

    data_dir = "/remote-home/klv/exps/rtv_icl/data"
    if not os.path.exists(data_dir):
        data_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data"

    base_dir = data_dir + "/spider"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(dataset)
    dataset.save_to_disk(base_dir)


def sample_opusparcus():
    data_dir = "/remote-home/klv/exps/rtv_icl/data"
    if not os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        data_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data"
    ds = datasets.load_dataset('json',
                               data_files=data_dir + '/opusparcus/train_en.70.jsonl')
    ds['train'] = ds['train'].filter(lambda x: x['quality'] >= 95 and x['sent1'] != x['sent2'], num_proc=8)
    ds['train'] = ds['train'].rename_columns({'sent1': 'input', 'sent2': 'target'})
    test_ds = load_dataset("GEM/opusparcus", lang="en", quality=100, ignore_verifications=True)
    ds['test'] = test_ds['test']
    ds['validation'] = test_ds['validation']
    # 去重
    for split in ['train', 'validation', 'test']:
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['input'] not in x_set:
                x_set.add(e['input'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    # overfit for debug
    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)
    base_dir = data_dir + "/opusparcus"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def sample_smcalflow():
    data_dir = "/remote-home/klv/exps/rtv_icl/data"
    if not os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        data_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data"
    data_dir = "data"
    ds = load_dataset("iohadrubin/smcalflow", name="smcalflow")
    # 去重
    for split in ['train']:
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['user_utterance'] not in x_set:
                x_set.add(e['user_utterance'])
                f += 1
            if e['fully_typed_lispress'] not in y_set:
                y_set.add(e['fully_typed_lispress'])
                f += 1
            if f == 2:
                lst.append(i)
        ds[split] = ds[split].select(lst)
    #
    # for split in ['train', 'validation']:
    #     ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
    #     ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)
    # overfit for debug
    # slct = [i for i in range(len(ds['train']))]
    # random.seed(42)
    # smp = random.sample(slct, 100000)
    # ds['debug'] = ds['train'].select(smp)
    base_dir = data_dir + "/smcalflow"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def sample_mtop_break():
    dataset = load_dataset("break_data", "QDMR")
    dataset.save_to_disk("data/break")
    dataset = load_dataset("iohadrubin/mtop", name="mtop")
    dataset.save_to_disk("data/mtop")

def sample_xsum():
    # ds = load_dataset("xsum")
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = datasets.load_from_disk("/nvme/xnli/lk_code/exps/rtv_icl/data/xsum_from3090")
    print(ds)
    ds = ds.remove_columns(['id'])

    def process_xsum(example):
        example['document'] = ' '.join(example['document'].replace('\n', ' ').split())
        example['summary'] = ' '.join(example['summary'].replace('\n', ' ').split())
        example['len_doc'] = len(tknz(example['document']).input_ids)
        example['len_sum'] = len(tknz(example['summary']).input_ids)
        return example

    ds = ds.map(process_xsum, num_proc=16)
    ds = ds.filter(lambda x: x['len_doc'] + x['len_sum'] < 900, num_proc=16)
    # add column idx
    for split in ['train', 'validation', 'test']:
        ds_id = Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = concatenate_datasets([ds_id, ds[split]], axis=1)
    # overfit for debug
    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)
    print(ds)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/xsum"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/xsum"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    ds.save_to_disk(base_dir)


def sample_cnndailymail():
    # ds = load_dataset("xsum")

    ds = load_dataset("cnn_dailymail", '3.0.0')
    print(ds)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = ds.remove_columns(['id'])

    def process_xsum(example):
        # example['article'] = ' '.join(example['article'].replace('\n', ' ').split())
        # example['highlights'] = ' '.join(example['highlights'].replace('\n', ' ').split())
        example['len_article'] = len(tknz(example['article']).input_ids)
        example['len_highlights'] = len(tknz(example['highlights']).input_ids)
        return example

    ds = ds.map(process_xsum, num_proc=16)
    ds = ds.filter(lambda x: x['len_article'] + x['len_highlights'] < 900, num_proc=16)
    # add column idx
    for split in ['train', 'validation', 'test']:
        ds_id = Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = concatenate_datasets([ds_id, ds[split]], axis=1)
    # overfit for debug
    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 100000)
    ds['debug'] = ds['train'].select(smp)
    print(ds)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/cnndailymail"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/cnndailymail"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    ds.save_to_disk(base_dir)


def sample_reddit():
    import json
    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/train_reddit.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_train = Dataset.from_list(d)
    ds_train = ds_train.remove_columns(['label'])

    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/val_reddit.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_val = Dataset.from_list(d)

    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/test_reddit.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_test = Dataset.from_list(d)
    ds_test = ds_test.remove_columns(['label'])

    ds = DatasetDict({'train': ds_train, 'validation': ds_val, 'test': ds_test})
    print(ds)
    ds = ds.remove_columns(['ext_idx', 'indices', 'score', 'candidate_id', 'text_id', 'summary_id'])
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    # ds = ds.remove_columns(['author', 'body', 'normalizedBody', 'subreddit', 'subreddit_id', 'id'])
    ds = ds.rename_columns({'text': 'question', 'summary': 'target'})

    def process_xsum(example):
        example['question'] = ' '.join(example['question'])
        example['target'] = ' '.join(example['target'])
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    ds = ds.map(process_xsum, num_proc=16)
    ds = ds.filter(lambda x: x['len_question'] + x['len_target'] < 900, num_proc=16)
    # add column idx
    for split in ['train', 'validation', 'test']:
        ds_id = Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = concatenate_datasets([ds_id, ds[split]], axis=1)
    # overfit for debug
    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)
    print(ds)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/reddit"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/reddit"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    ds.save_to_disk(base_dir)


def sample_wikihow():
    import json
    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/train_wikihow.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_train = Dataset.from_list(d)
    ds_train = ds_train.remove_columns(['label'])

    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/val_wikihow.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_val = Dataset.from_list(d)

    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/test_wikihow.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_test = Dataset.from_list(d)
    ds_test = ds_test.remove_columns(['label'])

    ds = DatasetDict({'train': ds_train, 'validation': ds_val, 'test': ds_test})
    print(ds)
    ds = ds.remove_columns(['ext_idx', 'indices', 'score', 'candidate_id', 'text_id', 'summary_id'])
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    # ds = ds.remove_columns(['author', 'body', 'normalizedBody', 'subreddit', 'subreddit_id', 'id'])
    ds = ds.rename_columns({'text': 'question', 'summary': 'target'})

    def process_xsum(example):
        example['question'] = ' '.join(example['question'])
        example['target'] = ' '.join(example['target'])
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    ds = ds.map(process_xsum, num_proc=16)
    ds = ds.filter(lambda x: x['len_question'] + x['len_target'] < 900, num_proc=16)
    # add column idx
    for split in ['train', 'validation', 'test']:
        ds_id = Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = concatenate_datasets([ds_id, ds[split]], axis=1)
    # overfit for debug
    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)
    print(ds)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/wikihow"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/wikihow"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    ds.save_to_disk(base_dir)


def sample_pubmed():
    import json
    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/train_pubmed.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_train = Dataset.from_list(d)
    ds_train = ds_train.remove_columns(['label'])

    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/val_pubmed.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_val = Dataset.from_list(d)

    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/test_pubmed.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_test = Dataset.from_list(d)
    ds_test = ds_test.remove_columns(['label'])

    ds = DatasetDict({'train': ds_train, 'validation': ds_val, 'test': ds_test})
    print(ds)
    ds = ds.remove_columns(['ext_idx', 'indices', 'score', 'candidate_id', 'text_id', 'summary_id'])

    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    # ds = ds.remove_columns(['author', 'body', 'normalizedBody', 'subreddit', 'subreddit_id', 'id'])
    ds = ds.rename_columns({'text': 'question', 'summary': 'target'})

    def process_xsum(example):
        example['question'] = ' '.join(example['question'])
        example['target'] = ' '.join(example['target'])
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    ds = ds.map(process_xsum, num_proc=16)
    ds = ds.filter(lambda x: x['len_question'] + x['len_target'] < 900, num_proc=16)
    # add column idx

    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    # overfit for debug
    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)
    print(ds)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/pubmed"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/pubmed"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    ds.save_to_disk(base_dir)

def sample_multinews():
    import json
    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/train_multinews.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_train = Dataset.from_list(d)
    ds_train = ds_train.remove_columns(['label'])

    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/val_multinews.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_val = Dataset.from_list(d)

    d = []
    with open('/nvme/xnli/lk_code/exps/rtv_icl/data/sum/test_multinews.jsonl', 'r') as f:
        for l in f.readlines():
            d.append(json.loads(l))
    ds_test = Dataset.from_list(d)
    ds_test = ds_test.remove_columns(['label'])

    ds = DatasetDict({'train': ds_train, 'validation': ds_val, 'test': ds_test})
    print(ds)
    ds = ds.remove_columns(['ext_idx', 'indices', 'score', 'candidate_id', 'text_id', 'summary_id'])

    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    # ds = ds.remove_columns(['author', 'body', 'normalizedBody', 'subreddit', 'subreddit_id', 'id'])
    ds = ds.rename_columns({'text': 'question', 'summary': 'target'})

    def process_xsum(example):
        example['question'] = ' '.join(example['question'])
        example['target'] = ' '.join(example['target'])
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    ds = ds.map(process_xsum, num_proc=16)
    ds = ds.filter(lambda x: x['len_question'] + x['len_target'] < 900, num_proc=16)
    # add column idx

    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    # overfit for debug
    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)
    print(ds)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/multinews"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/multinews"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    ds.save_to_disk(base_dir)

def sample_kp20k():
    dataset = load_dataset("midas/kp20k", "generation")
    # 去掉extractive_keyphrases为空的数据 去掉包含<eos>的数据
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
        ds_id = Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = concatenate_datasets([ds_id, dataset[split]], axis=1)
    # overfit for debug
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)
    base_dir = "/remote-home/klv/exps/rtv_icl/data/kp20k"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)


def sample_wikiauto():
    dataset = datasets.load_dataset('GEM/wiki_auto_asset_turk', 'train')
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    for split in ['train', 'validation', 'test_asset', 'test_turk', "test_wiki"]:
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['source'] not in x_set:
                x_set.add(e['source'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)

    def process_python(example):
        example['len_source'] = len(tknz(example['source']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_python, num_proc=8)
    dataset = dataset.filter(lambda x: x['len_source'] + x['len_target'] < 900)

    # add idx column
    for split in ['train', 'validation', 'test_asset', 'test_turk', "test_wiki"]:
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    print(dataset)
    slct = [i for i in range(len(dataset['train']))]
    print(slct[:50])
    print(slct[-50:])
    random.seed(42)
    smp = random.sample(slct, 100000)

    dataset['debug'] = dataset['train'].select(smp)
    data_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data"
    base_dir = data_dir + "/wikiauto"
    dataset.save_to_disk(base_dir)


def sample_iwslt():
    dataset = datasets.load_dataset("iwslt2017", "iwslt2017-de-en")
    dataset = dataset.flatten()
    print(dataset)
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['translation.de'] not in x_set:
                x_set.add(e['translation.de'])
                f += 1
            if e['translation.en'] not in y_set:
                y_set.add(e['translation.en'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)

        # lst = list(dataset[split])
        # st = set([tuple(d.items()) for d in lst])
        # lst = [dict(d) for d in st]
        # dataset[split] = datasets.Dataset.from_list(lst)
        # ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        # dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    print(slct[:50])
    print(slct[-50:])
    random.seed(42)
    smp = random.sample(slct, 20000)
    print(len(smp))
    print(smp[:50])
    print(smp[-50:])
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/iwslt"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/iwslt"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)


def sample_iwslt_en_fr():
    dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-fr")
    dataset = dataset.flatten()
    dataset = dataset.rename_columns({"translation.en": "question", "translation.fr": "target"})
    print(dataset)
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)

        # lst = list(dataset[split])
        # st = set([tuple(d.items()) for d in lst])
        # lst = [dict(d) for d in st]
        # dataset[split] = datasets.Dataset.from_list(lst)
        # ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        # dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    print(slct[:50])
    print(slct[-50:])
    random.seed(42)
    smp = random.sample(slct, 5000)
    print(len(smp))
    print(smp[:50])
    print(smp[-50:])
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/iwslt_en_fr"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/iwslt_en_fr"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)


def sample_iwslt_en_de():
    dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-de")
    dataset = dataset.flatten()
    dataset = dataset.rename_columns({"translation.en": "question", "translation.de": "target"})
    print(dataset)
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    print(slct[:50])
    print(slct[-50:])
    random.seed(42)
    smp = random.sample(slct, 5000)
    print(len(smp))
    print(smp[:50])
    print(smp[-50:])
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/iwslt_en_de"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/iwslt_en_de"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)

def sample_wmt_en_de():
    dataset = datasets.load_dataset("wmt16", "de-en")
    dataset = dataset.flatten()
    dataset = dataset.rename_columns({"translation.en": "question", "translation.de": "target"})
    print(dataset)
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)

    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    def process_wmt(example):
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_wmt, num_proc=32)
    dataset = dataset.filter(lambda x: x['len_question'] + x['len_target'] < 900)

    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)

    smp2 = random.sample(slct, 200000)
    dataset['train'] = dataset['train'].select(smp2)

    for split in ['train', 'validation', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/wmt_en_de"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/wmt_en_de"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)


def sample_wmt_de_en():
    dataset = datasets.load_dataset("wmt16", "de-en")
    dataset = dataset.flatten()
    dataset = dataset.rename_columns({"translation.de": "question", "translation.en": "target"})
    print(dataset)
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)

    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    def process_wmt(example):
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_wmt, num_proc=32)
    dataset = dataset.filter(lambda x: x['len_question'] + x['len_target'] < 900)

    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)

    smp2 = random.sample(slct, 200000)
    dataset['train'] = dataset['train'].select(smp2)

    for split in ['train', 'validation', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/wmt_de_en"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/wmt_de_en"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)

def sample_e2e():
    dataset = datasets.load_dataset("GEM/e2e_nlg")
    print(dataset)

    def process_e2e(example):
        example['references'] = "\n".join(example['references'])
        return example

    dataset = dataset.map(process_e2e, num_proc=8)
    dataset = dataset.rename_columns({"meaning_representation": "question"})
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/e2e"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/e2e"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)


def sample_python():
    base_url = "/nvme/xnli/lk_code/exps/rtv_icl/data/code_comment/dataset/python/"
    dataset = load_dataset("json",
                           data_files={"train": base_url + "train.jsonl", "validation": base_url + "valid.jsonl",
                                       'test': base_url + "test.jsonl"})
    print(dataset)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    def process_python(example):
        example['question'] = " ".join(" ".join(example['code_tokens']).replace("\n", ' ').split())
        example['target'] = " ".join(" ".join(example['docstring_tokens']).replace("\n", ' ').split())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_python, num_proc=8)
    dataset = dataset.filter(lambda x: x['len_question'] + x['len_target'] < 900)
    dataset = dataset.remove_columns(['repo', 'path', 'func_name', 'original_string', 'code_tokens', 'docstring_tokens',
                                      'code', 'docstring', 'language', 'sha', 'url', 'partition'])
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 100000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/python"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/python"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)


def sample_ruby():
    base_url = "/nvme/xnli/lk_code/exps/rtv_icl/data/code_comment/dataset/ruby/"
    dataset = load_dataset("json",
                           data_files={"train": base_url + "train.jsonl", "validation": base_url + "valid.jsonl",
                                       'test': base_url + "test.jsonl"})
    print(dataset)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    def process_ruby(example):
        example['question'] = " ".join(" ".join(example['code_tokens']).replace("\n", ' ').split())
        example['target'] = " ".join(" ".join(example['docstring_tokens']).replace("\n", ' ').split())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_ruby, num_proc=8)
    dataset = dataset.filter(lambda x: x['len_question'] + x['len_target'] < 900)
    dataset = dataset.remove_columns(['repo', 'path', 'func_name', 'original_string', 'code_tokens', 'docstring_tokens',
                                      'code', 'docstring', 'language', 'sha', 'url', 'partition'])
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/ruby"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/ruby"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)

def sample_java():
    base_url = "/nvme/xnli/lk_code/exps/rtv_icl/data/code_comment/dataset/java/"
    dataset = load_dataset("json",
                           data_files={"train": base_url + "train.jsonl", "validation": base_url + "valid.jsonl",
                                       'test': base_url + "test.jsonl"})
    print(dataset)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    def process_java(example):
        example['question'] = " ".join(" ".join(example['code_tokens']).replace("\n", ' ').split())
        example['target'] = " ".join(" ".join(example['docstring_tokens']).replace("\n", ' ').split())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_java, num_proc=8)
    dataset = dataset.filter(lambda x: x['len_question'] + x['len_target'] < 900)
    dataset = dataset.remove_columns(['repo', 'path', 'func_name', 'original_string', 'code_tokens', 'docstring_tokens',
                                      'code', 'docstring', 'language', 'sha', 'url', 'partition'])
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 100000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/java"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/java"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)

def sample_javascript():
    base_url = "/nvme/xnli/lk_code/exps/rtv_icl/data/code_comment/dataset/javascript/"
    dataset = load_dataset("json",
                           data_files={"train": base_url + "train.jsonl", "validation": base_url + "valid.jsonl",
                                       'test': base_url + "test.jsonl"})
    print(dataset)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    def process_java(example):
        example['question'] = " ".join(" ".join(example['code_tokens']).replace("\n", ' ').split())
        example['target'] = " ".join(" ".join(example['docstring_tokens']).replace("\n", ' ').split())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_java, num_proc=8)
    dataset = dataset.filter(lambda x: x['len_question'] + x['len_target'] < 900)
    dataset = dataset.remove_columns(['repo', 'path', 'func_name', 'original_string', 'code_tokens', 'docstring_tokens',
                                      'code', 'docstring', 'language', 'sha', 'url', 'partition'])
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/javascript"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/javascript"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)

def sample_go():
    base_url = "/nvme/xnli/lk_code/exps/rtv_icl/data/code_comment/dataset/go/"
    dataset = load_dataset("json",
                           data_files={"train": base_url + "train.jsonl", "validation": base_url + "valid.jsonl",
                                       'test': base_url + "test.jsonl"})
    print(dataset)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    def process_ruby(example):
        example['question'] = " ".join(" ".join(example['code_tokens']).replace("\n", ' ').split())
        example['target'] = " ".join(" ".join(example['docstring_tokens']).replace("\n", ' ').split())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_ruby, num_proc=8)
    dataset = dataset.filter(lambda x: x['len_question'] + x['len_target'] < 900)
    dataset = dataset.remove_columns(['repo', 'path', 'func_name', 'original_string', 'code_tokens', 'docstring_tokens',
                                      'code', 'docstring', 'language', 'sha', 'url', 'partition'])
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 100000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/go"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/go"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)

def sample_php():
    base_url = "/nvme/xnli/lk_code/exps/rtv_icl/data/code_comment/dataset/php/"
    dataset = load_dataset("json",
                           data_files={"train": base_url + "train.jsonl", "validation": base_url + "valid.jsonl",
                                       'test': base_url + "test.jsonl"})
    print(dataset)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    def process_ruby(example):
        example['question'] = " ".join(" ".join(example['code_tokens']).replace("\n", ' ').split())
        example['target'] = " ".join(" ".join(example['docstring_tokens']).replace("\n", ' ').split())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_ruby, num_proc=8)
    dataset = dataset.filter(lambda x: x['len_question'] + x['len_target'] < 900)
    dataset = dataset.remove_columns(['repo', 'path', 'func_name', 'original_string', 'code_tokens', 'docstring_tokens',
                                      'code', 'docstring', 'language', 'sha', 'url', 'partition'])
    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)
    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 100000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/php"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/php"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)

def sample_dart():
    dataset = load_dataset("GEM/dart")
    dataset = dataset.remove_columns(['gem_id', 'gem_parent_id', 'dart_id', 'target_sources', 'subtree_was_extended'])
    dataset = dataset.rename_columns({'tripleset': 'question'})
    print(dataset)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    def process_dart(example):
        example['question'] = ", ".join("[" + x[0] + ' | ' + x[1] + ' | ' + x[2] + "]" for x in example['question'])
        example['references'] = "\n".join(example['references'])
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_dart, num_proc=16)
    dataset = dataset.filter(lambda x: x['len_question'] + x['len_target'] < 900)

    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)

    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/dart"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/dart"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)

def sample_totto():
    dataset = load_dataset("GEM/totto")
    dataset = dataset.remove_columns(['gem_id', 'gem_parent_id', 'totto_id', 'table_page_title', 'table_webpage_url',
                                      'table_section_text', 'table', 'highlighted_cells', 'example_id', 'sentence_annotations',
                                      'overlap_subset'])
    dataset = dataset.rename_columns({'linearized_input': 'question'})
    print(dataset)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    def process_dart(example):
        example['references'] = "\n".join(example['references'])
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['len_target'] = len(tknz(example['target']).input_ids)
        return example

    dataset = dataset.map(process_dart, num_proc=16)
    dataset = dataset.filter(lambda x: x['len_question'] + x['len_target'] < 900)

    # add idx column
    for split in ['train', 'validation', 'test']:
        # 去重
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(dataset[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if e['target'] not in y_set:
                y_set.add(e['target'])
                f += 1
            if f == 2:
                lst.append(i)
        dataset[split] = dataset[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        dataset[split] = datasets.concatenate_datasets([ds_id, dataset[split]], axis=1)

    slct = [i for i in range(len(dataset['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    dataset['debug'] = dataset['train'].select(smp)
    print(dataset)
    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/totto"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/totto"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dataset.save_to_disk(base_dir)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", type=str, default=None)
    args = parser.parse_args()
    method_name = "sample_" + args.d

    symbol_table = globals()

    # get the function from the symbol table
    func = symbol_table.get(method_name)
    print("sample dataset")
    # call the function
    func()
