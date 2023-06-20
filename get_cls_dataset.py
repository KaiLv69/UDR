import os, random, datasets
from tqdm import tqdm

def get_trec():
    import pandas as pd
    from datasets import Dataset, DatasetDict
    df = pd.read_csv("/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/trec/train.csv",
                     names=['label', 'sentence'])
    ds_train = Dataset.from_pandas(df)

    df = pd.read_csv("/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/trec/test.csv",
                     names=['label', 'sentence'])
    ds_test = Dataset.from_pandas(df)

    ds = DatasetDict({'train': ds_train, 'test': ds_test})
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/trec"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/trec"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_sst2():
    import pandas as pd
    from datasets import Dataset, DatasetDict
    df = pd.read_csv("/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/SST-2/train.tsv",
                     sep='\t',)
    ds_train = Dataset.from_pandas(df)

    df = pd.read_csv("/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/SST-2/test.tsv",
                     sep='\t',
                     names=['label', 'sentence'])
    ds_test = Dataset.from_pandas(df)

    ds = DatasetDict({'train': ds_train, 'test': ds_test})
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/sst2"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/sst2"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_emotion():
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("mteb/emotion")
    print(ds)

    ds = ds.rename_column("text", "sentence")

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    # slct = [i for i in range(len(ds['train']))]
    # random.seed(42)
    # smp = random.sample(slct, 5000)
    # ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/emotion"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/emotion"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_tweet_sentiment_extraction():
    from datasets import load_dataset

    ds = load_dataset("mteb/tweet_sentiment_extraction")
    ds = ds.map(lambda x: {'sentence': x['text'].strip(), 'label': x['label']})
    ds = ds.remove_columns(['text', 'id'])
    print(ds)


    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    # slct = [i for i in range(len(ds['train']))]
    # random.seed(42)
    # smp = random.sample(slct, 5000)
    # ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/tweet_sentiment_extraction"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/tweet_sentiment_extraction"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_amazon_scenario():
    from datasets import load_dataset

    ds = load_dataset("mteb/amazon_massive_scenario")
    ds = ds.map(lambda x: {'sentence': x['text'].strip()})
    ds = ds.remove_columns(['text', 'id'])
    print(ds)


    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    # slct = [i for i in range(len(ds['train']))]
    # random.seed(42)
    # smp = random.sample(slct, 5000)
    # ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/amazon_scenario"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/amazon_scenario"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_sst5():
    from datasets import load_dataset
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/sst-5/train.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/sst-5/test.csv"},
                           column_names=["label", "sentence"])
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/sst5"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/sst5"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_cr():
    from datasets import load_dataset
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/cr/train.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/cr/test.csv"},
                           column_names=["label", "sentence"])
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/cr"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/cr"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_bank77():
    from datasets import load_dataset
    ds = load_dataset("mteb/banking77")
    ds = ds.rename_column("text", 'sentence')
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/bank77"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/bank77"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_imdb():
    from datasets import load_dataset
    ds = load_dataset("mteb/imdb")
    ds = ds.rename_column('text', 'sentence')
    print(ds)

    slct = [i for i in range(len(ds['test']))]
    random.seed(42)
    smp = random.sample(slct, 3000)
    ds['test'] = ds['test'].select(smp)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/imdb"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/imdb"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_boolq():
    from datasets import load_dataset
    ds = load_dataset("boolq")
    print(ds)
    def process_mnli(example):
        example["sentence"] = example["passage"] + " Based on the passage, " + \
                              example["question"] + '?'
        if example['answer'] is True:
            example['label'] = 1
        else:
            example['label'] = 0
        return example
    ds = ds.map(process_mnli, remove_columns=['question', 'passage'])

    for split in ['train', 'validation']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/boolq"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/boolq"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_financial_phrasebank():
    from datasets import load_dataset
    ds = load_dataset("financial_phrasebank", 'sentences_50agree')
    print(ds)

    for split in ['train']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)

    print(ds)
    random.seed(42)
    slct = [i for i in range(len(ds['train']))]
    smp = random.sample(slct, 500)
    train_smp = [x for x in slct if x not in smp]
    ds['test'] = ds['train'].select(smp)
    ds['train'] = ds['train'].select(train_smp)

    for split in ['train', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/financial_phrasebank"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/financial_phrasebank"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_mnli():
    from datasets import load_dataset

    ds = load_dataset("SetFit/mnli")
    ds_mm = load_dataset("SetFit/mnli_mm")
    ds['validation_mm'] = ds_mm['validation']
    print(ds)

    for split in ['train', 'validation', 'validation_mm']:
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['text1'] not in x_set:
                x_set.add(e['text1'])
                f += 1
            if e['text2'] not in y_set:
                y_set.add(e['text2'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
    print(ds)

    word2label = {"entailment": 0, "neutral": 1, "contradiction": 2}
    def process_mnli(example):
        example["sentence"] = example["text1"] + " Based on that information, is the claim " + \
                              example["text2"] + ' "Entailment", "Contradiction", or "Inconclusive"?'
        return example
    ds = ds.map(process_mnli, remove_columns=['text1', 'text2', 'idx'])

    for split in ['train', 'validation', 'validation_mm']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    random.seed(42)
    slct = [i for i in range(len(ds['train']))]
    smp = random.sample(slct, 100000)
    ds['debug'] = ds['train'].select(smp)
    slct = [i for i in range(len(ds['validation']))]
    smp = random.sample(slct, 3000)
    ds['validation'] = ds['validation'].select(smp)
    slct = [i for i in range(len(ds['validation_mm']))]
    smp = random.sample(slct, 3000)
    ds['validation_mm'] = ds['validation_mm'].select(smp)
    #
    # if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
    #     base_dir = "/remote-home/klv/exps/rtv_icl/data/mnli"
    # else:
    #     base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/mnli"
    base_dir = "data/mnli"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_qnli():
    from datasets import load_dataset

    ds = load_dataset("SetFit/qnli")
    print(ds)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    for split in ['train', 'validation']:
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['text1'] not in x_set:
                x_set.add(e['text1'])
                f += 1
            if e['text2'] not in y_set:
                y_set.add(e['text2'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
    print(ds)

    word2label = {"entailment": 0, "neutral": 1, "contradiction": 2}
    def process_mnli(example):
        example["sentence"] = example["text2"] + " Based on that information, is the claim " + \
                              example["text1"] + ' "Entailment" or "Inconclusive"?'
        example['len_sentence'] = len(tknz(example['sentence']).input_ids)
        return example
    ds = ds.map(process_mnli, remove_columns=['text1', 'text2', 'idx'])
    ds = ds.filter(lambda x: x['len_sentence'] < 900)
    for split in ['train', 'validation']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    # random.seed(42)
    # slct = [i for i in range(len(ds['train']))]
    # smp = random.sample(slct, 5000)
    # ds['debug'] = ds['train'].select(smp)
    # slct = [i for i in range(len(ds['validation']))]
    # smp = random.sample(slct, 3000)
    # ds['validation'] = ds['validation'].select(smp)
    # slct = [i for i in range(len(ds['validation_mm']))]
    # smp = random.sample(slct, 3000)
    # ds['validation_mm'] = ds['validation_mm'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/qnli"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/qnli"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_wnli():
    from datasets import load_dataset

    ds = load_dataset("SetFit/wnli")
    print(ds)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    for split in ['train']:
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['text1'] not in x_set:
                x_set.add(e['text1'])
                f += 1
            if e['text2'] not in y_set:
                y_set.add(e['text2'])
                f += 1
            if f == 2:
                lst.append(i)
        ds[split] = ds[split].select(lst)
    print(ds)

    def process_mnli(example):
        example["sentence"] = example["text1"] + " Based on that information, is the claim " + \
                              example["text2"] + ' "Entailment" or "Inconclusive"?'
        example['len_sentence'] = len(tknz(example['sentence']).input_ids)
        return example
    ds = ds.map(process_mnli, remove_columns=['text1', 'text2', 'idx'])
    ds = ds.filter(lambda x: x['len_sentence'] < 900)
    for split in ['train', 'validation']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    # random.seed(42)
    # slct = [i for i in range(len(ds['train']))]
    # smp = random.sample(slct, 5000)
    # ds['debug'] = ds['train'].select(smp)
    # slct = [i for i in range(len(ds['validation']))]
    # smp = random.sample(slct, 3000)
    # ds['validation'] = ds['validation'].select(smp)
    # slct = [i for i in range(len(ds['validation_mm']))]
    # smp = random.sample(slct, 3000)
    # ds['validation_mm'] = ds['validation_mm'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/wnli"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/wnli"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_snli():
    from datasets import load_dataset

    ds = load_dataset("snli")
    print(ds)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    for split in ['train', 'validation', 'test']:
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['premise'] not in x_set:
                x_set.add(e['premise'])
                f += 1
            if e['hypothesis'] not in y_set:
                y_set.add(e['hypothesis'])
                f += 1
            if f == 2:
                lst.append(i)
        ds[split] = ds[split].select(lst)
    print(ds)

    def process_mnli(example):
        example["sentence"] = example["premise"] + " Based on that information, is the claim " + \
                              example["hypothesis"] + ' "Entailment", "Contradiction", or "Inconclusive"?'
        example['len_sentence'] = len(tknz(example['sentence']).input_ids)
        return example
    ds = ds.map(process_mnli, remove_columns=['premise', 'hypothesis'])
    ds = ds.filter(lambda x: x['len_sentence'] < 900)
    for split in ['train', 'validation', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    random.seed(42)
    slct = [i for i in range(len(ds['train']))]
    smp = random.sample(slct, 100000)
    ds['debug'] = ds['train'].select(smp)
    # slct = [i for i in range(len(ds['validation']))]
    # smp = random.sample(slct, 3000)
    # ds['validation'] = ds['validation'].select(smp)
    # slct = [i for i in range(len(ds['validation_mm']))]
    # smp = random.sample(slct, 3000)
    # ds['validation_mm'] = ds['validation_mm'].select(smp)

    # if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
    #     base_dir = "/remote-home/klv/exps/rtv_icl/data/snli"
    # else:
    #     base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/snli"
    base_dir = "data/snli"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_rte():
    from datasets import load_dataset

    ds = load_dataset("SetFit/rte")
    print(ds)

    # for split in ['train']:
    #     x_set = set([])
    #     y_set = set([])
    #     lst = []
    #     for i, e in tqdm(enumerate(ds[split])):
    #         f = 0
    #         if e['text1'] not in x_set:
    #             x_set.add(e['text1'])
    #             f += 1
    #         if e['text2'] not in y_set:
    #             y_set.add(e['text2'])
    #             f += 1
    #         if f == 1:
    #             lst.append(i)
    #     ds[split] = ds[split].select(lst)
    # print(ds)

    word2label = {0: True, 1: False}
    def process_mnli(example):
        example["sentence"] = example["text1"] + " Question: " + \
                              example["text2"] + ' Ture of False?'
        return example
    ds = ds.map(process_mnli, remove_columns=['text1', 'text2', 'idx'])

    for split in ['train', 'validation']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/rte"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/rte"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_mrpc():
    from datasets import load_dataset

    ds = load_dataset("SetFit/mrpc")
    print(ds)
    ds = ds.remove_columns(['idx'])
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    for split in ['train', 'validation', 'test']:
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['text1'] not in x_set:
                x_set.add(e['text1'])
                f += 1
            if e['text2'] not in y_set:
                y_set.add(e['text2'])
                f += 1
            if f == 2:
                lst.append(i)
        ds[split] = ds[split].select(lst)
    print(ds)

    def process_mnli(example):
        example["sentence"] = "Sentence 1: " + example["text1"] + " Sentence 2: " + \
                              example["text2"] + ' Are these two sentences semantically equivalent?'
        example['len_sentence'] = len(tknz(example['sentence']).input_ids)
        return example
    ds = ds.map(process_mnli, remove_columns=['text1', 'text2'])
    ds = ds.filter(lambda x: x['len_sentence'] < 900)
    for split in ['train', 'validation', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    random.seed(42)
    # slct = [i for i in range(len(ds['train']))]
    # smp = random.sample(slct, 100000)
    # ds['debug'] = ds['train'].select(smp)
    # slct = [i for i in range(len(ds['validation']))]
    # smp = random.sample(slct, 3000)
    # ds['validation'] = ds['validation'].select(smp)
    # slct = [i for i in range(len(ds['validation_mm']))]
    # smp = random.sample(slct, 3000)
    # ds['validation_mm'] = ds['validation_mm'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/mrpc"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/mrpc"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_qqp():
    from datasets import load_dataset

    ds = load_dataset("SetFit/qqp")
    print(ds)
    ds = ds.remove_columns(['idx'])
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    for split in ['train', 'validation', 'test']:
        x_set = set([])
        y_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['text1'] not in x_set:
                x_set.add(e['text1'])
                f += 1
            if e['text2'] not in y_set:
                y_set.add(e['text2'])
                f += 1
            if f == 2:
                lst.append(i)
        ds[split] = ds[split].select(lst)
    print(ds)

    def process_mnli(example):
        example["sentence"] = "Question 1: " + example["text1"] + " Question 2: " + \
                              example["text2"] + ' Are these two questions semantically equivalent?'
        example['len_sentence'] = len(tknz(example['sentence']).input_ids)
        return example
    ds = ds.map(process_mnli, remove_columns=['text1', 'text2'])
    ds = ds.filter(lambda x: x['len_sentence'] < 900)
    for split in ['train', 'validation', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    random.seed(42)
    # slct = [i for i in range(len(ds['train']))]
    # smp = random.sample(slct, 100000)
    # ds['debug'] = ds['train'].select(smp)
    slct = [i for i in range(len(ds['validation']))]
    smp = random.sample(slct, 3000)
    ds['validation'] = ds['validation'].select(smp)
    # slct = [i for i in range(len(ds['validation_mm']))]
    # smp = random.sample(slct, 3000)
    # ds['validation_mm'] = ds['validation_mm'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/qqp"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/qqp"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_mtop_domain():
    from datasets import load_dataset

    ds = load_dataset("mteb/mtop_domain")
    print(ds)
    ds = ds.remove_columns(['id'])
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    for split in ['train', 'validation', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['text'] not in x_set:
                x_set.add(e['text'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
    print(ds)
    ds = ds.rename_column('text', 'sentence')
    # def process_mnli(example):
    #     example["sentence"] = example["text"]
    #     example['len_sentence'] = len(tknz(example['sentence']).input_ids)
    #     return example
    # ds = ds.map(process_mnli, remove_columns=['text', 'text2'])
    # ds = ds.filter(lambda x: x['len_sentence'] < 900)
    for split in ['train', 'validation', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    random.seed(42)
    # slct = [i for i in range(len(ds['train']))]
    # smp = random.sample(slct, 100000)
    # ds['debug'] = ds['train'].select(smp)
    # slct = [i for i in range(len(ds['validation']))]
    # smp = random.sample(slct, 3000)
    # ds['validation'] = ds['validation'].select(smp)
    # slct = [i for i in range(len(ds['validation_mm']))]
    # smp = random.sample(slct, 3000)
    # ds['validation_mm'] = ds['validation_mm'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/mtop_domain"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/mtop_domain"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_mr():
    from datasets import load_dataset
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/mr/train.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/mr/test.csv"},
                           column_names=["label", "sentence"])
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/mr"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/mr"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_subj():
    from datasets import load_dataset
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/subj/train.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/subj/test.csv"},
                           column_names=["label", "sentence"])
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/subj"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/subj"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_yelp_full():
    from datasets import load_dataset
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/TextClassificationDatasets/yelp_review_full_csv_small/train_30000.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/TextClassificationDatasets/yelp_review_full_csv_small/test.csv"},
                           column_names=["label", "sentence"])
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    def process_label(example):
        example['label'] = example['label'] - 1
        return example

    ds = ds.map(process_label, num_proc=8)

    random.seed(42)

    slct = [i for i in range(len(ds['train']))]
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)
    slct = [i for i in range(len(ds['test']))]
    smp = random.sample(slct, 3000)
    ds['test'] = ds['test'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/yelp_full"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/yelp_full"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_amazon():
    from datasets import load_dataset
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/TextClassificationDatasets/amazon_review_full_csv_small/train_30000.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/TextClassificationDatasets/amazon_review_full_csv_small/test.csv"},
                           column_names=["label", "headline", "sentence"])
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)

    slct = [i for i in range(len(ds['test']))]
    smp = random.sample(slct, 3000)
    ds['test'] = ds['test'].select(smp)

    for split in ['train', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    def process_label(example):
        example['label'] = example['label'] - 1
        return example

    ds = ds.map(process_label, num_proc=8)

    random.seed(42)

    slct = [i for i in range(len(ds['train']))]
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)


    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/amazon"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/amazon"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_agnews():
    from datasets import load_dataset
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/TextClassificationDatasets/ag_news_csv_small/train_30000.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/TextClassificationDatasets/ag_news_csv_small/test.csv"},
                           column_names=["label", "headline", "sentence"])
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)

    def process_label(example):
        example['label'] = example['label'] - 1
        return example

    ds = ds.map(process_label, num_proc=8)

    slct = [i for i in range(len(ds['test']))]
    smp = random.sample(slct, 3000)
    ds['test'] = ds['test'].select(smp)

    for split in ['train', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    random.seed(42)

    slct = [i for i in range(len(ds['train']))]
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/agnews"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/agnews"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_dbpedia():
    from datasets import load_dataset
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/TextClassificationDatasets/dbpedia_csv_small/train_30000.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/TextClassificationDatasets/dbpedia_csv_small/test.csv"},
                           column_names=["label", "headline", "sentence"])
    print(ds)

    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)

    def process_label(example):
        example['label'] = example['label'] - 1
        return example

    ds = ds.map(process_label, num_proc=8)

    slct = [i for i in range(len(ds['test']))]
    smp = random.sample(slct, 3000)
    ds['test'] = ds['test'].select(smp)

    for split in ['train', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    random.seed(42)

    slct = [i for i in range(len(ds['train']))]
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/dbpedia"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/dbpedia"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_yahoo():
    from datasets import load_dataset
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/TextClassificationDatasets/yahoo_answers_csv_small/train_30000.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/cls_data/original/TextClassificationDatasets/yahoo_answers_csv_small/test.csv"},
                           column_names=["label", "title", "content", "sentence"])
    print(ds)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
    ds = ds.filter(lambda x: x['sentence'] is not None and x['title'] is not None)
    def process_label(example):
        example['label'] = example['label'] - 1
        example['sentence'] = example['title'] + " Answer: " + example['sentence']
        example['len_sentence'] = len(tknz(example['sentence']).input_ids)
        return example

    ds = ds.map(process_label, num_proc=8)
    ds = ds.filter(lambda x: x['len_sentence'] < 900)
    slct = [i for i in range(len(ds['test']))]
    smp = random.sample(slct, 3000)
    ds['test'] = ds['test'].select(smp)

    for split in ['train', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    random.seed(42)

    slct = [i for i in range(len(ds['train']))]
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/yahoo"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/yahoo"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_cola():
    from datasets import load_dataset
    ds = load_dataset("linxinyuan/cola")
    ds = ds.rename_column("text", "sentence")
    print(ds)
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['sentence'] not in x_set:
                x_set.add(e['sentence'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)


    for split in ['train', 'test']:
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/cola"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/cola"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", type=str, default=None)
    args = parser.parse_args()
    method_name = "get_" + args.d

    symbol_table = globals()

    # get the function from the symbol table
    func = symbol_table.get(method_name)
    # call the function
    func()
