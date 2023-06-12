import os, random, datasets
from tqdm import tqdm


def get_commonsense_qa():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = load_dataset("commonsense_qa")
    print(ds)
    ds = ds.filter(lambda x: len(x['answerKey']))

    def process_commonsense_qa(example):
        ans = [l + ". " + t for l, t in zip(example['choices']['label'], example['choices']['text'])]
        ans = [t for l, t in zip(example['choices']['label'], example['choices']['text'])]

        # example['question'] = example['question'] + " " + " ".join(ans)
        example['question'] = example['question']
        label_text_dict = {l: t for l, t in zip(example['choices']['label'], example['choices']['text'])}
        example['label'] = example['answerKey'] + ". " + label_text_dict[example['answerKey']]
        example['label'] = label_text_dict[example['answerKey']]
        example['choices'] = "\n".join(ans)
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_commonsense_qa, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)
    ds = ds.remove_columns(['answerKey', 'id', 'question_concept'])
    for split in ['train', 'validation', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
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
        base_dir = "/remote-home/klv/exps/rtv_icl/data/commonsense_qa"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/commonsense_qa"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_cosmos_qa():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = load_dataset("cosmos_qa")
    print(ds)

    def process_cosmos_qa(example):
        # example['question'] = example['context'] + " " + example['question'] + " choices: "\
        #                       "A. " + example['answer0'] + \
        #                       " B. " + example['answer1'] + \
        #                       " C. " + example['answer2'] + \
        #                       " D. " + example['answer3']
        example['question'] = example['context'] + " " + example['question']
        label_text_dict = ["A. " + example['answer0'],
                           "B. " + example['answer1'],
                           "C. " + example['answer2'],
                           "D. " + example['answer3']]
        label_text_dict = [example['answer0'],
                           example['answer1'],
                           example['answer2'],
                           example['answer3']]

        example['label'] = label_text_dict[example['label']]
        example['choices'] = "\n".join(label_text_dict)
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict])
        return example

    ds = ds.map(process_cosmos_qa, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)

    for split in ['train', 'validation', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)
    ds = ds.remove_columns(['answer0', 'answer1', 'answer2', 'answer3', 'context', 'id'])
    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/cosmos_qa"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/cosmos_qa"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_arc_easy():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = load_dataset("ai2_arc", "ARC-Easy")
    print(ds)

    def process_cosmos_qa(example):
        # label_text_dict = {
        #     "A": "A. " + example['choices']['text'][0],
        #     "B": "B. " + example['choices']['text'][1],
        #     "C": "C. " + example['choices']['text'][2],
        #     "D": "D. " + example['choices']['text'][3]
        # }
        label_text_dict = {
            k: f"{v}" for k, v in zip(example['choices']['label'], example['choices']['text'])
        }
        example['question'] = example['question']

        example['label'] = label_text_dict[example['answerKey']]
        example['choices'] = "\n".join(label_text_dict.values())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_cosmos_qa, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)

    for split in ['train', 'validation', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)
    ds = ds.remove_columns(['answerKey', 'id'])

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/arc_easy"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/arc_easy"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_copa():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    ds = load_dataset("pkavumba/balanced-copa")
    ds = ds.filter(lambda x: x['mirrored'] is False)
    print(ds)

    def process_cosmos_qa(example):
        label_text_dict = {
            0: example['choice1'],
            1: example['choice2'],
        }

        if example['question'] == 'cause':
            example['question'] = example['premise'] + " What was the cause of this?"
        elif example['question'] == 'effect':
            example['question'] = example['premise'] + " What happened as a result?"
        else:
            raise ValueError

        example['label'] = label_text_dict[example['label']]
        example['choices'] = "\n".join(label_text_dict.values())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_cosmos_qa, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)

    for split in ['train', 'test']:
        # x_set = set([])
        # lst = []
        # for i, e in tqdm(enumerate(ds[split])):
        #     f = 0
        #     if e['question'] not in x_set:
        #         x_set.add(e['question'])
        #         f += 1
        #     if f == 1:
        #         lst.append(i)
        # ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)
    ds = ds.remove_columns(['choice1', 'choice2', 'id'])

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/copa"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/copa"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_balanced_copa():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    ds = load_dataset("pkavumba/balanced-copa")
    print(ds)

    def process_cosmos_qa(example):
        label_text_dict = {
            0: example['choice1'],
            1: example['choice2'],
        }

        if example['question'] == 'cause':
            example['question'] = example['premise'] + " What was the cause of this?"
        elif example['question'] == 'effect':
            example['question'] = example['premise'] + " What happened as a result?"
        else:
            raise ValueError

        example['label'] = label_text_dict[example['label']]
        example['choices'] = "\n".join(label_text_dict.values())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_cosmos_qa, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)

    for split in ['train', 'test']:
        # x_set = set([])
        # lst = []
        # for i, e in tqdm(enumerate(ds[split])):
        #     f = 0
        #     if e['question'] not in x_set:
        #         x_set.add(e['question'])
        #         f += 1
        #     if f == 1:
        #         lst.append(i)
        # ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)
    ds = ds.remove_columns(['choice1', 'choice2', 'id'])

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/balanced_copa"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/balanced_copa"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)

def get_social_i_qa():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = load_dataset("social_i_qa")
    print(ds)

    def process_cosmos_qa(example):
        example['question'] = example['context'] + " " + example['question'] + " choices: " \
                                                                               "A. " + example['answerA'] + \
                              " B. " + example['answerB'] + \
                              " C. " + example['answerC']
        label_text_dict = {"1": "A. " + example['answerA'],
                           "2": "B. " + example['answerB'],
                           "3": "C. " + example['answerC']}

        example['label'] = label_text_dict[example['label']]
        example['choices'] = "\n".join(label_text_dict.values())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_cosmos_qa, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)

    for split in ['train', 'validation']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)
    ds = ds.remove_columns(['answerA', 'answerB', 'answerC', 'context'])
    slct = [i for i in range(len(ds['train']))]
    random.seed(42)
    smp = random.sample(slct, 5000)
    ds['debug'] = ds['train'].select(smp)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/social_i_qa"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/social_i_qa"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_piqa():
    from datasets import load_dataset, DatasetDict
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = load_dataset("piqa")
    print(ds)
    ds = DatasetDict({'train': ds['train'], 'validation': ds['validation']})

    def process_cosmos_qa(example):
        example['question'] = example['goal'] + " Choices: " \
                                                "A. " + example['sol1'] + \
                              " B. " + example['sol2']
        example['question'] = example['goal']
        label_text_dict = {0: "A. " + example['sol1'],
                           1: "B. " + example['sol2'], }
        label_text_dict = {0: example['sol1'],
                           1: example['sol2'], }

        example['label_t'] = label_text_dict[example['label']]
        example['choices'] = "\n".join(label_text_dict.values())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_cosmos_qa, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)
    ds = ds.remove_columns(['sol1', 'sol2', 'goal', 'label'])
    ds = ds.rename_column('label_t', 'label')
    for split in ['train', 'validation']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
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
        base_dir = "/remote-home/klv/exps/rtv_icl/data/piqa"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/piqa"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_hellaswag():
    from datasets import load_dataset, DatasetDict
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = load_dataset("hellaswag")
    print(ds)
    ds = DatasetDict({'train': ds['train'], 'validation': ds['validation']})

    def process_cosmos_qa(example):
        example['question'] = example['ctx']

        label_text_dict = {'0': example['endings'][0],
                           '1': example['endings'][1],
                           '2': example['endings'][2],
                           '3': example['endings'][3], }

        example['label'] = label_text_dict[example['label']]
        example['choices'] = "\n".join(label_text_dict.values())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_cosmos_qa, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)
    ds = ds.remove_columns(
        ['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type'])
    for split in ['train', 'validation']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
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
        base_dir = "/remote-home/klv/exps/rtv_icl/data/hellaswag"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/hellaswag"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_openbookqa():
    from datasets import load_dataset, DatasetDict
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = load_dataset("openbookqa")
    print(ds)

    def process_cosmos_qa(example):
        example['question'] = example['question_stem']

        label_text_dict = {'A': example['choices']['text'][0],
                           'B': example['choices']['text'][1],
                           'C': example['choices']['text'][2],
                           'D': example['choices']['text'][3], }

        example['label'] = label_text_dict[example['answerKey']]
        example['choices'] = "\n".join(label_text_dict.values())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_cosmos_qa, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)
    ds = ds.remove_columns(['id', 'question_stem', 'answerKey'])
    for split in ['train', 'validation', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
                f += 1
            if f == 1:
                lst.append(i)
        ds[split] = ds[split].select(lst)
        # add column idx
        ds_id = datasets.Dataset.from_dict({"idx": list(range(len(ds[split])))})
        ds[split] = datasets.concatenate_datasets([ds_id, ds[split]], axis=1)

    if os.path.exists("/remote-home/klv/exps/rtv_icl/data"):
        base_dir = "/remote-home/klv/exps/rtv_icl/data/openbookqa"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/openbookqa"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_cs_explan():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/comve/SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALLdata/TrainingData/subtaskB_data_all.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/comve/SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALLdata/TestData/subtaskB_test_data.csv"
    },
                      delimiter=",", )
    ds_label = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/comve/SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALLdata/TrainingData/subtaskB_answers_all.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/comve/SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALLdata/TestData/subtaskB_gold_answers.csv"
    },
                            column_names=["id", "label"], delimiter=",")
    ds_label = ds_label.remove_columns(["id"])
    for split in ['train', 'test']:
        ds[split] = datasets.concatenate_datasets([ds[split], ds_label[split]], axis=1)
    ds = ds.filter(lambda x: x['FalseSent'] is not None and x['OptionC'] is not None)
    print(ds)

    def process_cs_explan(example):
        example['question'] = "Select the most corresponding reason why this statement is against common sense. " + \
                              example['FalseSent'] + \
                              " Options: A. " + example['OptionA'] + " B. " + example['OptionB'] + " C. " + example[
                                  'OptionC']
        label_text_dict = {'A': example['OptionA'], 'B': example['OptionB'], 'C': example['OptionC']}
        example['label'] = example['label'] + ". " + label_text_dict[example['label']]
        example['choices'] = "A. " + example['OptionA'] + \
                             "\nB. " + example['OptionB'] + \
                             "\nC. " + example['OptionC']
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_cs_explan, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)
    ds = ds.remove_columns(['id', 'FalseSent', 'OptionA', 'OptionB', 'OptionC'])
    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
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
        base_dir = "/remote-home/klv/exps/rtv_icl/data/cs_explan"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/cs_explan"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_cs_valid():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/comve/SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALLdata/TrainingData/subtaskA_data_all.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/comve/SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALLdata/TestData/subtaskA_test_data.csv"
    },
                      delimiter=",", )
    ds_label = load_dataset("csv", data_files={
        "train": "/nvme/xnli/lk_code/exps/rtv_icl/data/comve/SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALLdata/TrainingData/subtaskA_answers_all.csv",
        "test": "/nvme/xnli/lk_code/exps/rtv_icl/data/comve/SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALLdata/TestData/subtaskA_gold_answers.csv"
    },
                            column_names=["id", "label"], delimiter=",")
    ds_label = ds_label.remove_columns(["id"])
    for split in ['train', 'test']:
        ds[split] = datasets.concatenate_datasets([ds[split], ds_label[split]], axis=1)
    # ds = ds.filter(lambda x: x['FalseSent'] is not None and x['OptionC'] is not None)
    print(ds)

    def process_cs_explan(example):
        example['question'] = "Which statement of the two is against common sense?" + \
                              " Statement 1: " + example['sent0'] + \
                              " Statement 2: " + example['sent1']
        label_text_dict = {0: "Statement 1: " + example['sent0'], 1: "Statement 2: " + example['sent1']}
        # example['question'] = "Which statement of the two is consistent with common sense?" + \
        #                       " A: " + example['sent0'] + \
        #                       " B: " + example['sent1']
        # label_text_dict = {0: "B: "+ example['sent1'], 1: "A: "+ example['sent0']}
        # label_text_dict = {0: "A: "+ example['sent0'], 1: "B: "+ example['sent1']}

        example['label'] = label_text_dict[example['label']]

        example['choices'] = "\n".join(label_text_dict.values())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_cs_explan, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)
    ds = ds.remove_columns(['id', 'sent0', 'sent1'])
    for split in ['train', 'test']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
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
        base_dir = "/remote-home/klv/exps/rtv_icl/data/cs_valid"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/cs_valid"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(ds)
    ds.save_to_disk(base_dir)


def get_race():
    from datasets import load_dataset, DatasetDict
    from transformers import AutoTokenizer
    tknz = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    ds = load_dataset("race", "all")
    print(ds)

    def process_cosmos_qa(example):
        label_text_dict = {
            "A": "A. " + example['options'][0],
            "B": "B. " + example['options'][1],
            "C": "C. " + example['options'][2],
            "D": "D. " + example['options'][3]
        }
        example['question'] = example['article'] + " " + example['question'] + " choices: " \
                                                                               "A. " + example['options'][0] + \
                              " B. " + example['options'][1] + \
                              " C. " + example['options'][2] + \
                              " D. " + example['options'][3]

        example['label'] = label_text_dict[example['answer']]
        example['choices'] = "\n".join(label_text_dict.values())
        example['len_question'] = len(tknz(example['question']).input_ids)
        example['max_len_choices'] = max([len(tknz(c).input_ids) for c in label_text_dict.values()])
        return example

    ds = ds.map(process_cosmos_qa, num_proc=8)
    ds = ds.filter(lambda x: x['len_question'] + x['max_len_choices'] < 900)
    ds = ds.remove_columns(['example_id', 'article', 'answer', 'options'])
    for split in ['train', 'validation']:
        x_set = set([])
        lst = []
        for i, e in tqdm(enumerate(ds[split])):
            f = 0
            if e['question'] not in x_set:
                x_set.add(e['question'])
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
        base_dir = "/remote-home/klv/exps/rtv_icl/data/race"
    else:
        base_dir = "/nvme/xnli/lk_code/exps/rtv_icl/data/race"
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
