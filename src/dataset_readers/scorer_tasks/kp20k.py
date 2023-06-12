from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import re
import json
from src.utils.dataset_utils import load_train_dataset


class KP20kScorerTask:
    name = "kp20k"
    question_field = "document"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        # dataset = load_dataset("midas/kp20k", "generation")
        # # 去掉extractive_keyphrases为空的数据 去掉包含<eos>的数据
        # dataset = dataset.filter(lambda x: len(x['extractive_keyphrases']) > 0 and len(x['document']) + 4 * len(x['extractive_keyphrases']) < 1000)
        #
        # def process_kp20k(example):
        #     example['extractive_keyphrases'] = str(example['extractive_keyphrases']).replace("'", '"')
        #     example['abstractive_keyphrases'] = str(example['abstractive_keyphrases']).replace("'", '"')
        #     example['document'] = " ".join(example['document'])
        #     return example
        #
        # dataset = dataset.map(process_kp20k, num_proc=8)
        # # add idx column
        # dataset = dataset.remove_columns("id")
        # for split in ['train', 'validation', 'test']:
        #     ds_id = Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        #     dataset[split] = concatenate_datasets([dataset[split], ds_id], axis=1)
        dataset = load_from_disk("/remote-home/klv/exps/rtv_icl/data/kp20k")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)

    def get_fields(self, entry, index=-1):
        test_question = entry['test_document']
        question = entry['document']
        # decomposition = entry['abstractive_keyphrases']
        # test_decomposition = entry['test_abstractive_keyphrases']
        decomposition = entry['extractive_keyphrases']
        test_decomposition = entry['test_extractive_keyphrases']
        return question, decomposition, test_question, test_decomposition
