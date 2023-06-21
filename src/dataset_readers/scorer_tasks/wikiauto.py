from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import re
import json
from src.utils.dataset_utils import load_train_dataset


class WikiautoScorerTask:
    name = "wikiauto"
    question_field = "source"
    prompt_field = "ctxs"

    def __init__(self, example_file, ds_size=None) -> None:
        # dataset = load_dataset('GEM/wiki_auto_asset_turk', 'train')
        # dataset = dataset.filter(lambda x: len(x['target']) < 1000)
        # # add idx column
        # for split in ['train', 'validation', 'test_asset', 'test_turk', "test_wiki"]:
        #     ds_id = Dataset.from_dict({"idx": list(range(len(dataset[split])))})
        #     dataset[split] = concatenate_datasets([dataset[split], ds_id], axis=1)
        dataset = load_dataset("KaiLv/UDR_WikiAuto")

        self.hf_dataset = load_train_dataset(dataset, size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        self.postfix = "Simplified text: "

    def get_fields(self, entry, index=-1):
        question_prefix = "Simplify the text: "
        answer_prefix = "Simplified text: "
        test_question = question_prefix + entry['test_source']
        question = question_prefix + entry['source']
        decomposition = answer_prefix + entry['target']
        test_decomposition = entry['test_target']
        return question, decomposition, test_question, test_decomposition
