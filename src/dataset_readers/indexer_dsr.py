from datasets import load_dataset
from typing import Any, Dict, Iterable
from transformers import AutoTokenizer
import torch
import pandas as pd
from src.dataset_readers.bm25_tasks import BM25Task
import tqdm

class IndexerDatasetReader(torch.utils.data.Dataset):

    def __init__(self, task_name, model_name, setup_type, dataset_split, instruction) -> None:
        if instruction is False:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.task = BM25Task.from_name(task_name)(dataset_split,
                                                        setup_type)
        self.instruction = instruction
        for i, row in tqdm.tqdm(enumerate(self.task.dataset)):
            self.task.dataset[i]['enc_text'] = self.task.get_field(row)

        
    def __getitem__(self, index):
        return self.text_to_instance(self.task.dataset[index],index=index)

    def __len__(self):
        return len(self.task.dataset)

    def text_to_instance(self, entry: Dict[str, Any],index=-1):
        enc_text = " ".join(entry['enc_text'])
        if self.instruction is True:
            return {
                "instruction": "Represent the example for retrieving duplicate examples; Input: ",
                "enc_text": enc_text,
                "metadata": {"id": index}
            }
        tokenized_inputs = self.tokenizer.encode_plus(enc_text,truncation=True,return_tensors='pt')
        return {
                        'input_ids': tokenized_inputs.input_ids.squeeze(),
                        'attention_mask': tokenized_inputs.attention_mask.squeeze(),
                        "metadata":{"id":index}
                        
                    }

        
