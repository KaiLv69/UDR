from datasets import load_dataset
from typing import Any, Dict, Iterable
from transformers import AutoTokenizer
import torch
import pandas as pd
import numpy as np
import json
import random
import re
from src.dataset_readers.inference_tasks import InferenceTask
from src.utils.cache_util import get_cache_path
import more_itertools



class FewShotDatasetReader(torch.utils.data.Dataset):

    def __init__(self, model_name,task_name,prompt_file,ds_size=None, num_prompts=-1,gen=True,prompt_id=-1,n_tokens=1600,data_num=-1,order="ascending",template_idx=-1) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.tokenizer.pad_token = "<|endoftext|>"
        if template_idx != -1:
            self.task = InferenceTask.from_name(task_name)(prompt_file, self.tokenizer,ds_size,template_idx=template_idx)
        else:
            self.task = InferenceTask.from_name(task_name)(prompt_file, self.tokenizer,ds_size)
        self.num_prompts = num_prompts
        self.prompt_id = prompt_id
        if model_name == "gpt2-xl":
            self.n_tokens_in_prompt = 750
        else:
            self.n_tokens_in_prompt = n_tokens
        self.num_processes = 1
        self.process_index =  0
        self.gen = gen
        self.data_num = data_num
        self.order = order
        # self.cache_path = get_cache_path(self.task.hf_dataset)
        # self.length_cache = SqliteDict(self.cache_path, autocommit=True)
         
        
        
    def __getitem__(self, index):
        if self.gen:
            return self.text_to_instance(self.task.prompts[index], index=index)
        else:
            return self.cls_text_to_instance(self.task.prompts[index], index=index)

    def __len__(self):
        if self.data_num > 0:
            return min(self.data_num, len(self.task.prompts))
        else:
            return len(self.task.prompts)

    def get_length(self, text):
        tokenized_example = self.tokenizer.encode_plus(text,truncation=False,return_tensors='pt')
        shape = tokenized_example.input_ids.squeeze().shape
        if len(shape)==0:
            return 1
        else:
            return int(shape[0])
    
    def shard(self,accelerator):
            # self.task.hf_dataset = self.task.hf_dataset.shard(num_shards=accelerator.num_processes,index=accelerator.process_index)
            self.num_processes = accelerator.num_processes
            self.process_index =  accelerator.process_index
            self.task.prompts = list(more_itertools.distribute(accelerator.num_processes,self.task.prompts)[accelerator.process_index])


    def text_to_instance(self, entry: Dict[str, Any],index=-1):
        
        question, answer, prompts_list, lengths_list,idx_list = self.task.get_fields(entry)    
        q_length = self.get_length(question)
        max_prompts = np.searchsorted(np.cumsum(lengths_list), self.n_tokens_in_prompt - q_length)
        # if self.task.name == "cnndailymail":
        #     max_prompts = np.searchsorted(np.cumsum(lengths_list), self.n_tokens_in_prompt - q_length - 100)
        # else:
        #     max_prompts = np.searchsorted(np.cumsum(lengths_list),self.n_tokens_in_prompt-q_length-50)
        if self.num_prompts>0:
            max_prompts = min(self.num_prompts, max_prompts)
        if self.prompt_id> -1 and self.num_prompts==1:
            trunc_prompts_list = [prompts_list[::-1][self.prompt_id]]
        else:
            trunc_prompts_list = prompts_list[:max_prompts][::-1]

        if self.order == "descending":
            trunc_prompts_list = trunc_prompts_list[::-1]
        elif self.order == "ascending":
            trunc_prompts_list = trunc_prompts_list
        elif self.order == "random":
            random.shuffle(trunc_prompts_list)
        else:
            raise NotImplementedError

        prompt_enc_text = "\n".join(trunc_prompts_list)
        
        
        enc_text = self.task.postproccess(f"{prompt_enc_text}\n{question}\t{self.task.postfix}")
        tokenized_example = self.tokenizer.encode_plus(enc_text,truncation=False,return_tensors='pt',add_special_tokens=False)

        
        entry['id'] = self.num_processes*(self.process_index) + index
        entry['prompt_list'] = prompts_list
        entry['enc_text'] = enc_text
        entry['len_trunc_prompts_list'] = len(trunc_prompts_list)

        return {
                        'input_ids': tokenized_example.input_ids.squeeze(),
                        'attention_mask': tokenized_example.attention_mask.squeeze(),
                        "metadata": entry,
                        
                    }

    def cls_text_to_instance(self, entry: Dict[str, Any], index=-1):
        sentence, true_label, prompts_list, lengths_list, test_label = self.task.get_fields(entry)
        q_length = self.get_length(sentence)
        max_prompts = np.searchsorted(np.cumsum(lengths_list), self.n_tokens_in_prompt - q_length)
        if self.num_prompts > 0:
            max_prompts = min(self.num_prompts, max_prompts)
        if self.prompt_id > -1 and self.num_prompts == 1:
            trunc_prompts_list = [prompts_list[::-1][self.prompt_id]]
        else:
            trunc_prompts_list = prompts_list[:max_prompts][::-1]

        if self.order == "descending":
            trunc_prompts_list = trunc_prompts_list[::-1]
        elif self.order == "ascending":
            trunc_prompts_list = trunc_prompts_list
        elif self.order == "random":
            random.shuffle(trunc_prompts_list)
        else:
            raise NotImplementedError

        prompt_enc_text = "\n".join(trunc_prompts_list)
        enc_text = self.task.postproccess(f"{prompt_enc_text}\n{sentence}\t{test_label}")

        tokenized_example = self.tokenizer.encode_plus(enc_text, truncation=True, add_special_tokens=False,
                                                       return_tensors='pt')
        tokenized_labels = self.tokenizer.encode_plus(test_label, truncation=True, add_special_tokens=False,
                                                      return_tensors='pt')

        input_ids = tokenized_example.input_ids.squeeze(0)
        labels = tokenized_labels.attention_mask.squeeze(0)

        pad_mask = torch.nn.functional.pad(labels, (input_ids.shape[-1] - labels.shape[-1] - 1, 0), value=0)
        entry['true_label'] = true_label
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pad_mask': pad_mask,
            "metadata": entry
        }
