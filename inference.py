import warnings
import torch
import tqdm
from allennlp.nn.util import sequence_cross_entropy_with_logits
from torch.utils.data import DataLoader
from src.data.collators import DataCollatorWithPaddingAndCuda
import hydra.utils as hu
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import json
import os
from src.utils.cache_util import BufferedJsonWriter, BufferedJsonReader
from accelerate import Accelerator, DistributedType
import transformers
from src.utils import eval_datasets
import re
from omegaconf import OmegaConf
import glob


# from src.dataset_readers.tasks import Task
# from src.dataset_readers.few_shot_dsr.FewShotDatasetReader import
def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)


def renorm(text):
    text = text.split("\n")[0]
    text = re.sub("[\d]+\#\) ", ";", text)
    return text


class Inferencer:
    def __init__(self, cfg, accelerator) -> None:
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.dataset_reader.shard(accelerator)
        # print(len(self.dataset_reader.task.prompts))
        self.dataset_reader.tokenizer.pad_token_id = [self.dataset_reader.tokenizer.eos_token_id]
        if cfg.gen:
            self.dataset_reader.tokenizer.padding_side = "left"
        else:
            self.dataset_reader.tokenizer.padding_side = "right"
        self.accelerator = accelerator
        # co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer,device = 0 if self.accelerator.device is None else None)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=accelerator.device)

        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)
        # import pdb
        # pdb.set_trace()
        self.model = hu.instantiate(cfg.model)
        self.model = self.model.to(self.accelerator.device)
        self.model = self.model.eval().half()
        # self.model, self.dataloader = self.accelerator.prepare(
        #                     self.model, self.dataloader
        # )
        if hasattr(self.model, "module"):
            self.model = self.model.module

        self.output_file = cfg.output_file
        self.cfg = cfg
        if cfg.model != "gpt2-xl":
            self.max_length = cfg.max_length
        else:
            self.max_length = 900

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        with BufferedJsonWriter(f"{self.output_file}tmp_{self.accelerator.device}.bin") as buffer:
            for i, entry in enumerate(dataloader):
                if "stop" in self.cfg and self.cfg.stop and i == 3:
                    break
                metadata = entry.pop("metadata")
                with torch.no_grad():
                    # entry.input_ids = entry.input_ids.half()
                    # entry.attention_mask = entry.attention_mask.half()
                    if self.cfg.model != "gpt2-xl":
                        res = self.model.generate(input_ids=entry.input_ids,
                                                  attention_mask=entry.attention_mask,
                                                  eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
                                                  pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                                                  max_length=self.max_length,
                                                  do_sample=False)
                    elif self.cfg.model == "gpt2-xl":
                        res = self.model.generate(input_ids=entry.input_ids[:,:500],
                                                  attention_mask=entry.attention_mask[:,:500],
                                                  # eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
                                                  # pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                                                  max_length=self.max_length,
                                                  do_sample=False)
                    # inp_length_list = entry.attention_mask.sum(-1).squeeze().tolist()
                    a = int(entry.attention_mask.shape[1])
                    for mdata, res_el in zip(metadata, res.tolist()):
                        mdata['generated'] = self.dataset_reader.tokenizer.decode(res_el[a:])
                        buffer.write(mdata)

    def cls_forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        with BufferedJsonWriter(f"{self.output_file}tmp_{self.accelerator.device}.bin") as buffer:
            for i, entry in enumerate(dataloader):
                if "stop" in self.cfg and self.cfg.stop and i == 3:
                    break

                metadata = entry.pop("metadata")
                with torch.no_grad():
                    # entry.input_ids = entry.input_ids.half()
                    # entry.attention_mask = entry.attention_mask.half()
                    output = self.model(input_ids=entry.input_ids, attention_mask=entry.attention_mask)
                    pad_mask = entry.pad_mask
                    loss_list = sequence_cross_entropy_with_logits(logits=output.logits[:, :-1].contiguous(),
                                                                   targets=entry.input_ids[:, 1:].contiguous(),
                                                                   weights=pad_mask,
                                                                   average=None)
                    if len(loss_list.shape) == 0:
                        loss_list = loss_list.unsqueeze(0)
                    for mdata, loss in zip(metadata, loss_list):
                        mdata['loss'] = float(loss.item())
                    for m in metadata:
                        buffer.write(m)


    def write_predictions(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with BufferedJsonReader(path) as f:
                data.extend(f.read())
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)
        # TODO
        # zipped_data = [[entry['question'],renorm(entry['generated']).split("\n")[0],entry['decomposition']] for entry in data]
        # question,pred,gold = list(zip(*zipped_data))
        # acc_results = eval_many(question,pred,gold)
        # for entry,acc_res in zip(data,acc_results):
        #     entry['correct'] =

        with open(self.output_file, "w") as f:
            json.dump(data, f)
        # 移到 tmp_test.py 里评测
        # data = eval_datasets.app[eval_datasets.get_dataset(self.output_file)](self.output_file)
        # with open(self.output_file,"w") as f:
        #     json.dump(data,f)

        return data


@hydra.main(config_path="configs", config_name="inference")
def main(cfg):
    print(cfg)

    with open("cfg_inference.json", "w") as f:
        json.dump(OmegaConf.to_object(cfg), f)
    accelerator = Accelerator()
    inferencer = Inferencer(cfg, accelerator)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg.gen:
            inferencer.forward()
        else:
            inferencer.cls_forward()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            inferencer.write_predictions()


if __name__ == "__main__":
    main()
