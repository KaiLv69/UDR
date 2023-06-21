# UDR: Unified Demonstration Retriever for In-Context Learning

Implementation of out ACL'23 paper [Unified Demonstration Retriever for In-Context Learning](https://arxiv.org/pdf/2305.04320.pdf).

In this paper, we propose Unified Demonstration Retriever (UDR), a single model to retrieve demonstrations for a wide range of tasks. 
To train UDR, we cast various tasks’ training signals into a unified listwise ranking formulation by language model’s feedback. 
Then we propose a multi-task listwise ranking training framework, with an iterative mining strategy to find high-quality candidates, which can help UDR fully incorporate various tasks’ signals.

## How to run the code
### Dependencies
We use Python `3.8.0` and PyTorch `1.7.1` with cuda `11.0`.
We recommend installing `allennlp` and `apex` first, then you can install other dependencies by running:
```shell
pip install -r requirements.txt
```

### Training
The iterative training process of UDR can be done by the following commands:
```shell
# Initialize candidates of each training example by BM25
bash scripts/find_bm25.sh
# Score initial candidates by language model
bash scripts/score_bm25.sh
# Train UDR with the initial candidates  (iteration 0)
bash scripts/train_iter0.sh

# Update candidates by new retriever got in the last step
bash scripts/dr_iter0.sh
# Score updated candidates by language model
bash scripts/score_iter0.sh
# Train UDR with the updated candidates (iteration 1)
bash scripts/train_iter1.sh

# Update candidates by new retriever got in the last step
bash scripts/dr_iter1.sh
# Score updated candidates by language model
bash scripts/score_iter1.sh
# Train UDR with the updated candidates (iteration 2)
bash scripts/train_iter2.sh
```

### Testing
To test UDR, we can run the following command:
```shell
bash scripts/test_iter2.sh
```

## Data
All datasets are uploaded to Huggingface and can be found [here](https://huggingface.co/KaiLv).

## Citation
If you find this repo useful, please cite the following paper:
```text
@article{li2023unified,
  title={Unified Demonstration Retriever for In-Context Learning},
  author={Li, Xiaonan and Lv, Kai and Yan, Hang and Lin, Tianyang and Zhu, Wei and Ni, Yuan and Xie, Guotong and Wang, Xiaoling and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2305.04320},
  year={2023}
}
```