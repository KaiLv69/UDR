# Easy Elasticsearch

This repository contains a high-level encapsulation for using [Elasticsearch](https://www.elastic.co/downloads/elasticsearch) with python in just a few lines.

## Installation
Via pip:
```bash
pip install easy-elasticsearch
```
Via git repo:
```bash
git clone https://github.com/kwang2049/easy-elasticsearch
pip install -e . 
```

## Usage
To utilize the elasticsearch service, one can select from 3 ways:
- (1) Start an ES service manually and then indicate the `host` and `port_http` (please refere to [download_and_run.sh](easy_elasticsearch/examples/download_and_run.sh)); 
- (2) Or leave `host=None` by default to start a docker container itself;
- (3) Or leava `host=None` and setting `service_type=executable` to download an ES executable and start it in the back end.

Finally, just either call its ```rank``` or ```score``` function for retrieval or calculating BM25 scores.
```python
from easy_elasticsearch import ElasticSearchBM25

pool = {
    'id1': 'What is Python? Is it a programming language',
    'id2': 'Which Python version is the best?',
    'id3': 'Using easy-elasticsearch in Python is really convenient!'
}
bm25 = ElasticSearchBM25(pool, port_http='9222', port_tcp='9333')  # By default, when `host=None` and `mode="docker"`, a ES docker container will be started at localhost.

query = "What is Python?"
rank = bm25.query(query, topk=10)  # topk should be <= 10000
scores = bm25.score(query, document_ids=['id2', 'id3'])

print(query, rank, scores)
bm25.delete_index()  # delete the one-trial index named 'one_trial'
bm25.delete_container()  # remove the docker container'
```
Another example for retrieving Quora questions can be found in [easy_elasticsearch/examples/quora.py](easy_elasticsearch/examples/quora.py):
```bash
python -m easy_elasticsearch.examples.quora  --mode docker
```
or
```bash
python -m easy_elasticsearch.examples.quora  --mode executable
```
