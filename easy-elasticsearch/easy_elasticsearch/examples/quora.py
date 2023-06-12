from easy_elasticsearch import ElasticSearchBM25
import os
import csv
import requests
import json
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["docker", "executable", "existing"],
    default="docker",
    help="What kind of ES service",
)
mode = parser.parse_args().mode


url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
dataset_path = "quora_duplicate_questions.tsv"
max_corpus_size = 100000


def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print(
            "Exception when trying to download {}. Response {}".format(
                url, req.status_code
            ),
            file=sys.stderr,
        )
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm.tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


# Download dataset if needed
if not os.path.exists(dataset_path):
    print("Download dataset")
    http_get(url, dataset_path)

# Get all unique sentences from the file
all_questions = {}
with open(dataset_path, encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        all_questions[row["qid1"]] = row["question1"]
        if len(all_questions) >= max_corpus_size:
            break

        all_questions[row["qid2"]] = row["question2"]
        if len(all_questions) >= max_corpus_size:
            break

qids = list(all_questions.keys())
questions = [all_questions[qid] for qid in qids]
print("|questions|:", len(questions))
print("|words|:", len([word for q in questions for word in q.split()]))

if mode == "docker":
    bm25 = ElasticSearchBM25(
        dict(zip(qids, questions)),
        port_http="9222",
        port_tcp="9333",
        service_type="docker",
    )
elif mode == "executable":
    bm25 = ElasticSearchBM25(
        dict(zip(qids, questions)),
        port_http="9222",
        port_tcp="9333",
        service_type="executable",
    )
else:
    # Or use an existing ES service:
    assert mode == "existing"
    bm25 = ElasticSearchBM25(
        dict(zip(qids, questions)),
        host="http://localhost",
        port_http="9200",
        port_tcp="9300",
    )

query = "What is Python?"
rank = bm25.query(query, topk=10)  # topk should be <= 10000
scores = bm25.score(query, document_ids=list(rank.keys()))
print("###query###:", query)
print("###rank###:", json.dumps(rank, indent=4))
print("###scores###:", json.dumps(scores, indent=4))

bm25.delete_index()  # delete the one-trial index named 'one_trial'
if mode == "docker":
    bm25.delete_container()
elif mode == "executable":
    bm25.delete_excutable()
