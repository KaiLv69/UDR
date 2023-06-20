# before run this script, you should:
# 1. download elasticsearch-7.9.1
# 2. run ES_JAVA_OPTS="-Xms31g -Xmx31g" ./elasticsearch-7.9.1/bin/elasticsearch to start elasticsearch
datasets_full=("agnews" "amazon" "break" "cola" "common_gen" \
"copa" "cosmos_qa" "cr" "cs_explan" "cs_valid" "dart" "dbpedia" \
"e2e" "mr" "mtop" "pubmed" "reddit" "roc_ending_generation" "roc_story_generation" \
"rte" "smcalflow" "sst2" "sst5" "subj" "trec" "yahoo" "yelp_full")
datasets_sampled=("cnndailymail" "go" "java" "mnli" "php" "python" "snli" "wikiauto")

if [ ! -d "$PWD/data_bm25" ]; then
  mkdir "$PWD/data_bm25"
fi

for train_set in "train" "debug"; do
  if [ "$train_set" == "train" ]; then
    datasets=("${datasets_full[@]}")
  else
    datasets=("${datasets_sampled[@]}")
  fi

  for dataset in "${datasets[@]}"; do
    find_bm25_py_output_path="$PWD/data_bm25/${dataset}_${train_set}.json"
    echo -e "\n\n-find_bm25 ${dataset}-\n\n"
    if [ ! -f "${find_bm25_py_output_path}" ]; then
      HYDRA_FULL_ERROR=1 \
      python find_bm25_es.py \
           output_path="$find_bm25_py_output_path" \
           dataset_split=${train_set} \
           setup_type="a" \
           task_name=${dataset} \
           +ds_size=null \
           L=50
           hydra.run.dir="$PWD/exps/find_bm25/${dataset}/logs"
    fi
  done
done