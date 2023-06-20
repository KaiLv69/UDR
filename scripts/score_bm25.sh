main_process_port=$((RANDOM % 5001 + 25000))
cvd=0,1,2,3,4,5,6,7
num_gpus=8

datasets_full=("agnews" "amazon" "break" "cola" "common_gen" \
"copa" "cosmos_qa" "cr" "cs_explan" "cs_valid" "dart" "dbpedia" \
"e2e" "mr" "mtop" "pubmed" "reddit" "roc_ending_generation" "roc_story_generation" \
"rte" "smcalflow" "sst2" "sst5" "subj" "trec" "yahoo" "yelp_full")
datasets_sampled=("cnndailymail" "go" "java" "mnli" "php" "python" "snli" "wikiauto")

if [ ! -d "$PWD/data_score" ]; then
  mkdir "$PWD/data_score"
fi

for train_set in "train" "debug"; do
  if [ "$train_set" == "train" ]; then
    datasets=("${datasets_full[@]}")
  else
    datasets=("${datasets_sampled[@]}")
  fi

  for dataset in "${datasets[@]}"; do
    echo -e "\n\n-------score ${dataset}-------\n\n"
    scorer_py_output_path="$PWD/data_score/${dataset}_bm25.json"
    if [ ! -f "$scorer_py_output_path" ]; then
    CUDA_VISIBLE_DEVICES=$cvd \
    accelerate launch --num_processes $num_gpus --main_process_port ${main_process_port} --multi_gpu \
      scorer.py \
      example_file="$PWD/data_bm25/${dataset}_${train_set}.json" \
      setup_type=qa \
      output_file="$scorer_py_output_path" \
      batch_size=20 \
      +task_name=$dataset +dataset_reader.ds_size=null \
      hydra.run.dir="$PWD/exps/score_bm25/${dataset}/logs"
    fi
  done
done