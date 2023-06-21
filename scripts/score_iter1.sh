main_process_port=$((RANDOM % 5001 + 25000))
cvd=0,1,2,3,4,5,6,7
num_gpus=8
score_batch_size=10
exp_name="bs128_grad-acc4_rk4_seed1208"

datasets_full=("agnews" "amazon" "break" "cola" "common_gen" \
"copa" "cosmos_qa" "cr" "cs_explan" "cs_valid" "dart" "dbpedia" \
"e2e" "mr" "mtop" "pubmed" "reddit" "roc_ending_generation" "roc_story_generation" \
"rte" "smcalflow" "sst2" "sst5" "subj" "trec" "yahoo" "yelp_full")
datasets_sampled=("cnndailymail" "go" "java" "mnli" "php" "python" "snli" "wikiauto")

for train_set in "train" "debug"; do
  if [ "$train_set" == "train" ]; then
    datasets=("${datasets_full[@]}")
  else
    datasets=("${datasets_sampled[@]}")
  fi

  for dataset in "${datasets[@]}"; do
    echo -e "\n\n-------format conversion ${dataset}-------\n\n"
    in_scored_file="$PWD/exps/$exp_name/data/iter0/${dataset}_scoredqa_merged.json"
    input_dr_file="$PWD/exps/$exp_name/data/iter1/dr_data_${dataset}_${train_set}"
    out_dep_file="$PWD/exps/$exp_name/data/iter1/score_dep_file_${dataset}_${train_set}"
    output_file="$PWD/exps/$exp_name/data/iter1/dr_converted_${dataset}_${train_set}"

    if [ ! -f "$out_dep_file" ]; then
    python src/utils/convert_format.py \
      --dataset $dataset \
      --func "convert" \
      --num "all" \
      --input_file ${input_dr_file} \
      --input_scored_file "$in_scored_file" \
      --output_file "${output_file}" \
      --output_dep_file "${out_dep_file}" \
      --split "${train_set}"
    fi

    echo -e "\n\n-------score ${dataset}-------\n\n"
    scorer_py_output_path="$PWD/exps/$exp_name/data/iter1/${dataset}_scoredqa.json"
    if [ ! -f "$scorer_py_output_path" ]; then
    CUDA_VISIBLE_DEVICES=$cvd \
    accelerate launch --num_processes $num_gpus --main_process_port ${main_process_port} --multi_gpu \
      scorer.py \
      example_file=${output_file} \
      setup_type=qa \
      output_file="$scorer_py_output_path" \
      batch_size=${score_batch_size} \
      +task_name=$dataset +dataset_reader.ds_size=null \
      hydra.run.dir="$PWD/exps/$exp_name/iter1/logs"
    fi

    echo -e "\n\n-------merge ${dataset}-------\n\n"
    merge_out_path="$PWD/exps/$exp_name/data/iter1/${dataset}_scoredqa_merged.json"
    if [ ! -f "$merge_out_path" ]; then
    python src/utils/convert_format.py \
      --dataset $dataset \
      --func "merge" \
      --scored_file1 "$out_dep_file" \
      --scored_file2 "$scorer_py_output_path" \
      --output_file "$merge_out_path"
    fi
  done
done