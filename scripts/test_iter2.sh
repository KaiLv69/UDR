exp_name="bs128_grad-acc4_rk4_seed1208"
cvd=6
num_gpus=1
cls_test_datasets=("agnews" "amazon" "cola" "copa" "cosmos_qa" "cr" "cs_explan" "cs_valid" "dbpedia" "mr" "snli" "sst2" \
"sst5" "subj" "trec" "yahoo" "yelp_full")
cls_valid_datasets=("qnli" "rte" "wnli")
gen_test_datasets=("cnndailymail" "dart" "e2e" "go" "java" "javascript" "mtop" "php" "pubmed" "python" "reddit" \
"roc_ending_generation" "roc_story_generation")
gen_valid_datasets=("break" "common_gen" "smcalflow")

main_process_port=$((RANDOM % 5001 + 25000))
ckpt="$PWD/exps/$exp_name/iter2/model_ckpt/dpr_biencoder.9"
mkdir -p "$PWD/exps/$exp_name/data/iter2"

generate_embedding_batch_size=2048

for ds in "${cls_valid_datasets[*]}" "${cls_test_datasets[*]}" "${gen_valid_datasets[*]}" "${gen_test_datasets[*]}" "wikiauto" "mnli"; do

  if [[ ${ds} == "${cls_valid_datasets[*]}" ]]; then
    echo "cls valid datasets"
    splits=("validation")
    gen=False
    num_prompts=8
    inf_bs=15
  elif [[ ${ds} == "${cls_test_datasets[*]}" ]]; then
    echo "cls test datasets"
    splits=("test")
    gen=False
    num_prompts=8
    inf_bs=15
  elif [[ ${ds} == "${gen_valid_datasets[*]}" ]]; then
    echo "gen valid datasets"
    splits=("validation")
    gen=True
    num_prompts=-1
    inf_bs=8
  elif [[ ${ds} == "${gen_test_datasets[*]}" ]]; then
    echo "gen test datasets"
    splits=("test")
    gen=True
    num_prompts=-1
    inf_bs=8
  elif [[ ${ds} == "wikiauto" ]]; then
    echo "wikiauto test"
    splits=("test_asset" "test_turk" "test_wiki")
    gen=True
    num_prompts=-1
    inf_bs=8
  elif [[ ${ds} == "mnli" ]]; then
    echo "mnli test"
    splits=("validation" "validation_mm")
    gen=False
    num_prompts=8
    inf_bs=8
  fi

  datasets=(${ds})
  for dataset in "${datasets[@]}"; do

    echo -e "\n\n-testing-\n\n"
    echo -e "\nbash run \n python DPR/generate_dense_embeddings.py\n"
    generate_dense_embeddings_py_output_path="$PWD/exps/$exp_name/data/iter2/${dataset}_dpr_index"
    if [ ! -f "${generate_dense_embeddings_py_output_path}_0" ]; then
    CUDA_VISIBLE_DEVICES=$cvd \
    python DPR/generate_dense_embeddings.py \
         model_file="${ckpt}" \
         ctx_src=dpr_epr \
         shard_id=0 num_shards=1 \
         out_file=$generate_dense_embeddings_py_output_path \
         ctx_sources.dpr_epr.setup_type=qa \
         ctx_sources.dpr_epr.task_name=$dataset \
         +ctx_sources.dpr_epr.ds_size=null \
         batch_size=$generate_embedding_batch_size \
         hydra.run.dir="$PWD/exps/$exp_name/iter2/logs"
    fi

    for split in "${splits[@]}"; do
      echo -e "\nbash run \n python DPR/dense_retriever.py --split $split\n"
      dr_out="$PWD/exps/$exp_name/data/iter2/dr_data_${dataset}_${split}"
      if [ ! -f "$dr_out" ]; then
      TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES=$cvd \
      python DPR/dense_retriever.py \
           model_file="${ckpt}" \
           qa_dataset=qa_epr \
           ctx_datatsets=[dpr_epr] \
           datasets.qa_epr.dataset_split=$split \
           encoded_ctx_files=["${generate_dense_embeddings_py_output_path}_*"] \
           out_file="${dr_out}" \
           ctx_sources.dpr_epr.setup_type=qa \
           ctx_sources.dpr_epr.task_name=$dataset \
           datasets.qa_epr.task_name=$dataset \
           hydra.run.dir="$PWD/exps/$exp_name/logs"
      fi
      echo -e "bash run \n accelerate launch inference.py --split $split"

      inference_out="$PWD/exps/$exp_name/data/iter2/inference_${dataset}_${split}.json"
      if [ ! -f "$inference_out" ]; then
      CUDA_VISIBLE_DEVICES=$cvd \
      TOKENIZERS_PARALLELISM=false \
      HYDRA_FULL_ERROR=1 \
      accelerate launch --num_processes $num_gpus --main_process_port ${main_process_port} --multi_gpu\
           inference.py \
           prompt_file="${dr_out}" \
           task_name=$dataset \
           output_file="$inference_out" \
           gen="${gen}" \
           num_prompts=${num_prompts} \
           batch_size=${inf_bs} max_length=1950 \
           hydra.run.dir="$PWD/exps/$exp_name/logs"
      fi

      echo -e "bash run \n python tmp_test.py --split $split"
      if [ "${gen}" = "True" ]; then
      python gen_test.py --fp "${inference_out}" --dataset $dataset --split $split \
      --exp_name ${exp_name} --iter_num 2 --epoch_num 10 --method "udr" \
      --prompt_num ${num_prompts}
      else
      python cls_test.py --fp "${inference_out}" --dataset $dataset --split $split \
      --exp_name ${exp_name} --iter_num 2 --epoch_num 10 --method "udr" \
      --prompt_num ${num_prompts}
      fi
    done
  done
done
