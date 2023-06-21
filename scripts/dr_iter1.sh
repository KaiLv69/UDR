datasets_full=("agnews" "amazon" "break" "cola" "common_gen" \
"copa" "cosmos_qa" "cr" "cs_explan" "cs_valid" "dart" "dbpedia" \
"e2e" "mr" "mtop" "pubmed" "reddit" "roc_ending_generation" "roc_story_generation" \
"rte" "smcalflow" "sst2" "sst5" "subj" "trec" "yahoo" "yelp_full")
datasets_sampled=("cnndailymail" "go" "java" "mnli" "php" "python" "snli" "wikiauto")
exp_name="bs128_grad-acc4_rk4_seed1208"
model_ckpt_file="$PWD/exps/$exp_name/iter1/model_ckpt/dpr_biencoder.9"
pretrained_model_cfg="bert-base-uncased"

generate_embedding_batch_size=2048
cvd=0,1,2,3,4,5,6,7

for train_set in "train" "debug"; do
  if [ "$train_set" == "train" ]; then
    datasets=("${datasets_full[@]}")
  else
    datasets=("${datasets_sampled[@]}")
  fi

  for dataset in "${datasets[@]}"; do
    echo -e "\n\n-------generate dense embeddings for ${dataset}-------\n\n"
    generate_dense_embeddings_py_output_path="$PWD/exps/$exp_name/data/iter1/${dataset}_dpr_index"
    if [ ! -f "${generate_dense_embeddings_py_output_path}_0" ]; then
    CUDA_VISIBLE_DEVICES=$cvd \
    python DPR/generate_dense_embeddings.py \
       model_file="$model_ckpt_file" \
       ctx_src=dpr_epr \
       shard_id=0 num_shards=1 \
       out_file=$generate_dense_embeddings_py_output_path \
       ctx_sources.dpr_epr.setup_type=qa \
       ctx_sources.dpr_epr.task_name=$dataset \
       +ctx_sources.dpr_epr.ds_size=null \
       batch_size=$generate_embedding_batch_size \
       encoder.pretrained_model_cfg="${pretrained_model_cfg}" \
       hydra.run.dir="$PWD/exps/$exp_name/dr/iter1/logs"
    fi

    echo -e "\n\n-------dense retriever-------\n\n"
    dr_output_path="$PWD/exps/$exp_name/data/iter1/dr_data_${dataset}_${train_set}"
    if [ ! -f "${dr_output_path}" ]; then
    CUDA_VISIBLE_DEVICES=$cvd \
    python DPR/dense_retriever.py \
      model_file="$model_ckpt_file" \
      qa_dataset=qa_epr \
      n_docs=50 \
      ctx_datatsets=[dpr_epr] \
      datasets.qa_epr.dataset_split="${train_set}" \
      encoded_ctx_files=["${generate_dense_embeddings_py_output_path}_*"] \
      out_file="${dr_output_path}" \
      ctx_sources.dpr_epr.setup_type=qa \
      ctx_sources.dpr_epr.task_name=$dataset \
      datasets.qa_epr.task_name=$dataset \
      encoder.pretrained_model_cfg="${pretrained_model_cfg}" \
      hydra.run.dir="$PWD/exps/$exp_name/dr/iter1/logs"
    fi
  done
done