export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
cvd=2,3
num_gpus=2
exp_name="zero_shot_epr"
main_process_port=23199
generate_embedding_batch_size=2048
num_prompts=8
gen=False
inf_bs=10
test_split=("validation")
ckpt="/nvme/xnli/lk_code/exps/rtv_icl/v6/exps/epr_mnli_1215_bert/model_ckpt/dpr_biencoder.29"
#ckpt="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/epr_amazon_0106/model_ckpt/dpr_biencoder.0_1560"
#ckpt="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/epr_go_0106/model_ckpt/dpr_biencoder.0_781"
for dataset in "wnli"; do
#for dataset in "javascript"; do

  mkdir -p "$PWD/exps/$exp_name/${dataset}/data"
  mkdir -p "$PWD/exps/$exp_name/${dataset}/logs"


  echo -e "\n\n-testing-\n\n"
  echo -e "\nbash run \n python DPR/generate_dense_embeddings.py\n"
  generate_dense_embeddings_py_output_path="$PWD/exps/$exp_name/${dataset}/dpr_index"
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
       hydra.run.dir="$PWD/exps/$exp_name/${dataset}/logs"
  fi

  for split in "${test_split[@]}"; do

  echo -e "\nbash run \n python DPR/dense_retriever.py --split $split\n"
  dr_out="$PWD/exps/$exp_name/${dataset}/data/epr_dr_data_${dataset}_${split}"
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
       hydra.run.dir="$PWD/exps/$exp_name/${dataset}/logs"
  fi
  echo -e "bash run \n accelerate launch inference.py --split $split"

  inference_out="$PWD/exps/$exp_name/${dataset}/data/epr_inference_${dataset}_${split}_${num_prompts}prompts.json"
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
       hydra.run.dir="$PWD/exps/$exp_name/${dataset}/logs"
  fi

  echo -e "bash run \n python tmp_test.py --split $split"

  echo -e "bash run \n python tmp_test.py --split $split"
  if [ "${gen}" = "True" ]; then
  python tmp_test.py --fp "${inference_out}" --dataset $dataset --split $split \
  --exp_name ${exp_name} --iter_num -1 --epoch_num 29 --method epr_zs \
  --prompt_num ${num_prompts}
  else
  python cls_test.py --fp "${inference_out}" --dataset $dataset --split $split \
  --exp_name ${exp_name} --iter_num -1 --epoch_num 29 --method epr_zs \
  --prompt_num ${num_prompts}
  fi
  done
done
