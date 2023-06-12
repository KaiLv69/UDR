export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
cvd=0,1,3,4,5,6,7
num_gpus=7
dataset="cnndailymail"
main_process_port=23990
exp_name="unified_bs128_1219"
sc_path="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/${dataset}"

mkdir -p "${sc_path}"
mkdir -p "${sc_path}/logs"

split="debug"
epoch=0
step=77041
generate_embedding_batch_size=4096
ckpt="$PWD/exps/$exp_name/model_ckpt_iter0_all_opt/dpr_biencoder.${epoch}_${step}"

echo -e "\n\n-generate dense embeddings-\n\n"
generate_dense_embeddings_py_output_path="$PWD/exps/${exp_name}/dr_data/dpr_index_${dataset}_${epoch}_${step}"
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
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

echo -e "\nbash run \n python DPR/dense_retriever.py --split $split\n"
dr_out="${sc_path}/dr_${dataset}_${split}_${epoch}_${step}"
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
     n_docs=50 \
     ctx_sources.dpr_epr.setup_type=qa \
     ctx_sources.dpr_epr.task_name=$dataset \
     datasets.qa_epr.task_name=$dataset \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

in_scored_file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter0/${dataset}/${dataset}_${split}_scoredqa_merged_all.json"
out_file="${sc_path}/dr_converted_${dataset}_${split}"
out_dep_file="${sc_path}/dr_dep_${dataset}_${split}"
if [ ! -f "$out_file" ]; then
python src/utils/convert_format.py \
  --dataset $dataset \
  --func "convert" \
  --input_file "$dr_out" \
  --input_scored_file "$in_scored_file" \
  --output_file "${out_file}" \
  --output_dep_file "${out_dep_file}" \
  --split "${split}" \
  --num "all"
fi

scorer_py_output_path="${sc_path}/${dataset}_scored_qa.json"
echo -e "\n\n-score-\n\n"
if [ ! -f "${scorer_py_output_path}" ]; then
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=$cvd \
accelerate launch --num_processes $num_gpus --main_process_port ${main_process_port} --multi_gpu \
     scorer.py \
     example_file="${out_file}" \
     setup_type=qa \
     output_file="$scorer_py_output_path" \
     batch_size=8 \
     +task_name=$dataset +dataset_reader.ds_size=null \
     hydra.run.dir="/nvme/xnli/lk_code/exps/rtv_icl/scored_data/${dataset}/logs"
fi

echo -e "\n\n-------merge-------\n\n"
merge_out_path="${sc_path}/${dataset}_${split}_scoredqa_merged.json"
if [ ! -f "$merge_out_path" ]; then
python src/utils/convert_format.py \
  --dataset $dataset \
  --func "merge" \
  --scored_file1 "$out_dep_file" \
  --scored_file2 "$scorer_py_output_path" \
  --output_file "$merge_out_path"
fi