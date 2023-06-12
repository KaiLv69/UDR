export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
cvd=0,1,2,3
num_gpus=4
exp_name="unified_bs128_1219"
datasets=("agnews" "amazon" "cola" "copa" "cosmos_qa" "cr" "cs_explan" "cs_valid" "dbpedia" "mr" "rte" "snli" "sst2" \
"sst5" "subj" "trec" "yahoo" "yelp_full")
main_process_port=23100
num_prompts=-1
epoch=0
step=58056

mkdir -p "$PWD/exps/$exp_name/dr_data"
mkdir -p "$PWD/exps/$exp_name/inf_data"

generate_embedding_batch_size=2048
gen=False
split="test"

echo -e "\n\n-testing-\n\n"

for dataset in "${datasets[@]}"; do

echo -e "\n\n-testing-\n\n"
ckpt="$PWD/exps/$exp_name/model_ckpt/dpr_biencoder.${epoch}_${step}"
echo -e "\nbash run \n python DPR/generate_dense_embeddings.py\n"
generate_dense_embeddings_py_output_path="$PWD/exps/$exp_name/dr_data/dpr_index_${dataset}_${epoch}_${step}"
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
dr_out="$PWD/exps/$exp_name/dr_data/dr_${dataset}_${split}_${epoch}_${step}"
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

inference_out="$PWD/exps/$exp_name/inf_data/inference_${dataset}_${split}_${num_prompts}prompts_${epoch}_${step}.json"
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
     batch_size=10 max_length=1950 \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

echo -e "bash run \n python tmp_test.py --split $split"
if [ "${gen}" = "True" ]; then
python tmp_test.py --fp "${inference_out}" --dataset $dataset --split $split \
--exp_name ${exp_name} --iter_num -1 --epoch_num ${epoch} --method ${step} \
--prompt_num ${num_prompts}
else
python cls_test.py --fp "${inference_out}" --dataset $dataset --split $split \
--exp_name ${exp_name} --iter_num -1 --epoch_num ${epoch} --method ${step} \
--prompt_num ${num_prompts}
fi
done
