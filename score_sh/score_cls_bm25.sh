export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
cvd=0,1,3
num_gpus=3
datasets=("sst2" "sst5" "mr" "cr" "amazon" "trec" "agnews" "dbpedia" "subj" "cola" "rte" "yelp_full" "yahoo" "qnli" "wnli")
datasets=('boolq')
main_process_port=23990

for dataset in "${datasets[@]}"; do
echo "dataset: $dataset"
mkdir -p "/nvme/xnli/lk_code/exps/rtv_icl/scored_data/${dataset}"
mkdir -p "/nvme/xnli/lk_code/exps/rtv_icl/scored_data/${dataset}/logs"

sc_path="/nvme/xnli/lk_code/exps/rtv_icl/scored_data/${dataset}"
train_set="train"

echo "bm25 training"
setup_type="qa"
find_bm25_py_output_path="${sc_path}/bm25_${dataset}_${setup_type}_${train_set}.json"
echo -e "\n\n-find_bm25-\n\n"
if [ ! -f "${find_bm25_py_output_path}" ]
then
HYDRA_FULL_ERROR=1 \
python find_bm25_es.py \
     output_path="$find_bm25_py_output_path" \
     dataset_split=${train_set} \
     setup_type=${setup_type} \
     task_name=${dataset} \
     +ds_size=null \
     L=50 \
     hydra.run.dir="/nvme/xnli/lk_code/exps/rtv_icl/scored_data/${dataset}/logs"
fi

scorer_py_output_path="${sc_path}/${dataset}_scored_qa.json"
echo -e "\n\n-score-\n\n"
if [ ! -f "${scorer_py_output_path}" ]; then
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=$cvd \
accelerate launch --num_processes $num_gpus --main_process_port ${main_process_port} --multi_gpu \
     scorer.py \
     example_file="$find_bm25_py_output_path" \
     setup_type=qa \
     output_file="$scorer_py_output_path" \
     batch_size=10 \
     +task_name=$dataset +dataset_reader.ds_size=null \
     hydra.run.dir="/nvme/xnli/lk_code/exps/rtv_icl/scored_data/${dataset}/logs"
fi
done