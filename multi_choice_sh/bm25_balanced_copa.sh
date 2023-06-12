
export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
exps_name="bm25_balanced_copa_1217"

dataset="balanced_copa"

setup_type="q"
cvd=2,3
num_gpus=2
num_prompts=8
gen=False
main_process_port=23001
test_split=("test")

mkdir -p "$PWD/exps/${exps_name}/data"

echo -e "\n\ntesting\n\n"
for split in "${test_split[@]}"; do
bm25_path="$PWD/exps/${exps_name}/data/bm25_${dataset}_${setup_type}_${split}.json"
#if [ ! -f "${bm25_path}" ]; then
HYDRA_FULL_ERROR=1 \
python find_bm25_es.py output_path="$bm25_path" \
     dataset_split=$split setup_type=${setup_type} task_name=$dataset +ds_size=null L=50
#fi

echo "bm25_output_path ${bm25_path}"
out_f="$PWD/exps/${exps_name}/data/inference_bm25_${dataset}_${split}_${setup_type}_${num_prompts}prompts.json"
#if [ ! -f "${out_f}" ]; then
CUDA_VISIBLE_DEVICES=${cvd} \
HYDRA_FULL_ERROR=1 \
accelerate launch --num_processes ${num_gpus} --main_process_port ${main_process_port} --multi_gpu \
     inference.py \
     prompt_file="$bm25_path" \
     task_name=$dataset \
     gen="${gen}" \
     output_file="${out_f}" \
     num_prompts=${num_prompts} \
     batch_size=20 max_length=1950
#fi
echo "prediction_output_path $PWD/data/inference_bm25_${dataset}_${split}.json"

echo -e "bash run \n python tmp_test.py --split $split"
if [ "${gen}" = "True" ]; then
python tmp_test.py --fp "${out_f}" --dataset $dataset --split $split \
--exp_name ${exps_name} --iter_num ${setup_type} --epoch_num -1 --method bm25 \
--prompt_num ${num_prompts}
else
python cls_test.py --fp "${out_f}" --dataset $dataset --split $split \
--exp_name ${exps_name} --iter_num ${setup_type} --epoch_num -1 --method bm25 \
--prompt_num ${num_prompts}
fi
done

