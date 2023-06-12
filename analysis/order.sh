export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
cvd=0,1,2,3
num_gpus=4
exp_name="order"
inf_bs=6
main_process_port=23200

dataset="reddit"
gen=True
split="test"
num_prompts=-1
method="upr_new"
dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v6/exps/bm25_sst2_1206/data/bm25_sst2_q_test.json"
dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v6/exps/bm25_trec_1206/data/bm25_trec_q_test.json"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_1219/dr_data_iter0_all/dr_sst2_test_0_77041"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_1219/dr_data_iter0_all/dr_trec_test_0_77041"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_1219/dr_data_iter0_all/dr_smcalflow_validation_0_77041"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/bm25_smcalflow_0104/data/bm25_smcalflow_q_validation.json"
dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_1219/dr_data_iter0_all/dr_reddit_test_0_77041"
dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_1219/dr_data_iter0_all/dr_common_gen_validation_0_77041"
dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_sst2_test_11_11527"
dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_trec_test_11_11527"
dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_common_gen_validation_11_11527"
dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_reddit_test_11_11527"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_smcalflow_validation_11_11527"
mkdir -p "$PWD/exps/$exp_name/${dataset}"
mkdir -p "$PWD/exps/$exp_name/${dataset}/logs"

echo -e "\n\n-testing-\n\n"
for i in 3 4; do
for order in "random"; do
  echo -e "bash run \n accelerate launch inference.py --split $split --order $order"

  inference_out="$PWD/exps/$exp_name/${dataset}/inference_${method}_${dataset}_${split}_${num_prompts}prompts_${order}_${i}.json"
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
       order=${order} \
       hydra.run.dir="$PWD/exps/$exp_name/${dataset}/logs"
  fi

  echo -e "bash run \n python tmp_test.py --split $split"
  if [ "${gen}" = "True" ]; then
  python tmp_test.py --fp "${inference_out}" --dataset $dataset --split $split \
  --exp_name ${exp_name} --iter_num ${i} --epoch_num ${order} --method ${method} \
  --prompt_num ${num_prompts} --iter_scored_num "all"
  else
  python cls_test.py --fp "${inference_out}" --dataset $dataset --split $split \
  --exp_name ${exp_name} --iter_num ${i} --epoch_num ${order} --method ${method} \
  --prompt_num ${num_prompts} --iter_scored_num "all"
  fi
done
done