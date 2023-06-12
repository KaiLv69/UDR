export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
cvd=4,5
num_gpus=2
exp_name="analysis_num"

main_process_port=23397

gen=False
splits=("validation")
dataset="rte"
method="udr_new"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v6/exps/bm25_yelp_full_1215/data/bm25_yelp_full_q_test.json"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v6/exps/epr_yelp_full_1215_bert/data/epr_dr_data_yelp_full_test"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_1219/dr_data_iter0_all/dr_yelp_full_test_0_77041"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/epr_yahoo_0106/data/epr_dr_data_yahoo_test"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v6/exps/epr_rte_1215_bert/data/epr_dr_data_rte_validation"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v6/exps/bm25_rte_1215/data/bm25_rte_q_validation.json"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_rte_validation_6_11527"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_roc_story_generation_test_6_11527"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/bm25_roc_story_generation_1207/data/bm25_roc_story_generation_q_test.json"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_java_test_6_11527"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/bm25_java_1203/data/bm25_java_q_test.json"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_yelp_full_test_11_11527"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_wikiauto_test_wiki_11_11527"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_java_test_11_11527"
dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_acc4_rk4_0108/dr_data/dr_rte_validation_11_11527"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/epr_java_1203_bert/data/epr_dr_data_java_test"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/bm25_yahoo_0102/data/bm25_yahoo_q_test.json"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/data/wikiauto/epr_dr_wikiauto_test_wiki"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_1219/dr_data_iter0_all/dr_wikiauto_test_wiki_0_77041"
#dr_out="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/bm25_wikiauto_0104/data/bm25_wikiauto_q_test_wiki.json"
inf_bs=12
mkdir -p "$PWD/exps/$exp_name/${dataset}"

for num_prompts in 3 5 7; do

  for split in "${splits[@]}"; do
    echo -e "bash run \n accelerate launch inference.py --split $split"

    inference_out="$PWD/exps/$exp_name/${dataset}/inference_${method}_${dataset}_${split}_${num_prompts}prompts.json"
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
    python tmp_test.py --fp "${inference_out}" --dataset $dataset --split $split \
    --exp_name ${exp_name} --iter_num "num" --epoch_num 11 --method ${method} \
    --prompt_num ${num_prompts} --iter_scored_num "all"
    else
    python cls_test.py --fp "${inference_out}" --dataset $dataset --split $split \
    --exp_name ${exp_name} --iter_num "num" --epoch_num 11 --method ${method} \
    --prompt_num ${num_prompts} --iter_scored_num "all"
    fi
    done
done
