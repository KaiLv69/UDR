dataset="xsum"
dataset='iwslt'
exp_name="hybrid_xsum_1126"
exp_name="hybrid_iwslt_1127"
cvd=1,2,3
num_gpus=3
main_process_port=23600
#main_process_port=23601
#main_process_port=23602
mkdir -p "$PWD/exps/$exp_name/data"
mkdir -p "$PWD/exps/$exp_name/logs"
reindexing=False
setup_type=q
split="test"

alpha=1
#beilv=1
for beilv in 0.7 0.8 0.9; do
bm25_path="$PWD/exps/${exp_name}/data/bm25_${dataset}-${setup_type}_${split}.json"
echo -e "bm25 \n ${exp_name} ${dataset} ${setup_type} ${split}"
if [ ! -f "${bm25_path}" ]; then
python find_bm25_es.py output_path="$bm25_path" \
     reindexing=${reindexing} \
     dataset_split=$split setup_type=${setup_type} task_name=$dataset +ds_size=null \
      L=100 score=True
fi

echo -e "\n-hybrid-\n"
dense_path="/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/rk_iter_xsum_1124_bert/data/dr_xsum_test_0_9"
dense_path="/nvme/xnli/lk_code/exps/rtv_icl/v5/exps/rk_iter_iwslt_1124_bert/data/dr_iwslt_test_0_4"
hybrid_out_path="$PWD/exps/${exp_name}/data/hybrid_${dataset}_${split}_${alpha}_${beilv}.json"
if [ ! -f "${hybrid_out_path}" ]; then
  python src/utils/hybrid_bm25_dense_scores.py \
      --bm25_path "${bm25_path}" \
      --dense_path "${dense_path}" \
      --output_path "${hybrid_out_path}" \
      --alpha ${alpha} \
      --beilv ${beilv}
fi

echo -e "bash run \n accelerate launch inference.py --split $split"
inf_out="$PWD/exps/$exp_name/data/inference_${dataset}_${split}_${alpha}_${beilv}"
if [ ! -f "$inf_out" ]
then
CUDA_VISIBLE_DEVICES=$cvd \
TOKENIZERS_PARALLELISM=false \
accelerate launch --num_processes $num_gpus --main_process_port ${main_process_port} --multi_gpu\
     inference.py \
     prompt_file="$hybrid_out_path" \
     task_name=$dataset \
     output_file="$inf_out" \
     batch_size=20 \
     max_length=1950 \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

echo -e "bash run \n python tmp_test.py --split $split"

python tmp_test.py --fp "$inf_out" --dataset $dataset --split $split \
--exp_name ${exp_name} --iter_num "0" --epoch_num "4" --method hybrid --alpha ${alpha} --beilv ${beilv}
done