export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
cvd=2,3
num_gpus=2
exp_name="epr_copa_1217_bert_80"
dataset="copa"
main_process_port=23100
num_prompts=8
train=True
fp16=True
gradient_checkpointing=False

mkdir -p "$PWD/exps/$exp_name/data"
mkdir -p "$PWD/exps/$exp_name/logs"

generate_embedding_batch_size=2048
train_dense_encoder_lr=1.3e-4
train_dense_encoder_seed=1208
train_retriever_batch_size=128
train_retriever_gradient_accumulate_step=1
loss_type='epr'
pretrained_model_cfg='bert-base-uncased'
rank_loss_factor=1
rank_loss_top_sample=1
dpr_epoch=80
warmup_steps=100
gen=False
train_set="train"
test_split=("test")

echo "bm25 training"
setup_type="qa"
if [ $train = True ]; then
find_bm25_py_output_path="$PWD/exps/${exp_name}/data/bm25_${dataset}_${setup_type}_${train_set}.json"
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
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

scorer_py_output_path="$PWD/exps/${exp_name}/data/${dataset}_scored_qa.json"
echo -e "\n\n-score-\n\n"
if [ ! -f "${scorer_py_output_path}" ]; then
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=$cvd \
accelerate launch --num_processes $num_gpus --main_process_port ${main_process_port} --multi_gpu \
     scorer.py \
     example_file="$find_bm25_py_output_path" \
     setup_type=qa \
     output_file="$scorer_py_output_path" \
     batch_size=20 \
     +task_name=$dataset +dataset_reader.ds_size=null \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

train_dense_encoder_py_output_path="$PWD/exps/$exp_name/model_ckpt"
echo -e "\n\n-train dense encoder-\n\n"
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=$cvd \
python DPR/train_dense_encoder.py \
     seed=$train_dense_encoder_seed \
     fp16=${fp16} \
     loss_type=$loss_type \
     rank_loss_factor=$rank_loss_factor \
     train_datasets=[kp20k_dataset] \
     train=biencoder_local \
     output_dir="$train_dense_encoder_py_output_path" \
     datasets.kp20k_dataset.file="${scorer_py_output_path}" \
     datasets.kp20k_dataset.setup_type=qa \
     datasets.kp20k_dataset.hard_neg=true \
     datasets.kp20k_dataset.rank_loss_top_sample="${rank_loss_top_sample}" \
     datasets.kp20k_dataset.task_name="${dataset}" \
     datasets.kp20k_dataset.top_k=5 \
     train.gradient_accumulation_steps=$train_retriever_gradient_accumulate_step \
     train.batch_size=$train_retriever_batch_size \
     train.num_train_epochs=$dpr_epoch \
     train.warmup_steps=${warmup_steps} \
     train.learning_rate=$train_dense_encoder_lr \
     hydra.run.dir="$PWD/exps/$exp_name/logs" \
     encoder.gradient_checkpointing=${gradient_checkpointing} \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}"
fi

echo -e "\n\n-testing-\n\n"
ckpt="$PWD/exps/$exp_name/model_ckpt/dpr_biencoder.$((dpr_epoch-1))"
echo -e "\nbash run \n python DPR/generate_dense_embeddings.py\n"
generate_dense_embeddings_py_output_path="$PWD/exps/$exp_name/dpr_index"
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

for split in "${test_split[@]}"; do

echo -e "\nbash run \n python DPR/dense_retriever.py --split $split\n"
dr_out="$PWD/exps/$exp_name/data/epr_dr_data_${dataset}_${split}"
if [ ! -f "$PWD/exps/$exp_name/data/epr_dr_data_${dataset}_${split}" ]; then
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

inference_out="$PWD/exps/$exp_name/data/epr_inference_${dataset}_${split}_${num_prompts}prompts.json"
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
     batch_size=20 max_length=1950 \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

echo -e "bash run \n python tmp_test.py --split $split"

echo -e "bash run \n python tmp_test.py --split $split"
if [ "${gen}" = "True" ]; then
python tmp_test.py --fp "${inference_out}" --dataset $dataset --split $split \
--exp_name ${exp_name} --iter_num ${setup_type} --epoch_num $((dpr_epoch-1)) --method epr \
--prompt_num ${num_prompts}
else
python cls_test.py --fp "${inference_out}" --dataset $dataset --split $split \
--exp_name ${exp_name} --iter_num ${setup_type} --epoch_num $((dpr_epoch-1)) --method epr \
--prompt_num ${num_prompts}
fi
done
