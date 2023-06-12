export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://10.1.8.5:32680/
export https_proxy=http://10.1.8.5:32680/
cvd=0,1,2,3
num_gpus=4

main_process_port=23100
train=True
fp16=True
gradient_checkpointing=False
method="cbr"
generate_embedding_batch_size=2048
train_dense_encoder_lr=1.3e-4
train_dense_encoder_seed=1208
train_retriever_batch_size=128
train_retriever_gradient_accumulate_step=1
eval_per_epoch=1
loss_type='epr'
pretrained_model_cfg='bert-base-uncased'
rank_loss_factor=1
rank_loss_top_sample=1
dpr_epoch=1
warmup_steps=500
inf_bs=10

num_prompts=-1
gen=True
datasets=("dart" "e2e" "pubmed" "reddit" "roc_ending_generation" "roc_story_generation")
train_split="train"
test_split=("test")
datasets=("python" "go" "java" "php" "cnndailymail")
train_split="debug"
test_split=("test")
datasets=("common_gen" "break" "smcalflow")
train_split="train"
test_split=("validation")
hard_neg=true
train_sampling_rates="[30]"
for dataset in "${datasets[@]}"; do

  echo -e "\n${dataset}\n"

  mkdir -p "$PWD/exps/${method}/${dataset}/data"
  mkdir -p "$PWD/exps/${method}/${dataset}/logs"

  echo "bm25 training"
  setup_type="a"
  if [ $train = True ]; then
  bm25_path="$PWD/exps/${method}/${dataset}/data/bm25_cbr_${dataset}_${setup_type}_${train_split}.json"
  if [ ! -f "${bm25_path}" ]; then
  HYDRA_FULL_ERROR=1 \
  python find_bm25_es.py output_path="$bm25_path" \
       dataset_split=${train_split} setup_type=${setup_type} task_name=$dataset +ds_size=null L=50
  fi

  train_dense_encoder_py_output_path="$PWD/exps/${method}/${dataset}/model_ckpt"
  echo -e "\n\n-train dense encoder-\n\n"
  HYDRA_FULL_ERROR=1 \
  CUDA_VISIBLE_DEVICES=$cvd \
  python DPR/train_dense_encoder.py \
       seed=$train_dense_encoder_seed \
       fp16=${fp16} \
       train_sampling_rates=${train_sampling_rates} \
       loss_type=$loss_type \
       rank_loss_factor=$rank_loss_factor \
       train_datasets=[kp20k_dataset] \
       train=biencoder_local \
       output_dir="$train_dense_encoder_py_output_path" \
       datasets.kp20k_dataset.file="${bm25_path}" \
       datasets.kp20k_dataset.setup_type=qa \
       datasets.kp20k_dataset.hard_neg=${hard_neg} \
       datasets.kp20k_dataset.rank_loss_top_sample="${rank_loss_top_sample}" \
       datasets.kp20k_dataset.task_name="${dataset}" \
       datasets.kp20k_dataset.top_k=5 \
       train.gradient_accumulation_steps=$train_retriever_gradient_accumulate_step \
       train.batch_size=$train_retriever_batch_size \
       train.num_train_epochs=$dpr_epoch \
       train.warmup_steps=${warmup_steps} \
       train.eval_per_epoch=${eval_per_epoch} \
       train.learning_rate=$train_dense_encoder_lr \
       hydra.run.dir="$PWD/exps/${method}/${dataset}/logs" \
       encoder.gradient_checkpointing=${gradient_checkpointing} \
       encoder.pretrained_model_cfg="${pretrained_model_cfg}"
  fi

  echo -e "\n\n-testing-\n\n"
  ckpt=$(ls "$PWD/exps/${method}/${dataset}/model_ckpt")
  ckpt="$PWD/exps/${method}/${dataset}/model_ckpt/${ckpt}"
  echo -e "\nbash run \n python DPR/generate_dense_embeddings.py\n"
  generate_dense_embeddings_py_output_path="$PWD/exps/${method}/${dataset}/dpr_index"
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
       hydra.run.dir="$PWD/exps/${method}/${dataset}/logs"
  fi

  for split in "${test_split[@]}"; do

  echo -e "\nbash run \n python DPR/dense_retriever.py --split $split\n"
  dr_out="$PWD/exps/${method}/${dataset}/data/epr_dr_data_${dataset}_${split}"
  if [ ! -f "${dr_out}" ]; then
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
       hydra.run.dir="$PWD/exps/${method}/${dataset}/logs"
  fi
  echo -e "bash run \n accelerate launch inference.py --split $split"

  inference_out="$PWD/exps/${method}/${dataset}/data/epr_inference_${dataset}_${split}_${num_prompts}prompts.json"
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
       hydra.run.dir="$PWD/exps/${method}/${dataset}/logs"
  fi

  echo -e "bash run \n python tmp_test.py --split $split"
  if [ "${gen}" = "True" ]; then
  python tmp_test.py --fp "${inference_out}" --dataset $dataset --split $split \
  --exp_name "${method}/${dataset}" --iter_num ${setup_type} --epoch_num 29 --method ${method} \
  --prompt_num ${num_prompts}
  else
  python cls_test.py --fp "${inference_out}" --dataset $dataset --split $split \
  --exp_name "${method}/${dataset}" --iter_num ${setup_type} --epoch_num 29 --method ${method} \
  --prompt_num ${num_prompts}
  fi
  done
done
