export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
cvd=2,3,4,5
num_gpus=4
fp16=True
gradient_checkpointing=False
inference=True
train=True
main_process_port=23200
num_prompts=8
iter_scored_num="all"
bm25_train=True
iter_train=True
multi_test=True
iter_test=True
exp_name="rk_iter_openbookqa_1216_bert_30+2*10"
gen=False
dataset="openbookqa"
method="iter"
test_split=("test")
train_set="train"
multi_epoch=(29 14)
iter_base=(0 10)
iter_epoch=(4 9)
mkdir -p "$PWD/exps/$exp_name/data"
mkdir -p "$PWD/exps/$exp_name/logs"

generate_embedding_batch_size=2048
train_dense_encoder_lr=1.3e-4
warmup_steps=300
train_dense_encoder_seed=1208
train_retriever_batch_size=32
train_retriever_gradient_accumulate_step=4
loss_type='list_ranking'
rank_loss_factor=1
rank_loss_top_sample=1
dpr_epoch=30
epoch_per_iter=10
pretrained_model_cfg="bert-base-uncased"



if [ ${train} = True ]; then
echo "bm25 training"

scorer_py_output_path="$PWD/exps/${exp_name}/data/${dataset}_scored_qa.json"
scorer_py_output_path="/nvme/xnli/lk_code/exps/rtv_icl/v6/exps/epr_openbookqa_1216_bert/data/openbookqa_scored_qa.json"
echo -e "\n\n-score-\n\n"

score_kp20k_output_path="${scorer_py_output_path}"
train_dense_encoder_py_output_path="$PWD/exps/$exp_name/model_ckpt"
echo -e "\n\n-train dense encoder-\n\n"
if [ ${bm25_train} = True ]; then
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=$cvd \
python DPR/train_dense_encoder.py \
     seed=$train_dense_encoder_seed \
     fp16=${fp16} \
     loss_type=$loss_type \
     rank_loss_factor=$rank_loss_factor \
     train_datasets=[kp20k_dataset] \
     train=biencoder_local \
     train.warmup_steps=${warmup_steps} \
     output_dir="$train_dense_encoder_py_output_path" \
     datasets.kp20k_dataset.file="$score_kp20k_output_path" \
     datasets.kp20k_dataset.setup_type=qa \
     datasets.kp20k_dataset.hard_neg=true \
     datasets.kp20k_dataset.task_name=${dataset} \
     datasets.kp20k_dataset.top_k=5 \
     datasets.kp20k_dataset.rank_loss_top_sample="${rank_loss_top_sample}" \
     train.gradient_accumulation_steps=$train_retriever_gradient_accumulate_step \
     train.batch_size=$train_retriever_batch_size \
     train.num_train_epochs=$dpr_epoch \
     train.learning_rate=$train_dense_encoder_lr \
     hydra.run.dir="$PWD/exps/$exp_name/logs" \
     encoder.gradient_checkpointing=${gradient_checkpointing} \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}"
fi
fi

if [ "${multi_test}" = True ]; then
  echo 'multi test'
fp="$PWD/exps/$exp_name"

for j in "${multi_epoch[@]}"; do
    echo -e "\n\n\n*******************j=$j*******************\n\n\n"
ckpt="${fp}/model_ckpt/dpr_biencoder.${j}"
generate_dense_embeddings_py_output_path="$PWD/exps/$exp_name/data/dpr_index_${j}"

echo -e "\n\nbash run \n python DPR/generate_dense_embeddings.py\n\n"
if [ ! -f "${generate_dense_embeddings_py_output_path}_0" ]
then
CUDA_VISIBLE_DEVICES=$cvd \
python DPR/generate_dense_embeddings.py \
     model_file=$ckpt \
     ctx_src=dpr_epr \
     shard_id=0 num_shards=1 \
     out_file=$generate_dense_embeddings_py_output_path \
     ctx_sources.dpr_epr.setup_type=qa \
     ctx_sources.dpr_epr.task_name=$dataset \
     +ctx_sources.dpr_epr.ds_size=null \
     batch_size=$generate_embedding_batch_size \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}" \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

for split in "${test_split[@]}"
do

echo -e "bash run \n python DPR/dense_retriever.py --split $split"
if [ ! -f "$PWD/exps/$exp_name/data/dr_${dataset}_${split}_${j}" ]
then
CUDA_VISIBLE_DEVICES=$cvd \
TOKENIZERS_PARALLELISM=False \
python DPR/dense_retriever.py \
     model_file=$ckpt \
     qa_dataset=qa_epr \
     ctx_datatsets=[dpr_epr] \
     datasets.qa_epr.dataset_split=$split \
     encoded_ctx_files=["${generate_dense_embeddings_py_output_path}_*"] \
     out_file="$PWD/exps/$exp_name/data/dr_${dataset}_${split}_${j}" \
     ctx_sources.dpr_epr.setup_type=qa \
     ctx_sources.dpr_epr.task_name=$dataset \
     datasets.qa_epr.task_name=$dataset \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}" \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

echo -e "bash run \n accelerate launch inference.py --split $split"
inference_out="$PWD/exps/$exp_name/data/epr_result_prediction_${dataset}_${split}_${j}"
if [ ! -f "$PWD/exps/$exp_name/data/epr_result_prediction_${dataset}_${split}_${j}" ] && [ $inference = True ]
then
CUDA_VISIBLE_DEVICES=$cvd \
TOKENIZERS_PARALLELISM=false \
accelerate launch --num_processes $num_gpus --main_process_port ${main_process_port} --multi_gpu\
     inference.py \
     prompt_file="$PWD/exps/$exp_name/data/dr_${dataset}_${split}_${j}" \
     task_name=$dataset \
     gen="${gen}" \
     output_file="$PWD/exps/$exp_name/data/epr_result_prediction_${dataset}_${split}_${j}" \
     batch_size=20 \
    num_prompts=${num_prompts} \
     max_length=1950 \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

echo -e "bash run \n python tmp_test.py --split $split"
if [ $inference = True ]
then
  if [ "${gen}" = "True" ]; then
python tmp_test.py --fp "${inference_out}" --dataset $dataset --split $split \
--exp_name ${exp_name} --iter_num -1 --epoch_num ${j} --method "iter" \
--prompt_num ${num_prompts} --method ${method}
else
python cls_test.py --fp "${inference_out}" --dataset $dataset --split $split \
--exp_name ${exp_name} --iter_num -1 --epoch_num ${j} --method "iter" \
--prompt_num ${num_prompts} --method ${method}
fi
fi
done
done
fi


echo "--iterative training--"
if [ ${iter_train} = True ]; then
for ((i = 0; i < $((2 * epoch_per_iter)); i+=$((epoch_per_iter)))); do
  echo "base epoch: $i"

  echo -e "\n\n-------generate dense embeddings-------\n\n"
  if ((i == 0)); then
    model_ckpt_file="$train_dense_encoder_py_output_path/dpr_biencoder.$((dpr_epoch-1))"
  else
    model_ckpt_file="$train_dense_encoder_py_output_path/dpr_biencoder.$((epoch_per_iter-1))"
  fi

  generate_dense_embeddings_py_output_path="$PWD/exps/$exp_name/dpr_index$i"
  if [ ! -f "${generate_dense_embeddings_py_output_path}_0" ]; then
  CUDA_VISIBLE_DEVICES=$cvd \
  python DPR/generate_dense_embeddings.py \
     model_file="$model_ckpt_file" \
     ctx_src=dpr_epr \
     shard_id=0 num_shards=1 \
     out_file=$generate_dense_embeddings_py_output_path \
     ctx_sources.dpr_epr.setup_type=qa \
     ctx_sources.dpr_epr.task_name=$dataset \
     +ctx_sources.dpr_epr.ds_size=null \
     batch_size=$generate_embedding_batch_size \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}" \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
  fi

  echo -e "\n\n-------dense retriever-------\n\n"
  if [ ! -f "$PWD/exps/$exp_name/data/epr_dr_data_${dataset}_${train_set}_$i" ]; then
  CUDA_VISIBLE_DEVICES=$cvd \
  TOKENIZERS_PARALLELISM=False \
  python DPR/dense_retriever.py \
    model_file="$model_ckpt_file" \
    qa_dataset=qa_epr \
    n_docs=50 \
    ctx_datatsets=[dpr_epr] \
    datasets.qa_epr.dataset_split="${train_set}" \
    encoded_ctx_files=["${generate_dense_embeddings_py_output_path}_*"] \
    out_file="$PWD/exps/$exp_name/data/epr_dr_data_${dataset}_${train_set}_$i" \
    ctx_sources.dpr_epr.setup_type=qa \
    ctx_sources.dpr_epr.task_name=$dataset \
    datasets.qa_epr.task_name=$dataset \
    encoder.pretrained_model_cfg="${pretrained_model_cfg}" \
    hydra.run.dir="$PWD/exps/$exp_name/logs"
  fi

  echo -e "\n\n-------format conversion-------\n\n"
  if ((i==0)); then
    in_scored_file="$scorer_py_output_path"
  else
    in_scored_file="$merge_out_path"
  fi
  out_dep_file="$PWD/exps/$exp_name/data/score_dep_file_${dataset}_${train_set}_$i"

  if [ ! -f "$out_dep_file" ]; then
  python src/utils/convert_format.py \
    --dataset $dataset \
    --func "convert" \
    --input_file "$PWD/exps/$exp_name/data/epr_dr_data_${dataset}_${train_set}_$i" \
    --input_scored_file "$in_scored_file" \
    --num ${iter_scored_num} \
    --output_file "$PWD/exps/$exp_name/data/epr_dr_converted_${dataset}_${train_set}_${i}" \
    --output_dep_file "${out_dep_file}"
  fi

  echo -e "\n\n-------scorer-------\n\n"
  scorer_py_output_path="$PWD/exps/$exp_name/data/${dataset}_scoredqa_$i.json"
  if [ ! -f "$scorer_py_output_path" ]; then
  CUDA_VISIBLE_DEVICES=$cvd \
  accelerate launch --num_processes $num_gpus --main_process_port ${main_process_port} --multi_gpu \
    scorer.py \
    example_file="$PWD/exps/$exp_name/data/epr_dr_converted_${dataset}_${train_set}_${i}" \
    setup_type=qa \
    output_file="$scorer_py_output_path" \
    batch_size=20 \
    +task_name=$dataset +dataset_reader.ds_size=null \
    hydra.run.dir="$PWD/exps/$exp_name/logs"
  fi

  echo -e "\n\n-------merge-------\n\n"
  merge_out_path="$PWD/exps/$exp_name/data/${dataset}_${train_set}_scoredqa_merged_${i}.json"
  if [ ! -f "$merge_out_path" ]; then
  python src/utils/convert_format.py \
    --dataset $dataset \
    --func "merge" \
    --scored_file1 "$out_dep_file" \
    --scored_file2 "$scorer_py_output_path" \
    --output_file "$merge_out_path"
  fi

  echo -e "\n\n-------train dense encoder-------\n\n"
  train_dense_encoder_py_output_path="$PWD/exps/$exp_name/model_ckpt_$i"
  mkdir -p "$train_dense_encoder_py_output_path"
#  if ((i>0)); then
  CUDA_VISIBLE_DEVICES=$cvd \
  python DPR/train_dense_encoder.py \
    model_file="$model_ckpt_file" \
    ignore_checkpoint_offset="True" \
    ignore_checkpoint_optimizer="True" \
     seed=$train_dense_encoder_seed \
     fp16=${fp16} \
     loss_type=$loss_type \
     rank_loss_factor=$rank_loss_factor \
     train_datasets=[kp20k_dataset] \
     train=biencoder_local \
    train.warmup_steps=100 \
     output_dir="$train_dense_encoder_py_output_path" \
     datasets.kp20k_dataset.file="$merge_out_path" \
     datasets.kp20k_dataset.setup_type=qa  datasets.kp20k_dataset.hard_neg=true \
     datasets.kp20k_dataset.rank_loss_top_sample="${rank_loss_top_sample}" \
     datasets.kp20k_dataset.task_name=$dataset \
     datasets.kp20k_dataset.top_k=5 \
     train.gradient_accumulation_steps=$train_retriever_gradient_accumulate_step \
     train.batch_size=$train_retriever_batch_size \
     train.num_train_epochs=$epoch_per_iter \
     train.learning_rate=$train_dense_encoder_lr \
     hydra.run.dir="$PWD/exps/$exp_name/logs" \
     encoder.gradient_checkpointing=${gradient_checkpointing} \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}"
#    fi
done
fi


if [ ${iter_test} = True ]; then
echo -e "\n\n-------testing-------\n\n"

fp="$PWD/exps/$exp_name"

echo -e "\n\n-------iter test-------\n\n"
for i in "${iter_base[@]}"; do
  for j in "${iter_epoch[@]}"; do
    echo -e "\n\n\n*******************i=${i}, j=$j*******************\n\n\n"
ckpt="${fp}/model_ckpt_${i}/dpr_biencoder.${j}"
generate_dense_embeddings_py_output_path="$PWD/exps/$exp_name/data/dpr_index_${i}_${j}"

echo -e "\n\nbash run \n python DPR/generate_dense_embeddings.py\n\n"
if [ ! -f "${generate_dense_embeddings_py_output_path}_0" ]
then
CUDA_VISIBLE_DEVICES=$cvd \
python DPR/generate_dense_embeddings.py \
     model_file=$ckpt \
     ctx_src=dpr_epr \
     shard_id=0 num_shards=1 \
     out_file=$generate_dense_embeddings_py_output_path \
     ctx_sources.dpr_epr.setup_type=qa \
     ctx_sources.dpr_epr.task_name=$dataset \
     +ctx_sources.dpr_epr.ds_size=null \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}" \
     batch_size=$generate_embedding_batch_size \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

for split in "${test_split[@]}"
do

echo -e "bash run \n python DPR/dense_retriever.py --split $split"
if [ ! -f "$PWD/exps/$exp_name/data/dr_${dataset}_${split}_${i}_${j}" ]
then
CUDA_VISIBLE_DEVICES=$cvd \
TOKENIZERS_PARALLELISM=False \
python DPR/dense_retriever.py \
     model_file=$ckpt \
     qa_dataset=qa_epr \
     ctx_datatsets=[dpr_epr] \
     datasets.qa_epr.dataset_split=$split \
     encoded_ctx_files=["${generate_dense_embeddings_py_output_path}_*"] \
     out_file="$PWD/exps/$exp_name/data/dr_${dataset}_${split}_${i}_${j}" \
     ctx_sources.dpr_epr.setup_type=qa \
     ctx_sources.dpr_epr.task_name=$dataset \
     datasets.qa_epr.task_name=$dataset \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}" \

     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

echo -e "bash run \n accelerate launch inference.py --split $split"
inference_out="$PWD/exps/$exp_name/data/epr_result_prediction_${dataset}_${split}_${i}_${j}"
if [ ! -f "$inference_out" ] && [ $inference = True ]
then
CUDA_VISIBLE_DEVICES=$cvd \
TOKENIZERS_PARALLELISM=false \
accelerate launch --num_processes $num_gpus --main_process_port ${main_process_port} --multi_gpu\
     inference.py \
     prompt_file="$PWD/exps/$exp_name/data/dr_${dataset}_${split}_${i}_${j}" \
     task_name=$dataset \
     gen="${gen}" \
    num_prompts=${num_prompts} \
     output_file="$inference_out" \
     batch_size=20 \
     max_length=1950 \
     hydra.run.dir="$PWD/exps/$exp_name/logs"
fi

echo -e "bash run \n python tmp_test.py --split $split"
if [ $inference = True ]
then
  if [ "${gen}" = "True" ]; then
python tmp_test.py --fp "${inference_out}" --dataset $dataset --split $split \
--exp_name ${exp_name} --iter_num ${i} --epoch_num ${j} --method "iter" \
--prompt_num ${num_prompts} --method ${method} --iter_scored_num ${iter_scored_num}
else
python cls_test.py --fp "${inference_out}" --dataset $dataset --split $split \
--exp_name ${exp_name} --iter_num ${i} --epoch_num "${j}" --method "iter" \
--prompt_num ${num_prompts} --method ${method} --iter_scored_num ${iter_scored_num}
fi
fi
done
done
done
fi
