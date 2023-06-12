export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
cvd=4,5,6,7
fp16=True
gradient_checkpointing=False
exp_name="unified_bs128"
mkdir -p "$PWD/exps/$exp_name/data"
mkdir -p "$PWD/exps/$exp_name/logs"

train_dense_encoder_lr=1.3e-4
warmup_steps=5000
train_dense_encoder_seed=1208
#train_dense_encoder_seed=6006
train_retriever_batch_size=128
train_retriever_gradient_accumulate_step=1
loss_type='list_ranking'
rank_loss_factor=1
dpr_epoch=30
pretrained_model_cfg="bert-base-uncased"
#pretrained_model_cfg="roberta-base"

train_datasets="[agnews_dataset,amazon_dataset,break_dataset,cnndailymail_dataset,cola_dataset,common_gen_dataset,\
copa_dataset,cosmos_qa_dataset,cr_dataset,cs_explan_dataset,cs_valid_dataset,dart_dataset,dbpedia_dataset,\
e2e_dataset,go_dataset,java_dataset,javascript_dataset,mnli_dataset,mr_dataset,mtop_dataset,php_dataset,\
pubmed_dataset,python_dataset,qnli_dataset,reddit_dataset,roc_ending_generation_dataset,roc_story_generation_dataset,\
rte_dataset,smcalflow_dataset,snli_dataset,sst2_dataset,sst5_dataset,subj_dataset,trec_dataset,\
wikiauto_dataset,wnli_dataset,yahoo_dataset,yelp_full_dataset]"
train_sampling_rates="[2,2,2,2,2,2,1,2,2,2,2,1,2,2,2,1,1,2,2,2,2,5.33,1,2,1,2,2,2,4,2,1,2,2,2,2,2,2,2]"
echo "bm25 training"

train_dense_encoder_py_output_path="$PWD/exps/$exp_name/model_ckpt"
echo -e "\n\n-train dense encoder-\n\n"

HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=$cvd \
python DPR/train_dense_encoder.py \
     seed=$train_dense_encoder_seed \
     fp16=${fp16} \
     loss_type=$loss_type \
     rank_loss_factor=$rank_loss_factor \
     train_sampling_rates=${train_sampling_rates} \
     train_datasets="${train_datasets}" \
     train=biencoder_local \
     train.warmup_steps=${warmup_steps} \
     output_dir="$train_dense_encoder_py_output_path" \
     train.gradient_accumulation_steps=$train_retriever_gradient_accumulate_step \
     train.batch_size=$train_retriever_batch_size \
     train.num_train_epochs=$dpr_epoch \
     train.learning_rate=$train_dense_encoder_lr \
     hydra.run.dir="$PWD/exps/$exp_name/logs" \
     encoder.gradient_checkpointing=${gradient_checkpointing} \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}"

