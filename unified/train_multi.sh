export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://pjlab:pjlab321@10.1.52.47:3333
export https_proxy=http://pjlab:pjlab321@10.1.52.47:3333
cvd=4,5,6,7
fp16=True
gradient_checkpointing=False
exp_name="unified_bs128_1219"
mkdir -p "$PWD/exps/$exp_name/data"
mkdir -p "$PWD/exps/$exp_name/logs"

train_dense_encoder_lr=1.3e-4
warmup_steps=2000
train_dense_encoder_seed=1208
#train_dense_encoder_seed=6006
train_retriever_batch_size=128
train_retriever_gradient_accumulate_step=1
loss_type='list_ranking'
rank_loss_factor=1
dpr_epoch=3
eval_per_epoch=30
pretrained_model_cfg="bert-base-uncased"
#pretrained_model_cfg="roberta-base"

train_datasets="[agnews_dataset,amazon_dataset,break_dataset,cnndailymail_dataset,cola_dataset,common_gen_dataset,\
copa_dataset,cosmos_qa_dataset,cr_dataset,cs_explan_dataset,cs_valid_dataset,dart_dataset,dbpedia_dataset,\
e2e_dataset,go_dataset,java_dataset,javascript_dataset,mnli_dataset,mr_dataset,mtop_dataset,php_dataset,\
pubmed_dataset,python_dataset,qnli_dataset,reddit_dataset,roc_ending_generation_dataset,roc_story_generation_dataset,\
rte_dataset,smcalflow_dataset,snli_dataset,sst2_dataset,sst5_dataset,subj_dataset,trec_dataset,\
wikiauto_dataset,wnli_dataset,yahoo_dataset,yelp_full_dataset]"
train_sampling_rates="[6.68583272,6.666666667,20.0,2.013753939,23.44665885,6.159342182,20.0,10.65643649,40.0,20.0080032,20.01601281,3.321155762,20.0040008,40.0,1.0,1.0,1.733252448,2.0,23.08935581,40.08955988,1.0,9.481144374,1.0,4.916662569,2.717465149,2.275261086,1.142517652,40.0,30.0,2.0,14.46968601,23.43566909,25.00312539,37.18163227,2.000080003,40.0,6.861063465,6.666666667]"
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
     train.eval_per_epoch=${eval_per_epoch} \
     train.batch_size=$train_retriever_batch_size \
     train.num_train_epochs=$dpr_epoch \
     train.learning_rate=$train_dense_encoder_lr \
     hydra.run.dir="$PWD/exps/$exp_name/logs" \
     encoder.gradient_checkpointing=${gradient_checkpointing} \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}"

