export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
export http_proxy=http://10.1.8.5:32680/
export https_proxy=http://10.1.8.5:32680/
cvd=4,5,6,7
fp16=True
gradient_checkpointing=False
exp_name="unified_bs128_1219"
mkdir -p "$PWD/exps/$exp_name/data"
mkdir -p "$PWD/exps/$exp_name/logs"
mkdir -p "$PWD/exps/$exp_name/iter1/logs"

train_dense_encoder_lr=1.3e-4
warmup_steps=2000
train_dense_encoder_seed=1208
#train_dense_encoder_seed=6006
train_retriever_batch_size=128
train_retriever_gradient_accumulate_step=1
loss_type='list_ranking'
rank_loss_factor=1
dpr_epoch=1
eval_per_epoch=20
pretrained_model_cfg="bert-base-uncased"
#pretrained_model_cfg="roberta-base"

train_datasets="[agnews_dataset,amazon_dataset,break_dataset,cnndailymail_dataset,cola_dataset,common_gen_dataset,\
copa_dataset,cosmos_qa_dataset,cr_dataset,cs_explan_dataset,cs_valid_dataset,dart_dataset,dbpedia_dataset,\
e2e_dataset,go_dataset,java_dataset,javascript_dataset,mnli_dataset,mr_dataset,mtop_dataset,php_dataset,\
pubmed_dataset,python_dataset,qnli_dataset,reddit_dataset,roc_ending_generation_dataset,roc_story_generation_dataset,\
rte_dataset,smcalflow_dataset,snli_dataset,sst2_dataset,sst5_dataset,subj_dataset,trec_dataset,\
wikiauto_dataset,yahoo_dataset,yelp_full_dataset]"
train_sampling_rates="[6.68583272,6.666666667,20.0,2.013753939,23.44665885,6.159342182,20.0,10.65643649,40.0,20.0080032,20.01601281,3.321155762,20.0040008,40.0,1.0,1.0,1.733252448,2.0,23.08935581,40.08955988,1.0,9.481144374,1.0,4.916662569,2.717465149,2.275261086,1.142517652,40.0,30.0,2.0,14.46968601,23.43566909,25.00312539,37.18163227,2.000080003,6.861063465,6.666666667]"

train_dense_encoder_py_output_path="$PWD/exps/$exp_name/model_ckpt_iter1_all"
echo -e "\n\n-train dense encoder-\n\n"

ckpt="/nvme/xnli/lk_code/exps/rtv_icl/v7/exps/unified_bs128_1219/model_ckpt_iter0_all_opt/dpr_biencoder.0_77041"

HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=$cvd \
python DPR/train_dense_encoder.py \
     model_file="${ckpt}" \
     ignore_checkpoint_offset="True" \
     ignore_checkpoint_optimizer="False" \
     seed=$train_dense_encoder_seed \
     fp16=${fp16} \
     loss_type=$loss_type \
     rank_loss_factor=$rank_loss_factor \
     train_sampling_rates=${train_sampling_rates} \
     train_datasets="${train_datasets}" \
     datasets.agnews_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/agnews/agnews_train_scoredqa_merged.json" \
     datasets.amazon_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/amazon/amazon_train_scoredqa_merged.json" \
     datasets.break_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/break/break_train_scoredqa_merged_all.json" \
     datasets.cnndailymail_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/cnndailymail/cnndailymail_debug_scoredqa_merged_all.json" \
     datasets.cola_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/cola/cola_train_scoredqa_merged.json" \
     datasets.common_gen_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/common_gen/common_gen_train_scoredqa_merged_all.json" \
     datasets.copa_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/copa/copa_train_scoredqa_merged.json" \
     datasets.cosmos_qa_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/cosmos_qa/cosmos_qa_train_scoredqa_merged.json" \
     datasets.cr_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/cr/cr_train_scoredqa_merged.json" \
     datasets.cs_explan_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/cs_explan/cs_explan_train_scoredqa_merged.json" \
     datasets.cs_valid_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/cs_valid/cs_valid_train_scoredqa_merged.json" \
     datasets.dart_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/dart/dart_train_scoredqa_merged_all.json" \
     datasets.dbpedia_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/dbpedia/dbpedia_train_scoredqa_merged.json" \
     datasets.e2e_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/e2e/e2e_train_scoredqa_merged_all.json" \
     datasets.go_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/go/go_debug_scoredqa_merged_all.json" \
     datasets.java_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/java/java_debug_scoredqa_merged_all.json" \
     datasets.javascript_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/javascript/javascript_train_scoredqa_merged_all.json" \
     datasets.mnli_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/mnli/mnli_debug_scoredqa_merged.json" \
     datasets.mr_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/mr/mr_train_scoredqa_merged.json" \
     datasets.mtop_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/mtop/mtop_train_scoredqa_merged_all.json" \
     datasets.php_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/php/php_debug_scoredqa_merged_all.json" \
     datasets.pubmed_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/pubmed/pubmed_train_scoredqa_merged_all.json" \
     datasets.python_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/python/python_debug_scoredqa_merged_all.json" \
     datasets.qnli_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/qnli/qnli_train_scoredqa_merged.json" \
     datasets.reddit_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/reddit/reddit_train_scoredqa_merged_all.json" \
     datasets.roc_ending_generation_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/roc_ending_generation/roc_ending_generation_train_scoredqa_merged_all.json" \
     datasets.roc_story_generation_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/roc_story_generation/roc_story_generation_train_scoredqa_merged_all.json" \
     datasets.rte_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/rte/rte_train_scoredqa_merged.json" \
     datasets.smcalflow_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/smcalflow/smcalflow_debug_scoredqa_merged_dedup_all.json" \
     datasets.snli_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/snli/snli_debug_scoredqa_merged.json" \
     datasets.sst2_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/sst2/sst2_train_scoredqa_merged.json" \
     datasets.sst5_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/sst5/sst5_train_scoredqa_merged.json" \
     datasets.subj_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/subj/subj_train_scoredqa_merged.json" \
     datasets.trec_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/trec/trec_train_scoredqa_merged.json" \
     datasets.wikiauto_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/wikiauto/wikiauto_debug_scoredqa_merged_all.json" \
     datasets.yahoo_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/yahoo/yahoo_train_scoredqa_merged.json" \
     datasets.yelp_full_dataset.file="/nvme/xnli/lk_code/exps/rtv_icl/scored_data_iter1/yelp_full/yelp_full_train_scoredqa_merged.json" \
     train=biencoder_local \
     train.warmup_steps=${warmup_steps} \
     output_dir="$train_dense_encoder_py_output_path" \
     train.gradient_accumulation_steps=$train_retriever_gradient_accumulate_step \
     train.eval_per_epoch=${eval_per_epoch} \
     train.batch_size=$train_retriever_batch_size \
     train.num_train_epochs=$dpr_epoch \
     train.learning_rate=$train_dense_encoder_lr \
     hydra.run.dir="$PWD/exps/$exp_name/iter1/logs" \
     encoder.gradient_checkpointing=${gradient_checkpointing} \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}"

