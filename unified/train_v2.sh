#export LD_LIBRARY_PATH=/nvme/xnli/anaconda3/envs/lk_epr/lib:$LD_LIBRARY_PATH
#export http_proxy=http://10.1.8.5:32680/
#export https_proxy=http://10.1.8.5:32680/
cvd=0,1,2,3
fp16=True
gradient_checkpointing=False
exp_name="unified_bs128_acc4_rk4_0108"
exp_name="unified_bs128_acc4_rk4_0612"

mkdir -p "$PWD/exps/$exp_name/data"
mkdir -p "$PWD/exps/$exp_name/logs"

train_dense_encoder_lr=1.3e-4
warmup_steps=2000
train_dense_encoder_seed=1208
#train_dense_encoder_seed=6006
train_retriever_batch_size=128
train_retriever_gradient_accumulate_step=4
loss_type='list_ranking'
rank_loss_factor=4
dpr_epoch=30
eval_per_epoch=1
pretrained_model_cfg="bert-base-uncased"
#pretrained_model_cfg="roberta-base"

train_datasets="[agnews_dataset,amazon_dataset,break_dataset,cnndailymail_dataset,cola_dataset,common_gen_dataset,\
copa_dataset,cosmos_qa_dataset,cr_dataset,cs_explan_dataset,cs_valid_dataset,dart_dataset,dbpedia_dataset,\
e2e_dataset,go_dataset,java_dataset,mnli_dataset,mr_dataset,mtop_dataset,php_dataset,\
pubmed_dataset,python_dataset,reddit_dataset,roc_ending_generation_dataset,roc_story_generation_dataset,\
rte_dataset,smcalflow_dataset,snli_dataset,sst2_dataset,sst5_dataset,subj_dataset,trec_dataset,\
wikiauto_dataset,yahoo_dataset,yelp_full_dataset]"
train_datasets="[agnews_dataset,amazon_dataset,break_dataset,cnndailymail_dataset,cola_dataset,common_gen_dataset,\
copa_dataset,cosmos_qa_dataset,cr_dataset,cs_explan_dataset,cs_valid_dataset,dart_dataset,dbpedia_dataset,\
e2e_dataset,go_dataset,java_dataset,mnli_dataset,mr_dataset,mtop_dataset,php_dataset,\
pubmed_dataset,python_dataset,reddit_dataset,roc_ending_generation_dataset,roc_story_generation_dataset,\
rte_dataset,snli_dataset,sst2_dataset,sst5_dataset,subj_dataset,trec_dataset,\
wikiauto_dataset,yahoo_dataset,yelp_full_dataset]"
train_datasets="[agnews_dataset,amazon_dataset]"
train_sampling_rates="[1.32796954,1.326064751,1.09098953,0.728808723,2.486857169,1.274610601,10.27165339,1.676549424,5.457780654,2.297271022,2.297730798,1.323640297,2.297041238,2.049173252,0.726315577,0.726315577,0.726315577,2.467835848,1.837037767,0.726315577,0.968403553,0.726315577,1.19731204,0.774686486,0.776348677,4.606539463,0.972108817,0.726315577,2.762834762,2.486274289,2.568073857,3.131659862,0.726330103,1.345259534,1.326064751]"
train_sampling_rates="[1.32796954,1.326064751,1.09098953,0.728808723,2.486857169,1.274610601,10.27165339,1.676549424,5.457780654,2.297271022,2.297730798,1.323640297,2.297041238,2.049173252,0.726315577,0.726315577,0.726315577,2.467835848,1.837037767,0.726315577,0.968403553,0.726315577,1.19731204,0.774686486,0.776348677,4.606539463,0.726315577,2.762834762,2.486274289,2.568073857,3.131659862,0.726330103,1.345259534,1.326064751]"
train_sampling_rates="[1.32796954,1.326064751]"
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
     datasets.agnews_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/agnews/agnews_scored_qa.json" \
     datasets.amazon_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/amazon/amazon_train_scoredqa_merged.json" \
     datasets.break_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/break/break_train_scoredqa_merged_all.json" \
     datasets.cnndailymail_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/cnndailymail/cnndailymail_debug_scoredqa_merged_all.json" \
     datasets.cola_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/cola/cola_train_scoredqa_merged.json" \
     datasets.common_gen_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/common_gen/common_gen_train_scoredqa_merged_all.json" \
     datasets.copa_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/copa/copa_train_scoredqa_merged.json" \
     datasets.cosmos_qa_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/cosmos_qa/cosmos_qa_train_scoredqa_merged.json" \
     datasets.cr_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/cr/cr_train_scoredqa_merged.json" \
     datasets.cs_explan_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/cs_explan/cs_explan_train_scoredqa_merged.json" \
     datasets.cs_valid_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/cs_valid/cs_valid_train_scoredqa_merged.json" \
     datasets.dart_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/dart/dart_train_scoredqa_merged_all.json" \
     datasets.dbpedia_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/dbpedia/dbpedia_train_scoredqa_merged.json" \
     datasets.e2e_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/e2e/e2e_train_scoredqa_merged_all.json" \
     datasets.go_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/go/go_debug_scoredqa_merged_all.json" \
     datasets.java_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/java/java_debug_scoredqa_merged_all.json" \
     datasets.mnli_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/mnli/mnli_debug_scoredqa_merged.json" \
     datasets.mr_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/mr/mr_train_scoredqa_merged.json" \
     datasets.mtop_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/mtop/mtop_train_scoredqa_merged_all.json" \
     datasets.php_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/php/php_debug_scoredqa_merged_all.json" \
     datasets.pubmed_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/pubmed/pubmed_train_scoredqa_merged_all.json" \
     datasets.python_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/python/python_debug_scoredqa_merged_all.json" \
     datasets.reddit_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/reddit/reddit_train_scoredqa_merged_all.json" \
     datasets.roc_ending_generation_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/roc_ending_generation/roc_ending_generation_train_scoredqa_merged_all.json" \
     datasets.roc_story_generation_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/roc_story_generation/roc_story_generation_train_scoredqa_merged_all.json" \
     datasets.rte_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/rte/rte_train_scoredqa_merged.json" \
     datasets.smcalflow_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/smcalflow/smcalflow_debug_scoredqa_merged_dedup_all.json" \
     datasets.snli_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/snli/snli_debug_scoredqa_merged.json" \
     datasets.sst2_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/sst2/sst2_train_scoredqa_merged.json" \
     datasets.sst5_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/sst5/sst5_train_scoredqa_merged.json" \
     datasets.subj_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/subj/subj_train_scoredqa_merged.json" \
     datasets.trec_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/trec/trec_train_scoredqa_merged.json" \
     datasets.wikiauto_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/wikiauto/wikiauto_debug_scoredqa_merged_all.json" \
     datasets.yahoo_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/yahoo/yahoo_train_scoredqa_merged.json" \
     datasets.yelp_full_dataset.file="/remote-home/klv/exps/rtv_icl/data/scored_data/yelp_full/yelp_full_train_scoredqa_merged.json" \
     datasets.agnews_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.amazon_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.break_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.cnndailymail_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.cola_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.common_gen_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.copa_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.cosmos_qa_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.cr_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.cs_explan_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.cs_valid_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.dart_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.dbpedia_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.e2e_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.go_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.java_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.mnli_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.mr_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.mtop_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.php_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.pubmed_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.python_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.reddit_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.roc_ending_generation_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.roc_story_generation_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.rte_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.smcalflow_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.snli_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.sst2_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.sst5_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.subj_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.trec_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.wikiauto_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.yahoo_dataset.rank_loss_factor="${rank_loss_factor}" \
     datasets.yelp_full_dataset.rank_loss_factor="${rank_loss_factor}" \
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

