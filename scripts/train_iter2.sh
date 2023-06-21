cvd=4,5,6,7
fp16=True
gradient_checkpointing=False
exp_name="bs128_grad-acc4_rk4_seed1208"

mkdir -p "$PWD/exps/$exp_name/data"
mkdir -p "$PWD/exps/$exp_name/logs"

train_dense_encoder_lr=1.3e-4
warmup_steps=2000
train_dense_encoder_seed=1208
train_retriever_batch_size=128
train_retriever_gradient_accumulate_step=4
loss_type='list_ranking'
rank_loss_factor=4
dpr_epoch=10
eval_per_epoch=1
pretrained_model_cfg="bert-base-uncased"

train_datasets="[agnews_dataset,amazon_dataset,break_dataset,cnndailymail_dataset,cola_dataset,common_gen_dataset,\
copa_dataset,cosmos_qa_dataset,cr_dataset,cs_explan_dataset,cs_valid_dataset,dart_dataset,dbpedia_dataset,\
e2e_dataset,go_dataset,java_dataset,mnli_dataset,mr_dataset,mtop_dataset,\
pubmed_dataset,python_dataset,reddit_dataset,roc_ending_generation_dataset,roc_story_generation_dataset,\
rte_dataset,smcalflow_dataset,snli_dataset,sst2_dataset,sst5_dataset,subj_dataset,trec_dataset,\
wikiauto_dataset,yahoo_dataset,yelp_full_dataset]"
train_sampling_rates="[1.32796954,1.326064751,1.09098953,0.728808723,2.486857169,1.274610601,10.27165339,1.676549424,5.457780654,2.297271022,2.297730798,1.323640297,2.297041238,2.049173252,0.726315577,0.726315577,0.726315577,2.467835848,1.837037767,0.726315577,0.968403553,0.726315577,1.19731204,0.774686486,0.776348677,4.606539463,0.726315577,2.762834762,2.486274289,2.568073857,3.131659862,0.726330103,1.345259534,1.326064751]"
train_dense_encoder_py_output_path="$PWD/exps/$exp_name/iter2/model_ckpt"
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
     datasets.agnews_dataset.file="${PWD}/exps/$exp_name/data/iter1/agnews_scoredqa_merged.json" \
     datasets.amazon_dataset.file="${PWD}/exps/$exp_name/data/iter1/amazon_scoredqa_merged.json" \
     datasets.break_dataset.file="${PWD}/exps/$exp_name/data/iter1/break_scoredqa_merged.json" \
     datasets.cnndailymail_dataset.file="${PWD}/exps/$exp_name/data/iter1/cnndailymail_scoredqa_merged.json" \
     datasets.cola_dataset.file="${PWD}/exps/$exp_name/data/iter1/cola_scoredqa_merged.json" \
     datasets.common_gen_dataset.file="${PWD}/exps/$exp_name/data/iter1/common_gen_scoredqa_merged.json" \
     datasets.copa_dataset.file="${PWD}/exps/$exp_name/data/iter1/copa_scoredqa_merged.json" \
     datasets.cosmos_qa_dataset.file="${PWD}/exps/$exp_name/data/iter1/cosmos_qa_scoredqa_merged.json" \
     datasets.cr_dataset.file="${PWD}/exps/$exp_name/data/iter1/cr_scoredqa_merged.json" \
     datasets.cs_explan_dataset.file="${PWD}/exps/$exp_name/data/iter1/cs_explan_scoredqa_merged.json" \
     datasets.cs_valid_dataset.file="${PWD}/exps/$exp_name/data/iter1/cs_valid_scoredqa_merged.json" \
     datasets.dart_dataset.file="${PWD}/exps/$exp_name/data/iter1/dart_scoredqa_merged.json" \
     datasets.dbpedia_dataset.file="${PWD}/exps/$exp_name/data/iter1/dbpedia_scoredqa_merged.json" \
     datasets.e2e_dataset.file="${PWD}/exps/$exp_name/data/iter1/e2e_scoredqa_merged.json" \
     datasets.go_dataset.file="${PWD}/exps/$exp_name/data/iter1/go_scoredqa_merged.json" \
     datasets.java_dataset.file="${PWD}/exps/$exp_name/data/iter1/java_scoredqa_merged.json" \
     datasets.mnli_dataset.file="${PWD}/exps/$exp_name/data/iter1/mnli_scoredqa_merged.json" \
     datasets.mr_dataset.file="${PWD}/exps/$exp_name/data/iter1/mr_scoredqa_merged.json" \
     datasets.mtop_dataset.file="${PWD}/exps/$exp_name/data/iter1/mtop_scoredqa_merged.json" \
     datasets.php_dataset.file="${PWD}/exps/$exp_name/data/iter1/php_scoredqa_merged.json" \
     datasets.pubmed_dataset.file="${PWD}/exps/$exp_name/data/iter1/pubmed_scoredqa_merged.json" \
     datasets.python_dataset.file="${PWD}/exps/$exp_name/data/iter1/python_scoredqa_merged.json" \
     datasets.reddit_dataset.file="${PWD}/exps/$exp_name/data/iter1/reddit_scoredqa_merged.json" \
     datasets.roc_ending_generation_dataset.file="${PWD}/exps/$exp_name/data/iter1/roc_ending_generation_scoredqa_merged.json" \
     datasets.roc_story_generation_dataset.file="${PWD}/exps/$exp_name/data/iter1/roc_story_generation_scoredqa_merged.json" \
     datasets.rte_dataset.file="${PWD}/exps/$exp_name/data/iter1/rte_scoredqa_merged.json" \
     datasets.smcalflow_dataset.file="${PWD}/exps/$exp_name/data/iter1/smcalflow_scoredqa_merged.json" \
     datasets.snli_dataset.file="${PWD}/exps/$exp_name/data/iter1/snli_scoredqa_merged.json" \
     datasets.sst2_dataset.file="${PWD}/exps/$exp_name/data/iter1/sst2_scoredqa_merged.json" \
     datasets.sst5_dataset.file="${PWD}/exps/$exp_name/data/iter1/sst5_scoredqa_merged.json" \
     datasets.subj_dataset.file="${PWD}/exps/$exp_name/data/iter1/subj_scoredqa_merged.json" \
     datasets.trec_dataset.file="${PWD}/exps/$exp_name/data/iter1/trec_scoredqa_merged.json" \
     datasets.wikiauto_dataset.file="${PWD}/exps/$exp_name/data/iter1/wikiauto_scoredqa_merged.json" \
     datasets.yahoo_dataset.file="${PWD}/exps/$exp_name/data/iter1/yahoo_scoredqa_merged.json" \
     datasets.yelp_full_dataset.file="${PWD}/exps/$exp_name/data/iter1/yelp_full_scoredqa_merged.json" \
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
     hydra.run.dir="$PWD/exps/$exp_name/iter1/logs" \
     encoder.gradient_checkpointing=${gradient_checkpointing} \
     encoder.pretrained_model_cfg="${pretrained_model_cfg}"
