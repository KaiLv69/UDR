# js
bash my_scripts/pipeline_epr.sh
# js
bash my_scripts/pipeline_iter_cnn.sh

# totto
bash my_scripts/pipeline_iter_python.sh



bash my_scripts/pipeline_bm25.sh

bash my_scripts/pipeline_epr_ruby.sh

cd ../v6
bash my_scripts/pipeline_iter_e2e.sh

bash baseline_sh/sbert_q.sh
bash baseline_sh/cbr_mtop.sh

bash unified/test_multi.sh
bash unified/test_multi_transfer.sh

bash unified/test_iter0.sh



bash finished_sh/pipeline_bm25_mnli.sh
bash finished_sh/pipeline_iter_cs_valid.sh

bash score_sh/score_cls_debug.sh
bash score_sh/score_reddit.sh
bash score_sh/score_mtop.sh
bash score_sh/score_wikiauto.sh
bash score_sh/score_roc_ending_generation.sh
bash score_sh/score_smcalflow.sh
bash score_sh/score_gen_all.sh
bash score_sh/score_cls.sh




bash finished_sh/pipeline_bm25_mrpc.sh
bash finished_sh/pipeline_epr_rte.sh
bash finished_sh/pipeline_iter_rte.sh


bash finished_sh/pipeline_bm25_yelp_full.sh
bash finished_sh/pipeline_epr_yelp_full.sh
bash finished_sh/pipeline_iter_yelp_full.sh

bash multi_choice_sh/iter_arc_easy.sh
