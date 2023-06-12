from tqdm import tqdm
import argparse
import json


#original
def hybrid_scores(bm25_scores, dense_scores, alpha, beilv=100):
    # beilv: re-scaling dense score as it is percentage.
    scores = {}
    for idx, v in tqdm(dense_scores.items()):
        new_v = {}
        if idx not in bm25_scores:
            scores[idx] = v
            continue
        v2 = bm25_scores[idx]
        v_min = min(list(v.values()))
        v2_min = min(list(v2.values()))
        for _id, score in v.items():
            if _id not in v2:
                new_v[_id] = beilv * score + alpha * v2_min
            else:
                new_v[_id] = beilv * score + alpha * v2[_id]
        for _id, score in v2.items():
            if _id not in new_v:
                new_v[_id] = alpha * score + beilv * v_min
        scores[idx] = new_v
    return scores

def get_hybrid_scores(bm25_ctxs, dense_ctxs, alpha, beilv):
    # bm25_ctxs: list
    # dense_ctxs: list
    # bm25_min_score = min()
    bm25_idx_ctxs_dict = {}
    dense_idx_ctxs_dict = {}

    bm25_idx_scores_dict = {}
    dense_idx_scores_dict = {}

    for bm25_ctx in bm25_ctxs:
        bm25_idx_ctxs_dict[bm25_ctx['id']] = bm25_ctx

        bm25_idx_scores_dict[bm25_ctx['id']] = bm25_ctx['score']

    for dense_ctx in dense_ctxs:

        dense_idx_ctxs_dict[dense_ctx['id']] = dense_ctx

        dense_idx_scores_dict[dense_ctx['id']] = float(dense_ctx['score'])

    # bm25_scores = list(map(lambda x:x['score'],bm25_ctxs))
    # dense_scores = list(map(lambda x:x['score'],dense_ctxs))


    hybrid_scores_dict = {}

    if len(bm25_idx_scores_dict):
        bm25_min_score = min(bm25_idx_scores_dict.values())
    else:
        bm25_min_score = 0
    dense_min_score = min(dense_idx_scores_dict.values())

    for idx, dense_score in dense_idx_scores_dict.items():
        if idx not in bm25_idx_scores_dict:
            hybrid_scores_dict[idx] = beilv * dense_score + alpha * bm25_min_score
        else:
            hybrid_scores_dict[idx] = beilv * dense_score + alpha * bm25_idx_scores_dict[idx]

    for idx, bm25_score in bm25_idx_scores_dict.items():
        if idx not in hybrid_scores_dict:
            hybrid_scores_dict[idx] = alpha * bm25_score + beilv * dense_min_score

    # for idx, hybrid_score in hybrid_scores_dict.items():

    hybrid_scores_list = list(hybrid_scores_dict.items())

    hybrid_scores_list.sort(key=lambda x:x[1],reverse=True)

    hybrid_ctxs_list = []

    for idx, hybrid_score in hybrid_scores_list:
        # now_ctx =
        now_ctx = None
        if idx in dense_idx_ctxs_dict:
            now_ctx = dense_idx_ctxs_dict[idx]
            now_ctx['dense_score'] = now_ctx['score']

            if idx in bm25_idx_ctxs_dict:
                now_ctx['bm25_score'] = bm25_idx_ctxs_dict[idx]['score']
            else:
                now_ctx['bm25_score'] = -999

        else:
            now_ctx = bm25_idx_ctxs_dict[idx]
            now_ctx['bm25_score'] = now_ctx['score']
            now_ctx['dense_score'] = -999

        now_ctx['score'] = hybrid_score

        hybrid_ctxs_list.append(now_ctx)

    return hybrid_ctxs_list


def get_hybrid_scores_from_file():
    with open(args.bm25_path, "r") as f:
        bm25_data = json.load(f)
    with open(args.dense_path, "r") as f:
        dense_data = json.load(f)
    print("bm25_data: ", len(bm25_data))
    print("dense_data: ", len(dense_data))
    out_data = []
    if 'document' in bm25_data[0]:
        question_field = 'document'
    else:
        question_field = None
    for i, e in enumerate(bm25_data):
        # if e['question'] != bm25_data[i][question_field]:
        #     print(e['question'])
        #     print(bm25_data[i][question_field])

        hybrid_ctxs = get_hybrid_scores(e['ctxs'], dense_data[i]['ctxs'], args.alpha, args.beilv)
        e['ctxs'] = hybrid_ctxs
        out_data.append(e)
    with open(args.output_path, "w") as f:
        json.dump(out_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bm25_path', type=str)
    parser.add_argument('--dense_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beilv', type=float, default=1)
    args = parser.parse_args()
    get_hybrid_scores_from_file()

