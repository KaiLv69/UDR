import os
import json
import torch

def prepro_sentence(test_inputs, max_length, bos_token_id, eos_token_id):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        ids = [bos_token_id] + test_input + [eos_token_id]
        n_mask = max_length-len(ids)
        input_ids.append(ids+[0 for _ in range(n_mask)])
        attention_mask.append([1 for _ in ids] +
                              [0 for _ in range(n_mask)])
        token_type_ids.append([1 for _ in ids] +
                              [0 for _ in range(n_mask)])

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids}


def prepro_sentence_pair_single(ids1, ids2, max_length,
                                bos_token_id, eos_token_id, negate=False,
                                allow_truncation=False):

    assert not negate

    if bos_token_id is not None:
        ids1 = [bos_token_id] + ids1
    if eos_token_id is not None:
        ids2 = ids2 + [eos_token_id]
    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length

    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    if negate:
        token_type_ids = [0 for _ in ids1] + [-1 for _ in ids2] + [0 for _ in range(n_mask)]
    else:
        token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    return input_ids, attention_mask, token_type_ids


def prepro_sentence_pair(train_inputs, test_inputs, max_length,
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids = \
                prepro_sentence_pair_single(train_input, test_input, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}

def flatten_label_losses(label_losses, dev_data):
    for label in range(len(label_losses)):
        k = int(len(label_losses[label]) / len(dev_data))
        label_losses[label] = [
            label_losses[label][k*i:k*(i+1)]
            for i in range(len(dev_data))]
    return label_losses

# get templates + verbalizers
def get_prompts(task, idx):

    if task in ["sst2", "sst5", "mr", "cr", 'emotion', 'tweet_sentiment_extraction', 'financial_phrasebank', 'imdb']:
        templates = ["A %s one . ", "It was %s . ",
                     "All in all %s . ", "A %s piece . "]
    elif task in ["yelp_full", "yelp_binary", "amazon"]:
        templates = ["A %s one. ", "It was %s. ",
                     "All in all %s. ", "A %s piece. "]
    elif task=="trec":
        templates = ["%s : ", "Q: %s : ", "Why %s ? ", "Answer: %s . "]
    elif task in ["agnews", "sogou", "dbpedia", "yahoo", 'mtop_domain', 'amazon_scenario', 'bank77']:
        templates = ["Topic: %s. ", "Subject: %s. ",
                     "This is about %s. ", "It is about %s. "]
    elif task=="subj":
        templates = ["This is %s . ", "It's all %s . ",
                     "It's %s . ", "Is it %s ? "]
    elif task=="cola":
        templates = ["This is %s .",
                     "It is %s .",
                     "You are %s .",
                     "I am %s ."]
    elif task in ["mnli", "rte", 'qnli', 'snli', 'wnli', 'mrpc', 'qqp', 'boolq']:
        templates = ["Answer: %s ."]
    else:
        raise NotImplementedError(task)

    if task in ["sst2", "mr", "cr", "yelp_binary", 'imdb']:
        label_words = ["terrible", "great"]
    elif task in ['emotion']:
        label_words = ["sadness", 'joy', "love", "anger", "fear", "surprise"]
    elif task in ["sst5", "yelp_full", "amazon"]:
        label_words = ["terrible", "bad", "okay", "good", "great"]
    elif task in ["tweet_sentiment_extraction", 'financial_phrasebank']:
        label_words = ["terrible", "okay", "great"]
    elif task in ["agnews"]:
        label_words = ["World", "Sports", "Business", "Technology"]
    elif task in ["amazon_scenario"]:
        label_words = ['social', 'transport', 'calendar', 'play', 'news', 'datetime', 'recommendation', 'email', 'iot',
                       'general', 'audio', 'lists', 'qa', 'cooking', 'takeaway', 'music', 'alarm', 'weather']
    elif task in ["trec"]:
        label_words = ["Description", "Entity", "Expression",
                       "Human", "Location", "Number"]
    elif task in ["sogou"]:
        label_words = ["Sports", "Finance", "Entertainment",
                       "Automobile", "Technology"]
    elif task in ["subj"]:
        label_words = ["subjective", "objective"]
    elif task in ["cola"]:
        label_words = ["not grammatical", "grammatical"]
    elif task in ["dbpedia"]:
        label_words = ["Company",
                       "Educational Institution",
                       "Artist",
                       "Athlete",
                       "Office Holder",
                       "Mean of Transportation",
                       "Building",
                       "Natural Place",
                       "Village",
                       "Animal",
                       "Plant",
                       "Album",
                       "Film",
                       "Written Work"]
    elif task in ['bank77']:
        label_words = ['activate my card', 'age limit', 'apple pay or google pay', 'atm support', 'automatic top up', 'balance not updated after bank transfer', 'balance not updated after cheque or cash deposit', 'beneficiary not allowed', 'cancel transfer', 'card about to expire', 'card acceptance', 'card arrival', 'card delivery estimate', 'card linking', 'card not working', 'card payment fee charged', 'card payment not recognised', 'card payment wrong exchange rate', 'card swallowed', 'cash withdrawal charge', 'cash withdrawal not recognised', 'change pin', 'compromised card', 'contactless not working', 'country support', 'declined card payment', 'declined cash withdrawal', 'declined transfer', 'direct debit payment not recognised', 'disposable card limits', 'edit personal details', 'exchange charge', 'exchange rate', 'exchange via app', 'extra charge on statement', 'failed transfer', 'fiat currency support', 'get disposable virtual card', 'get physical card', 'getting spare card', 'getting virtual card', 'lost or stolen card', 'lost or stolen phone', 'order physical card', 'passcode forgotten', 'pending card payment', 'pending cash withdrawal', 'pending top up', 'pending transfer', 'pin blocked', 'receiving money', 'Refund not showing up', 'request refund', 'reverted card payment?', 'supported cards and currencies', 'terminate account', 'top up by bank transfer charge', 'top up by card charge', 'top up by cash or cheque', 'top up failed', 'top up limits', 'top up reverted', 'topping up by card', 'transaction charged twice', 'transfer fee charged', 'transfer into account', 'transfer not received by recipient', 'transfer timing', 'unable to verify identity', 'verify my identity', 'verify source of funds', 'verify top up', 'virtual card not working', 'visa or mastercard', 'why verify identity', 'wrong amount of cash received', 'wrong exchange rate for cash withdrawal']
    elif task in ["yahoo"]:
        label_words = ["Society & Culture",
                       "Science & Mathematics",
                       "Health",
                       "Education & Reference",
                       "Computers & Internet",
                       "Sports",
                       "Business & Finance",
                       "Entertainment & Music",
                       "Family & Relationships",
                       "Politics & Government"]
    elif task in ['mtop_domain']:
        label_words = ['messaging', 'weather', 'alarm', 'recipes', 'calling', 'reminder', 'news', 'event', 'music', 'timer', 'people']
    elif task in ["mnli", 'snli']:
        label_words = ["Entailment", "Inconclusive", "Contradiction"]
    elif task in ["qnli", 'wnli']:
        label_words = ["Entailment", "Inconclusive"]
    elif task in ["rte"]:
        label_words = ["True", "False"]
    elif task in ["mrpc", 'qqp', 'boolq']:
        label_words = ["No", "Yes"]
    else:
        raise NotImplementedError(task)

    return [templates[idx] % label_word for label_word in label_words]


def get_one_prompt(task, idx, cls):

    if task in ["sst2", "sst5", "mr", "cr", 'emotion', 'tweet_sentiment_extraction', 'financial_phrasebank', 'imdb']:
        templates = ["A %s one . ", "It was %s . ",
                     "All in all %s . ", "A %s piece . "]
    elif task in ["yelp_full", "yelp_binary", "amazon"]:
        templates = ["A %s one. ", "It was %s. ",
                     "All in all %s. ", "A %s piece. "]
    elif task=="trec":
        templates = ["%s : ", "Q: %s : ", "Why %s ? ", "Answer: %s . "]
    elif task in ["agnews", "sogou", "dbpedia", "yahoo", 'mtop_domain', 'amazon_scenario', 'bank77']:
        templates = ["Topic: %s. ", "Subject: %s. ",
                     "This is about %s. ", "It is about %s. "]
    elif task=="subj":
        templates = ["This is %s . ", "It's all %s . ",
                     "It's %s . ", "Is it %s ? "]
    elif task=="cola":
        templates = ["This is %s .",
                     "It is %s .",
                     "You are %s .",
                     "I am %s ."]
    elif task in ["mnli", 'rte', 'qnli', 'snli', 'wnli', 'mrpc', 'qqp', 'boolq']:
        templates = ["Answer: %s ."]
    else:
        raise NotImplementedError(task)

    if task in ["sst2", "mr", "cr", "yelp_binary", 'imdb']:
        label_words = ["terrible", "great"]
    elif task in ['emotion']:
        label_words = ["sadness", 'joy', "love", "anger", "fear", "surprise"]
    elif task in ["sst5", "yelp_full", "amazon"]:
        label_words = ["terrible", "bad", "okay", "good", "great"]
    elif task in ["tweet_sentiment_extraction", 'financial_phrasebank']:
        label_words = ["terrible", "okay", "great"]
    elif task in ["agnews"]:
        label_words = ["World", "Sports", "Business", "Technology"]
    elif task in ["amazon_scenario"]:
        label_words = ['social', 'transport', 'calendar', 'play', 'news', 'datetime', 'recommendation', 'email', 'iot',
                       'general', 'audio', 'lists', 'qa', 'cooking', 'takeaway', 'music', 'alarm', 'weather']
    elif task in ["trec"]:
        label_words = ["Description", "Entity", "Expression",
                       "Human", "Location", "Number"]
    elif task in ["sogou"]:
        label_words = ["Sports", "Finance", "Entertainment",
                       "Automobile", "Technology"]
    elif task in ["subj"]:
        label_words = ["subjective", "objective"]
    elif task in ["cola"]:
        label_words = ["not grammatical", "grammatical"]
    elif task in ["dbpedia"]:
        label_words = ["Company",
                       "Educational Institution",
                       "Artist",
                       "Athlete",
                       "Office Holder",
                       "Mean of Transportation",
                       "Building",
                       "Natural Place",
                       "Village",
                       "Animal",
                       "Plant",
                       "Album",
                       "Film",
                       "Written Work"]
    elif task in ['bank77']:
        label_words = ['activate my card', 'age limit', 'apple pay or google pay', 'atm support', 'automatic top up', 'balance not updated after bank transfer', 'balance not updated after cheque or cash deposit', 'beneficiary not allowed', 'cancel transfer', 'card about to expire', 'card acceptance', 'card arrival', 'card delivery estimate', 'card linking', 'card not working', 'card payment fee charged', 'card payment not recognised', 'card payment wrong exchange rate', 'card swallowed', 'cash withdrawal charge', 'cash withdrawal not recognised', 'change pin', 'compromised card', 'contactless not working', 'country support', 'declined card payment', 'declined cash withdrawal', 'declined transfer', 'direct debit payment not recognised', 'disposable card limits', 'edit personal details', 'exchange charge', 'exchange rate', 'exchange via app', 'extra charge on statement', 'failed transfer', 'fiat currency support', 'get disposable virtual card', 'get physical card', 'getting spare card', 'getting virtual card', 'lost or stolen card', 'lost or stolen phone', 'order physical card', 'passcode forgotten', 'pending card payment', 'pending cash withdrawal', 'pending top up', 'pending transfer', 'pin blocked', 'receiving money', 'Refund not showing up', 'request refund', 'reverted card payment?', 'supported cards and currencies', 'terminate account', 'top up by bank transfer charge', 'top up by card charge', 'top up by cash or cheque', 'top up failed', 'top up limits', 'top up reverted', 'topping up by card', 'transaction charged twice', 'transfer fee charged', 'transfer into account', 'transfer not received by recipient', 'transfer timing', 'unable to verify identity', 'verify my identity', 'verify source of funds', 'verify top up', 'virtual card not working', 'visa or mastercard', 'why verify identity', 'wrong amount of cash received', 'wrong exchange rate for cash withdrawal']
    elif task in ["yahoo"]:
        label_words = ["Society & Culture",
                       "Science & Mathematics",
                       "Health",
                       "Education & Reference",
                       "Computers & Internet",
                       "Sports",
                       "Business & Finance",
                       "Entertainment & Music",
                       "Family & Relationships",
                       "Politics & Government"]
    elif task in ['mtop_domain']:
        label_words = ['messaging', 'weather', 'alarm', 'recipes', 'calling', 'reminder', 'news', 'event', 'music', 'timer', 'people']
    elif task in ["mnli", 'snli']:
        label_words = ["Entailment", "Inconclusive", "Contradiction"]
    elif task in ["qnli", 'wnli']:
        label_words = ["Entailment", "Inconclusive"]
    elif task in ["rte"]:
        label_words = ["True", "False"]
    elif task in ["mrpc", 'qqp', 'boolq']:
        label_words = ["No", "Yes"]
    else:
        raise NotImplementedError(task)

    return templates[idx] % label_words[cls]


def get_label_from_template(task, template):
    if task in ["sst2", "mr", "cr", "yelp_binary", 'imdb', 'imdb']:
        label_words = ["terrible", "great"]
    elif task in ['emotion']:
        label_words = ["sadness", 'joy', "love", "anger", "fear", "surprise"]
    elif task in ["sst5", "yelp_full", "amazon"]:
        label_words = ["terrible", "bad", "okay", "good", "great"]
    elif task in ["tweet_sentiment_extraction", 'financial_phrasebank']:
        label_words = ["terrible", "okay", "great"]
    elif task in ["agnews"]:
        label_words = ["World", "Sports", "Business", "Technology"]
    elif task in ["amazon_scenario"]:
        label_words = ['social', 'transport', 'calendar', 'play', 'news', 'datetime', 'recommendation', 'email', 'iot',
                       'general', 'audio', 'lists', 'qa', 'cooking', 'takeaway', 'music', 'alarm', 'weather']
    elif task in ["trec"]:
        label_words = ["Description", "Entity", "Expression",
                       "Human", "Location", "Number"]
    elif task in ["sogou"]:
        label_words = ["Sports", "Finance", "Entertainment",
                       "Automobile", "Technology"]
    elif task in ["subj"]:
        label_words = ["subjective", "objective"]
    elif task in ["cola"]:
        label_words = ["not grammatical", "grammatical"]
    elif task in ["dbpedia"]:
        label_words = ["Company",
                       "Educational Institution",
                       "Artist",
                       "Athlete",
                       "Office Holder",
                       "Mean of Transportation",
                       "Building",
                       "Natural Place",
                       "Village",
                       "Animal",
                       "Plant",
                       "Album",
                       "Film",
                       "Written Work"]
    elif task in ['bank77']:
        label_words = ['activate my card', 'age limit', 'apple pay or google pay', 'atm support', 'automatic top up', 'balance not updated after bank transfer', 'balance not updated after cheque or cash deposit', 'beneficiary not allowed', 'cancel transfer', 'card about to expire', 'card acceptance', 'card arrival', 'card delivery estimate', 'card linking', 'card not working', 'card payment fee charged', 'card payment not recognised', 'card payment wrong exchange rate', 'card swallowed', 'cash withdrawal charge', 'cash withdrawal not recognised', 'change pin', 'compromised card', 'contactless not working', 'country support', 'declined card payment', 'declined cash withdrawal', 'declined transfer', 'direct debit payment not recognised', 'disposable card limits', 'edit personal details', 'exchange charge', 'exchange rate', 'exchange via app', 'extra charge on statement', 'failed transfer', 'fiat currency support', 'get disposable virtual card', 'get physical card', 'getting spare card', 'getting virtual card', 'lost or stolen card', 'lost or stolen phone', 'order physical card', 'passcode forgotten', 'pending card payment', 'pending cash withdrawal', 'pending top up', 'pending transfer', 'pin blocked', 'receiving money', 'Refund not showing up', 'request refund', 'reverted card payment?', 'supported cards and currencies', 'terminate account', 'top up by bank transfer charge', 'top up by card charge', 'top up by cash or cheque', 'top up failed', 'top up limits', 'top up reverted', 'topping up by card', 'transaction charged twice', 'transfer fee charged', 'transfer into account', 'transfer not received by recipient', 'transfer timing', 'unable to verify identity', 'verify my identity', 'verify source of funds', 'verify top up', 'virtual card not working', 'visa or mastercard', 'why verify identity', 'wrong amount of cash received', 'wrong exchange rate for cash withdrawal']
    elif task in ["yahoo"]:
        label_words = ["Society & Culture",
                       "Science & Mathematics",
                       "Health",
                       "Education & Reference",
                       "Computers & Internet",
                       "Sports",
                       "Business & Finance",
                       "Entertainment & Music",
                       "Family & Relationships",
                       "Politics & Government"]
    elif task in ['mtop_domain']:
        label_words = ['messaging', 'weather', 'alarm', 'recipes', 'calling', 'reminder', 'news', 'event', 'music', 'timer', 'people']
    elif task in ["mnli", 'snli']:
        label_words = ["Entailment", "Inconclusive", "Contradiction"]
    elif task in ["qnli", 'wnli']:
        label_words = ["Entailment", "Inconclusive"]
    elif task in ["rte"]:
        label_words = ["True", "False"]
    elif task in ["mrpc", 'qqp', 'boolq']:
        label_words = ["No", "Yes"]
    else:
        raise NotImplementedError(task)

    for i, label in enumerate(label_words):
        if label in template:
            return i

    raise Exception("No label words")


def get_paths(out_dir, gpt2, method, task, do_zeroshot,
              k, seed, train_seed, split, template_idx,
              batch_size=None, lr=None, warmup_steps=None,
              use_demonstrations=False,
              ensemble=False,
              prompt_tune=False,
              head_tune=False,
              transform_tune=False,
              n_prefix=20):

    model_name = gpt2

    if not do_zeroshot:
        if prompt_tune:
            model_name += "-prompt-ft"
            if n_prefix!=20:
                model_name += "-{}".format(n_prefix)
        elif head_tune:
            model_name += "-head-ft"
        elif transform_tune:
            model_name += "-transform-ft"
        else:
            model_name += "-all-ft"

    base_dir = os.path.join(out_dir,
                            model_name,
                            "{}{}{}".format(method,
                                              "-demon" if use_demonstrations else "",
                                              "-ensemble" if ensemble else ""),
                            task)


    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if do_zeroshot:
        cache_path = str(split)

        if use_demonstrations:
            cache_path += "-k={}-seed={}".format(k, seed)

        if use_demonstrations:
            cache_path += "-tseed={}".format(train_seed)

        cache_path += "-t={}".format(template_idx)

        return os.path.join(base_dir, cache_path+".pkl")

    assert batch_size is not None and lr is not None and warmup_steps is not None

    out_dir = "BS={}-k={}-t={}-seed={}-tseed={}-lr={}{}".format(
            batch_size, k, template_idx, seed, train_seed, lr,
            "-wamrup={}".format(warmup_steps) if warmup_steps>0 else "",
    )


    return os.path.join(base_dir, out_dir)


def prepend_task_tokens(tokenizer, inputs, n_prefix):
    task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(n_prefix)]
    tokenizer.add_tokens(task_tokens)
    task_token_ids = tokenizer(" ".join(task_tokens), return_tensors="pt")["input_ids"]
    assert task_token_ids.shape[-1]==n_prefix

    def convert(inputs):
        n_train = inputs["input_ids"].shape[0]

        new_input_ids=torch.cat([
                task_token_ids.repeat(n_train, 1),
                inputs["input_ids"][:,1:]], 1)

        inputs = dict(
            input_ids=new_input_ids,
            attention_mask=torch.cat([
                torch.ones((n_train, n_prefix-1), dtype=torch.long),
                inputs["attention_mask"]], 1),
            token_type_ids=torch.cat([
                torch.zeros((n_train, n_prefix-1), dtype=torch.long),
                inputs["token_type_ids"]], 1),
            labels=torch.cat([
                torch.zeros((n_train, n_prefix-1), dtype=torch.long),
                inputs["input_ids"]], 1))
        return inputs

    if type(inputs)==list:
        return [convert(_inputs) for _inputs in inputs]

    return convert(inputs)

def reassign_output_tokens(inputs, for_labels=True, mapping=None):
    '''
    if for_labels=True, keep input_ids and convert labels
    otherwise, keep labels and convert input_ids
    '''

    def get_unique_tokens(inputs):
        input_ids = inputs["input_ids"].detach().numpy().tolist()
        token_type_ids = inputs["token_type_ids"].detach().numpy().tolist()
        unique_tokens = set()
        for _input_ids, _token_type_ids in zip(input_ids, token_type_ids):
            unique_tokens |= set([_id for _id, _token_id in zip(_input_ids, _token_type_ids) if _token_id==int(for_labels)])
        return unique_tokens

    def convert_set_to_mapping(unique_tokens):
        unique_tokens = sorted(unique_tokens)
        return {token: new_token for new_token, token in enumerate(unique_tokens)}

    def apply_mapping(inputs, mapping):
        input_ids = inputs["input_ids"].detach().numpy().tolist()
        token_type_ids = inputs["token_type_ids"].detach().numpy().tolist()
        converted_input_ids = []
        for _input_ids, _token_type_ids in zip(input_ids, token_type_ids):
            converted_input_ids.append([])
            for _id, _token_id in zip(_input_ids, _token_type_ids):
                if _token_id==int(for_labels):
                    converted_input_ids[-1].append(mapping[_id])
                else:
                    converted_input_ids[-1].append(0)
        converted_input_ids = torch.LongTensor(converted_input_ids)
        if for_labels:
            return dict(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"],
                        labels=converted_input_ids)
        return dict(input_ids=converted_input_ids,
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                    labels=inputs["input_ids"])

    if type(inputs)==list:
        if mapping is None:
            unique_tokens = set()
            for _inputs in inputs:
                unique_tokens |= get_unique_tokens(_inputs)
            mapping = convert_set_to_mapping(unique_tokens)
        rev_mapping = {v:k for k, v in mapping.items()}
        return rev_mapping, [apply_mapping(_inputs, mapping) for _inputs in inputs]

    assert mapping is None
    mapping = convert_set_to_mapping(get_unique_tokens(inputs))
    rev_mapping = {v:k for k, v in mapping.items()}
    return rev_mapping, apply_mapping(inputs, mapping)




