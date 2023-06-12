#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""
import json
import logging
import pickle
import random

import itertools
import math
import torch
from torch import Tensor as T
from typing import List, Iterator, Callable, Tuple

logger = logging.getLogger()

import random

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

    return templates[idx] % label_words[cls]

def load_train_dataset(dataset,size=None,listify=True):
    if size is not None:
        p = size
        data = dataset['train']
        total_size = len(data)

        rand = random.Random(x=int(p*total_size))
        index_list = list(range(total_size))
        rand.shuffle(index_list)
        x = data.select(index_list[:int(p*total_size)])

        
    else:
        x = dataset['train']
    if listify:
        return list(x)
    else:
        return x

class App:
    def __init__(self):
        self.functions = {}
    def add(self, key):
        def adder(func):
            self.functions[key] = func
            return func
        return adder



def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            logger.info("Reading file %s", path)
            data = pickle.load(reader)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    logger.info("Total data size: {}".format(len(results)))
    return results


def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % path)
            data = json.load(f)
            results = data
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """

    def __init__(
        self,
        data: torch.utils.data.Dataset,
        shard_id: int = 0,
        num_shards: int = 1,
        batch_size: int = 1,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        strict_batch_size: bool = False,
    ):

        self.data = data
        total_size = len(data)

        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)

        samples_per_shard = math.ceil(total_size / self.shards_num)

        self.shard_start_idx = self.shard_id * samples_per_shard

        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)

        logger.info(
            "samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d",
            samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations,
        )

        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size

    def total_data_len(self) -> int:
        return len(self.data)

    def iterations_num(self) -> int:
        return self.max_iterations - self.iteration

    def max_iterations_num(self) -> int:
        return self.max_iterations

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for sample in self.data:
            visitor_func(sample)

    def get_shard_indices(self, epoch: int):
        indices = list(range(len(self.data)))
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices[self.shard_start_idx : self.shard_end_idx]
        return shard_indices

    # TODO: merge with iterate_ds_sampled_data
    def iterate_ds_data(self, epoch: int = 0) -> Iterator[List]:
        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
        max_iterations = self.max_iterations - self.iteration
        shard_indices = self.get_shard_indices(epoch)

        for i in range(
            self.iteration * self.batch_size, len(shard_indices), self.batch_size
        ):
            items_idxs = shard_indices[i : i + self.batch_size]
            if self.strict_batch_size and len(items_idxs) < self.batch_size:
                logger.debug("Extending batch to max size")
                items_idxs.extend(shard_indices[0 : self.batch_size - len(items)])
            self.iteration += 1
            items = [self.data[idx] for idx in items_idxs]
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        while self.iteration < max_iterations:
            logger.debug("Fulfilling non complete shard=".format(self.shard_id))
            self.iteration += 1
            items_idxs = shard_indices[0 : self.batch_size]
            items = [self.data[idx] for idx in items_idxs]
            yield items

        logger.info(
            "Finished iterating, iteration={}, shard={}".format(
                self.iteration, self.shard_id
            )
        )
        # reset the iteration status
        self.iteration = 0

    def iterate_ds_sampled_data(
        self, num_iterations: int, epoch: int = 0
    ) -> Iterator[List]:
        self.iteration = 0
        shard_indices = self.get_shard_indices(epoch)
        cycle_it = itertools.cycle(shard_indices)
        for i in range(num_iterations):
            items_idxs = [next(cycle_it) for _ in range(self.batch_size)]
            self.iteration += 1
            items = [self.data[idx] for idx in items_idxs]
            yield items

        logger.info(
            "Finished iterating, iteration={}, shard={}".format(
                self.iteration, self.shard_id
            )
        )
        # TODO: reset the iteration status?
        self.iteration = 0

    def get_dataset(self) -> torch.utils.data.Dataset:
        return self.data


class MultiSetDataIterator(object):
    """
    Iterator over multiple data sources. Useful when all samples form a single batch should be from the same dataset.
    """

    def __init__(
        self,
        datasets: List[ShardedDataIterator],
        shuffle_seed: int = 0,
        shuffle=True,
        sampling_rates: List = [],
        rank: int = 0,
    ):
        self.iterables = datasets
        data_lengths = [it.total_data_len() for it in datasets]
        self.total_data = sum(data_lengths)
        logger.info("rank=%d; Multi set data sizes %s", rank, data_lengths)
        logger.info("rank=%d; Multi set total data %s", rank, self.total_data)
        logger.info("rank=%d; Multi set sampling_rates %s", rank, sampling_rates)
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.iteration = 0
        self.rank = rank

        if sampling_rates:
            self.max_its_pr_ds = [
                int(ds.max_iterations_num() * sampling_rates[i])
                for i, ds in enumerate(datasets)
            ]
        else:
            self.max_its_pr_ds = [ds.max_iterations_num() for ds in datasets]

        self.max_iterations = sum(self.max_its_pr_ds)
        logger.info(
            "rank=%d; Multi set max_iterations per dataset %s", rank, self.max_its_pr_ds
        )
        logger.info("rank=%d; Multi set max_iterations %d", rank, self.max_iterations)

    def total_data_len(self) -> int:
        return self.total_data

    def get_max_iterations(self):
        return self.max_iterations

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[Tuple[List, int]]:

        logger.info("rank=%d; Iteration start", self.rank)
        logger.info(
            "rank=%d; Multi set iteration: iteration ptr per set: %s",
            self.rank,
            [it.get_iteration() for it in self.iterables],
        )

        data_src_indices = []
        iterators = []
        for source, src_its in enumerate(self.max_its_pr_ds):
            logger.info(
                "rank=%d; Multi set iteration: source %d, batches to be taken: %s",
                self.rank,
                source,
                src_its,
            )
            data_src_indices.extend([source] * src_its)

            iterators.append(
                self.iterables[source].iterate_ds_sampled_data(src_its, epoch=epoch)
            )

        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(data_src_indices)

        logger.info(
            "rank=%d; data_src_indices len=%d", self.rank, len(data_src_indices)
        )
        for i, source_idx in enumerate(data_src_indices):
            it = iterators[source_idx]
            next_item = next(it, None)
            if next_item is not None:
                self.iteration += 1
                yield (next_item, source_idx)
            else:
                logger.warning(
                    "rank=%d; Next item in the source %s is None", self.rank, source_idx
                )

        logger.info("rank=%d; last iteration %d", self.rank, self.iteration)

        logger.info(
            "rank=%d; Multi set iteration finished: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables],
        )
        [next(it, None) for it in iterators]

        # TODO: clear iterators in some non-hacky way
        for it in self.iterables:
            it.iteration = 0
        logger.info(
            "rank=%d; Multi set iteration finished after next: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables],
        )
        # reset the iteration status
        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration

    def get_dataset(self, ds_id: int) -> torch.utils.data.Dataset:
        return self.iterables[ds_id].get_dataset()

    def get_datasets(self) -> List[torch.utils.data.Dataset]:
        return [it.get_dataset() for it in self.iterables]


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError
