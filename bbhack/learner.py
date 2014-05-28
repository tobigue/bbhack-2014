from __future__ import (absolute_import, division, print_function,
                        with_statement)

import re
import sys
import json

import numpy as np
from sklearn.linear_model import Perceptron

from sklearn.feature_extraction.text import HashingVectorizer

from bbhack.base import BaseListener


def print_tick():
    sys.stdout.write(".")
    sys.stdout.flush()


class StreamingLearner(BaseListener):
    """
    Trains a Perceptron classifier on a stream of data
    (updates with every sample) using feature hashing
    (as you cannot know the vocabulary in before).

    In this example only English tweets containing a happy
    :) or sad :( emoticons, which are used as annotation
    for the sentiment of the message, are used as training
    and testing data. Every 5th tweet is used for evaluation
    of the model.
    """

    def __init__(self, zmq_sub_string, channel):

        self.classes = ["pos", "neg"]
        self.re_emoticons = re.compile(r":\)|:\(")
        self.vec = HashingVectorizer(n_features=2 ** 20, non_negative=True)
        self.clf = Perceptron()

        self.count = {
            "train": {
                "pos": 0,
                "neg": 0,
            },
            "test": {
                "pos": 0,
                "neg": 0,
            }
        }

        self.train = 1
        self.eval_count = {
            "pos": {"tp": 0, "fp": 0, "fn": 0},
            "neg": {"tp": 0, "fp": 0, "fn": 0},
        }

        super(StreamingLearner, self).__init__(zmq_sub_string, channel)

    def on_msg(self, tweet):
        print_tick()

        if tweet.get("lang") != "en":
            return  # skip non english tweets

        emoticons = self.re_emoticons.findall(tweet["text"])

        if not emoticons:
            return  # skip tweets without emoticons

        text = self.re_emoticons.sub("", tweet["text"].replace("\n", ""))

        X = self.vec.transform([text])

        # label for message
        last_emoticon = emoticons[-1]
        if last_emoticon == ":)":
            label = "pos"
        elif last_emoticon == ":(":
            label = "neg"
        y = np.asarray([label])

        if not self.train:
            # use every 5th message for evaluation

            print("")
            print("TEST %s |" % label, text)

            self.count["test"][label] += 1

            y_pred = self.clf.predict(X)
            pred_label, gold_label = y_pred[0], label

            print("PRED: ", pred_label)

            if pred_label == gold_label:
                self.eval_count[gold_label]["tp"] += 1
            else:
                self.eval_count[pred_label]["fp"] += 1
                self.eval_count[gold_label]["fn"] += 1

            pos_acc = (
                self.eval_count["pos"]["tp"] / self.count["test"]["pos"]
            ) if self.count["test"]["pos"] else 0

            neg_acc = (
                self.eval_count["neg"]["tp"] / self.count["test"]["neg"]
            ) if self.count["test"]["neg"] else 0

            print("*** CLF TESTED ON: %s :) samples (Acc %.3f),"
                  " %s :( samples (Acc %.3f)" %
                 (self.count["test"]["pos"], pos_acc,
                  self.count["test"]["neg"], neg_acc))
            print(json.dumps(self.eval_count, indent=2))
            print()

        else:
            self.count["train"][label] += 1

            # set higher sample weight for underrepresented class
            tc = self.count["train"]
            if label == "pos":
                sample_weight = min(3, max(1, tc["neg"] - tc["pos"]))
            elif label == "neg":
                sample_weight = min(3, max(1, tc["pos"] - tc["neg"]))
            else:
                sample_weight = 0

            print("\nTRAIN %s (weight %s) |" % (label, sample_weight), text)

            print(">>> CLF TRAINED ON: %s :) samples, %s :( samples" % (
                self.count["train"]["pos"], self.count["train"]["neg"]))

            self.clf.partial_fit(X, y, self.classes, [sample_weight])

        self.train += 1
        # use every 5th message for evaluation
        if not self.train % 5:
            self.train = 0


def main():
    """Start the StreamingLearner."""

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--zmq_sub_string', default='tcp://*:5556')
    p.add_argument('--channel', default='tweet.stream')

    options = p.parse_args()

    stream = StreamingLearner(options.zmq_sub_string, options.channel)
    # this call will block
    stream.start()
