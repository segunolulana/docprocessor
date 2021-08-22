r"""
Text Summarization
==================

Demonstrates summarizing text by extracting the most important sentences from it.

"""
from gensim.summarization import keywords
import glob
import pandas as pd
from os.path import expanduser
import logging
from pprint import pprint
from gensim.summarization import summarize

from sci_summ_utils import work_on_csv

home = expanduser("~")
import argparse


def main(args):
    text = ""
    for file in glob.glob(args.files, recursive=True):
        doc = work_on_csv(file, args.ct, args.cc, True)
        text += doc

    print("Summarizing text...")
    summary = summarize(text, word_count=300)
    pprint(trim_sentences_to_max_three_occur(summary))

    # (the default is 20%).
    # pprint(summarize(text, ratio=0.5))
    # pprint(summarize(text, word_count=50))

    print("Printing keywords...")
    kws = keywords(text, ratio=0.05, scores=True)
    pprint(kws)

    # ###############################################################################
    # # Larger example
    # # --------------
    # #
    # # Let us try an example with a larger piece of text. We will be using a
    # # synopsis of the movie "The Matrix", which we have taken from `this
    # # <http://www.imdb.com/title/tt0133093/synopsis?ref_=ttpl_pl_syn>`_ IMDb page.
    # #
    # # In the code below, we read the text file directly from a web-page using
    # # "requests". Then we produce a summary and some keywords.
    # #

    # import requests

    # text = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text
    # pprint(text)

    # ###############################################################################
    # # First, the summary
    # #
    # pprint(summarize(text, ratio=0.01))

    # ###############################################################################
    # # And now, the keywords:
    # #
    # pprint(keywords(text, ratio=0.01))


def trim_sentences_to_max_three_occur(doc):
    sentence_list = doc.split("\n")
    unique_sentences = []
    sentence_dict = {}
    for sentence in sentence_list:
        if sentence not in sentence_dict:
            sentence_dict[sentence] = 1
            unique_sentences.append(sentence)
        elif sentence_dict[sentence] < 3:
            unique_sentences.append(sentence)
            sentence_dict[sentence] += 1
    return '\n'.join(unique_sentences)


parser = argparse.ArgumentParser(description='Summarizes Delimited File')
parser.add_argument('-s', '--start')
parser.add_argument('-e', '--end')
parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
parser.add_argument("-f", "--files", help="Glob pattern for files")
parser.add_argument('--ct', help='csvtype, csv or tsv')
parser.add_argument('--cc', help='csv column')
args = parser.parse_args()

if args.debug:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pd.set_option('display.max_colwidth', None)

main(args)
