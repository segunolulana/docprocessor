r"""
Text Summarization
==================

Demonstrates summarizing text by extracting the most important sentences from it.

"""
import os
from gensim.summarization import keywords
import glob
import pandas as pd
from os.path import expanduser
import logging
from pprint import pprint
from gensim.summarization import summarize

home = expanduser("~")
import argparse
import datetime


def main(args):
    text = ""
    for file in glob.glob(home + "/Documents/TimeSink/**/*.csv", recursive=True):
        start_time = "0000-00-00"
        end_time = datetime.datetime.now().strftime('%Y-%m-%d')
        if args.start:
            start_time = args.start
        if args.end:
            end_time = args.end
        date_part = (os.path.basename(file))[:-4]
        if date_part >= start_time and date_part < end_time:
            # if args.mode == "e" and
            doc = work_on_csv(file)
            text += doc

    print("Summarizing text...")
    pprint(summarize(text, split=True, word_count=300))

    # (the default is 20%).
    # pprint(summarize(text, ratio=0.5))
    # pprint(summarize(text, word_count=50))

    print("Printing keywords...")
    pprint(keywords(text, ratio=0.05, scores=True))


def work_on_csv(file):
    df = pd.read_csv(file)
    logging.debug(file)
    df = df[df['Application'].str.contains("Waterfox", na=False)]
    doc = df.Window.drop_duplicates().to_string(index=False)
    logging.debug(doc)
    return doc


parser = argparse.ArgumentParser(description='Summarizes Time Sink')
parser.add_argument('-s', '--start')
parser.add_argument('-e', '--end')
parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
parser.add_argument(
    '-m', '--mode', choices=("a", "e", "m"), default="a", help="a for all, e for evening, m for morning"
)
# parser.add_argument('--sh', action='store_true', help='Short summary')
args = parser.parse_args()

if args.debug:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pd.set_option('display.max_colwidth', None)

main(args)
