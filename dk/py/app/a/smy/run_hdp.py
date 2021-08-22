from collections import defaultdict
from gensim.utils import tokenize
import logging
import argparse
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import sys
import os
from binaryornot.check import is_binary
from pprint import pprint
import re
import datetime
import glob
from gensim.models.hdpmodel import HdpModel
from gensim.parsing.preprocessing import (
    preprocess_string,
    remove_stopwords,
    strip_numeric,
    strip_punctuation,
    strip_tags,
)
from sci_summ_utils import get_sumy_stopwords
from icecream import ic


# https://www.kaggle.com/akashram/topic-modeling-intro-implementation
# However, LSI has one major weakness – ambiguity.
# For example, how could a system determine if you are talking about Microsoft office, or the office in which you work.
# This is where LDA comes in.
# Although NMF gave the highest coherence score, LDA is the most used technique and considered to be consistent as it is likely to provide more "coherent" topics.
# NMF performs better where the topic probabilities should remain fixed per document.
# HDP on the other hand is less preferred since the number of topics is not determined in prior and hence used rarely.

# try:
#     # Works from site but not individually
#     from .summ_utils import split
# except ImportError:
#     from summ_utils import split


def main(args):
    total_doc = {}
    file_at_limit_reached, reached_limit, total_doc = extract_docs(args, total_doc)
    logging.debug(total_doc["doc_with_placeholders"][0:101])
    lines = total_doc["doc_with_placeholders"].split("\n\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    documents = non_empty_lines

    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric]
    preprocessed_docs = [preprocess_string(document, CUSTOM_FILTERS) for document in documents]

    # remove common words and tokenize
    stoplist = get_sumy_stopwords()
    # stoplist.extend([":", "::", "*", "]", "\\", ""])
    stoplist.extend(["www", "com", "co"])
    stoplist = set(stoplist)
    texts = [[word for word in preprocessed_doc if word not in stoplist] for preprocessed_doc in preprocessed_docs]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    # texts = [line.split() for line in non_empty_lines]
    # texts = [word for word in texts if word not in set(stop_words)]
    # print("texts", texts[0:6])

    journals_dictionary = Dictionary(texts)
    logging.debug(ic.format(journals_dictionary.token2id))
    logging.debug(ic.format(journals_dictionary[0]))
    logging.debug(ic.format(journals_dictionary[1]))
    journals_corpus = [journals_dictionary.doc2bow(text) for text in texts]

    if args.mode == "lda":
        num_topics_extracted = 100
        if args.num_topics_extracted:
            num_topics_extracted = args.num_topics_extracted
        print("lda")
        lda = LdaModel(journals_corpus, num_topics=num_topics_extracted, id2word=journals_dictionary, random_state=0)
        lda_topics = lda.print_topics(num_words=5)
        for topic in lda_topics:
            print(topic)
    elif args.mode == "hdp":
        # Might need to use tomotopy library instead. See https://towardsdatascience.com/dont-be-afraid-of-nonparametric-topic-models-part-2-python-e5666db347a
        print("hdp")
        # Unfortunately gives different results on every run
        hdp = HdpModel(corpus=journals_corpus, id2word=journals_dictionary, chunksize=1024)
        hdp_topics = hdp.print_topics(num_topics=100, num_words=2)
        # hdp_topics = hdp.print_topics(num_topics=20, num_words=10)
        for topic in hdp_topics:
            print(topic)


def extract_docs(args, total_doc):
    total_doc["line_count"] = 0
    total_doc["character_count"] = 0
    total_doc["word_count"] = 0
    total_doc["doc_with_placeholders"] = ""
    total_doc["doc"] = ""
    reached_limit = False
    file_at_limit_reached = ""
    # not args.files needed else will be stuck while debugging input through files in Pycharm
    if not sys.stdin.isatty() and not args.files:
        doc = "".join(sys.stdin)
        doc_with_placeholders = custom_preprocess(doc)
        doc_without_placeholders = re.sub(r"\s\*\d\s", "", doc_with_placeholders)
        doc_dict = {"doc": doc, "doc_with_placeholders": doc_with_placeholders}
        doc_properties = calc_doc_properties(doc_without_placeholders)
        doc_dict.update(doc_properties)
        total_doc["line_count"] += doc_dict["line_count"]
        total_doc["word_count"] += doc_dict["word_count"]
        total_doc["character_count"] += doc_dict["character_count"]
        total_doc["doc"] += doc_dict["doc"]
        total_doc["doc_with_placeholders"] += doc_dict["doc_with_placeholders"]
    else:
        # Using sorted(glob.glob('...')) so it follows file name order and not arbitrary order
        files_filtered = sorted(set(glob.glob(os.path.expanduser(args.files), recursive=True)))
        files_filtered = filter_by_time(args, files_filtered)
        files_filtered = filter_by_work_filename(args, files_filtered)
        for file in files_filtered:
            if os.path.isfile(file) and not is_binary(file) and not reached_limit:
                doc_dict = extract_doc(file)
                print(file)
                if total_doc["line_count"] + doc_dict["character_count"] < 4200000:
                    total_doc["line_count"] += doc_dict["line_count"]
                    total_doc["word_count"] += doc_dict["word_count"]
                    total_doc["character_count"] += doc_dict["character_count"]
                    total_doc["doc"] += doc_dict["doc"]
                    total_doc["doc_with_placeholders"] += doc_dict["doc_with_placeholders"]
                else:
                    reached_limit = True
                    file_at_limit_reached = file
    return file_at_limit_reached, reached_limit, total_doc


def filter_by_work_filename(args, files_filtered):
    if getattr(args, "work", False):
        work_filtered = []
        for file in files_filtered:
            if "Waterfox56Research" in file:
                continue
            work_filtered.append(file)
        files_filtered = work_filtered
    return files_filtered


def filter_by_time(args, files_filtered):
    if getattr(args, "timestamped", False):
        time_filtered = []
        for file in files_filtered:
            start_time = "0000-00-00"
            end_time = datetime.datetime.now().strftime("%Y-%m-%d")
            if args.start:
                start_time = args.start
            if args.end:
                end_time = args.end
            date_part = (os.path.basename(file))[int(args.tfs) : int(args.tfe)]
            if start_time <= date_part < end_time:
                time_filtered.append(file)
        files_filtered = time_filtered
    return files_filtered


def extract_doc(file):
    doc = open(file).read()
    logging.debug(file)
    doc_with_placeholders = custom_preprocess(doc)
    # doc_with_placeholders = doc
    doc_without_placeholders = re.sub(r"\s\*\d\s", "", doc_with_placeholders)
    doc_properties = calc_doc_properties(doc_without_placeholders)
    doc_dict = {"doc": doc, "doc_with_placeholders": doc_with_placeholders}
    doc_dict.update(doc_properties)
    return doc_dict


def custom_preprocess(text):
    pprint("Remove date and time...")
    state_str = r"( *\- State.*)"  # - State "DONE"       from "TODO"
    org_with_date_str = r"(:*[a-zA-Z]+:.*)"  # SCHEDULED: <2020-06-17 Wed .+28d>
    time_str = r"(([01]\d|2[0-3]):([0-5]\d)|24:00)"
    date_str = r"\d{4}-\d{2}-\d{2} "
    time_re_1 = re.compile(r"^" + state_str + r"|" + org_with_date_str + time_str, re.MULTILINE)
    text = time_re_1.sub("", text)
    starts_with_date = re.compile("^ *" + date_str + time_str, re.MULTILINE)
    text = starts_with_date.sub("", text)
    # TODO Evaluate effect of https as stop word
    # Inbuilt stopwords here https://github.com/RaRe-Technologies/gensim/blob/d5556ea2700333e07c8605385def94dd96fb2c94/gensim/parsing/preprocessing.py
    pprint("Remove my stopwords...")
    # Leave just name and surname
    stop_phrases = []
    new_sps = ["properties", "end", "created", "id"]
    stop_phrases.extend(new_sps)
    new_sps2 = ["try", "probably", "mon", "tue", "thu", "fri", "look into", "use", r"mygtd\d*"]
    stop_phrases.extend(new_sps2)
    pattern = re.compile(r"\b(" + r"|".join(stop_phrases) + r")\b\s*", re.IGNORECASE)
    text = pattern.sub("", text)
    remove_lines_with_phrases = ["www.google.com", "DONE", "HOLD", "CANCELLED"]
    pattern = re.compile(r"^.*(" + r"|".join(remove_lines_with_phrases) + r")\b.*$", re.IGNORECASE | re.MULTILINE,)
    text = pattern.sub("", text)
    text = re.sub(r"\.", " ", text)
    # text = re.sub(r"\-", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"https", " ", text)
    text = re.sub(r"TODO", " ", text)
    text = re.sub(r"PERSONAL", " ", text)
    text = re.sub(r"http", " ", text)
    # text = inbuilt_preprocess(text)
    return text


def calc_doc_properties(clean_doc):
    lines = clean_doc.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    clean_doc = ""
    count = len(non_empty_lines)
    word_count = 0
    character_count = 0
    for line in non_empty_lines:
        word_count += len(line.split())
        character_count += len(line)
        clean_doc += line + "\n"
    logging.debug(clean_doc)
    return {
        "line_count": count,
        "word_count": word_count,
        "character_count": character_count,
    }


def inbuilt_preprocess(text):
    text = text.lower()
    text = remove_stopwords(text)
    # tokens = tokenize(text)
    # text = strip_punctuation(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarizes Doc")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-f", "--files", help="Glob pattern for files")
    parser.add_argument("-r", "--summarize", action="store_true", help="Summarize")
    parser.add_argument("--kw", action="store_true", help="Find keywords")
    parser.add_argument("--sh", action="store_true", help="Short summary (600 words)")
    parser.add_argument("--me", action="store_true", help="Medium summary (1600 words)")
    parser.add_argument(
        "-o",
        "--org",
        action="store_true",
        help="Save summary output as .org, add weights when summarizing org mode docs",
    )
    parser.add_argument("-l", "--output", help="output location")
    parser.add_argument("-p", "--output_prefix", help="output prefix")
    parser.add_argument(
        "--hp",
        "--only_high_priority",
        action="store_true",
        help="Only high priority org mode docs",
        dest="only_high_priority",
    )
    parser.add_argument("--timestamped", "-t", action="store_true", help="timestamped like timesink")
    parser.add_argument("--tfs", help="time format substring start")
    parser.add_argument("--tfe", help="time format substring end")
    parser.add_argument("-s", "--start")
    parser.add_argument("-e", "--end")
    parser.add_argument("-w", "--work")
    parser.add_argument(
        "-a", "--append_keywords", action="store_true",
    )
    parser.add_argument(
        "-m", "--mode", choices=("lda", "hdp"), default="lda", help="Options are lda, hdp",
    )
    parser.add_argument("-n", "--num_topics_extracted", help="Number of topics extracted")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logging.getLogger("gensim").setLevel(logging.INFO)
    logging.getLogger("binaryornot").setLevel(logging.INFO)

    main(args)
