r"""
Text Summarization
==================

Demonstrates summarizing text by extracting the most important sentences from it.

"""
from collections import Counter
import builtins
import os
import re
from gensim.summarization import keywords
import glob
from os.path import expanduser
import logging
from pprint import pprint, pformat
from gensim.summarization import summarize
import argparse
import datetime
from binaryornot.check import is_binary
import sys
import random
import pandas as pd
import time
from concurrent.futures.process import ProcessPoolExecutor
from sumy.parsers.plaintext import (
    PlaintextParser,
)  # We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
import summ_utils
import math
from summ_utils import remove_empty_lines, string_found, get_sumy_stopwords
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from run_sum_helper import create_text_file_output_names
from sumy.utils import get_stop_words


try:
    from summarizer.sentence_handler import SentenceHandler
except ImportError:
    logging.warning("bert-extractive-summarizer not installed")

PROFILING = False
try:
    profile = builtins.profile
    PROFILING = True
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


# Same-Meaning-Stopwords that don't change meanings like "not", "neither" would change
SM_STOP_WORDS = ["a", "b", "c", "d", "e", "f", "like", "better", "blob", "master", "look", "into", "try"]
SM_STOP_WORDS.extend(["probably", "use", "check"])
# python_sws = ["self", "def", "args", "true", "false"]
# SM_STOP_WORDS.extend(python_sws)


# Lexrank actually slow from my test. Took 1min 37 s for summarising 16229 words, 3731 lines to 60 lines
# Lexrank has documented slowness https://github.com/miso-belica/sumy/issues/109
# To run on mygtd*.org, Bert took 5min 13 s (at one time) for summarising 12862 words, 1479 lines to about 34 lines
# For Bert, running in debug mode shows that live site huggingface.co is being called!
# Ex 1: run_summarization.py --kw -r -f "ThirtyScreenshots/ss*.txt" --tfs 3 --tfe 13 -t -s 2020-09-21 -e 2020-09-25
# Ex 2: run_summarization.py --kw -r -f "Dropbox/orgnotes/mygtd*.org" -o -p org_sum -l ~/Dropbox/org_summaries -a
@profile
def main(args):
    _, key_output_lcn, output_lcn = create_text_file_output_names(args)
    if args.summarize:
        outF = open(output_lcn, "a")
    keyOutF = open(key_output_lcn, "a")
    file_at_limit_reached, reached_limit, total_doc = extract_docs(args)

    pprint("Remove date and time...")
    time_re = re.compile(r".*\[.*(([01]\d|2[0-3]):([0-5]\d)|24:00).*\]")
    text = time_re.sub("", total_doc["doc_with_placeholders"])

    # TODO Remove text = remove_stop_words(text)
    # text = trim_sentences_to_max_occur(text)
    priority_text = generate_high_priority(text)
    priority_text_total_doc = total_doc.copy()
    priority_text_total_doc["doc_with_placeholders"] = priority_text
    priority_text_total_doc.update(calc_doc_properties(priority_text))

    date_removed_total_doc = total_doc.copy()
    date_removed_total_doc["doc_with_placeholders"] = text
    date_removed_total_doc.update(calc_doc_properties(text))

    # Must be done before customized sentence summarization that emphasizes words in org mode
    if args.kw:
        print("Printing keywords...")
        keyOutF.write("Printing keywords...\n")
        if args.debug:
            print("date_removed_total_doc['doc_with_placeholders']", date_removed_total_doc["doc_with_placeholders"])
            # print("date_removed_total_doc", date_removed_total_doc[0:2])
        if args.only_high_priority:
            kw_result = summarize_as_keywords(args.sh, args.me, priority_text_total_doc, keyOutF)
        else:
            kw_result = summarize_as_keywords(args.sh, args.me, date_removed_total_doc, keyOutF)
        col_width = max(len(kw_one[0]) for kw_one in kw_result) + 7 + 2
        num_columns = 2
        for count, item in enumerate(kw_result[0:200], 1):
            print((item[0] + " " + format(item[1], ".5f")).ljust(col_width), sep=" ", end="")
            if count % num_columns == 0:
                print()
        print("Truncated for space...")
        keyOutF.write(pformat(kw_result))
        keyOutF.write("\n")

    # TODO Look into speeding up. For example, summarizing "The Matrix" synopsis(about 36, 000 characters) takes about
    # 3.1 seconds, while summarizing 35, 000 characters of "Honest Abe" by Alonzo Rothschild takes about 8.5 seconds
    if args.summarize:
        # with ProcessPoolExecutor(max_workers=None) as executor:
        #     the_futures = {}
        #     executor.submit(summarize_as_sentences, args, file_at_limit_reached, outF, reached_limit, text, total_doc["character_count"],
        #                     total_doc["line_count"], total_doc["word_count"])
        #     priority_text = generate_high_priority(text)
        #     executor.submit(summarize_as_sentences, args, file_at_limit_reached, outF, reached_limit, priority_text, total_doc["character_count"],
        #                     total_doc["line_count"], total_doc["word_count"])
        if args.only_high_priority:
            summarize_as_sentences(
                args, file_at_limit_reached, outF, reached_limit, priority_text, total_doc, kw_result
            )
        else:
            summarize_as_sentences(args, file_at_limit_reached, outF, reached_limit, text, total_doc, kw_result)
        outF.close()
    keyOutF.close()

    # (the default is 20%).
    # pprint(summarize(text, ratio=0.5))
    # pprint(summarize(text, word_count=50))


def extract_docs(args):
    total_doc = {}
    total_doc["line_count"] = 0
    total_doc["character_count"] = 0
    total_doc["word_count"] = 0
    total_doc["doc_with_placeholders"] = ""
    total_doc["doc"] = ""
    total_doc["placeholder_count"] = 0
    reached_limit = False
    file_at_limit_reached = ""
    # not args.files needed else will be stuck while debugging input through files in Pycharm
    if not sys.stdin.isatty() and not args.files:
        doc = "".join(sys.stdin)
        doc = remove_stop_words(doc)
        placeholder_count, doc_with_placeholders = put_placeholders(args, doc)
        doc_without_placeholders = re.sub(r"\s\*\d\s", "", doc_with_placeholders)
        doc_dict = {"doc": doc, "doc_with_placeholders": doc_with_placeholders, "placeholder_count": placeholder_count}
        doc_properties = calc_doc_properties(doc_without_placeholders)
        doc_dict.update(doc_properties)
        total_doc["line_count"] += doc_dict["line_count"]
        total_doc["word_count"] += doc_dict["word_count"]
        total_doc["character_count"] += doc_dict["character_count"]
        total_doc["doc"] += doc_dict["doc"]
        total_doc["doc_with_placeholders"] += doc_dict["doc_with_placeholders"]
        total_doc["placeholder_count"] += doc_dict["placeholder_count"]
    else:
        # Using sorted(glob.glob('...')) so it follows file name order and not arbitrary order
        files_filtered = sorted(set(glob.glob(os.path.expanduser(args.files), recursive=True)))
        files_filtered = filter_by_time(args, files_filtered)
        files_filtered = filter_by_work_filename(args, files_filtered)
        for file in files_filtered:
            if os.path.isfile(file) and not is_binary(file) and not reached_limit:
                doc_dict = extract_doc(args, file)
                print(file)
                if total_doc["line_count"] + doc_dict["character_count"] < 4200000:
                    total_doc["line_count"] += doc_dict["line_count"]
                    total_doc["word_count"] += doc_dict["word_count"]
                    total_doc["character_count"] += doc_dict["character_count"]
                    total_doc["doc"] += doc_dict["doc"]
                    total_doc["doc_with_placeholders"] += doc_dict["doc_with_placeholders"]
                    total_doc["placeholder_count"] += doc_dict["placeholder_count"]
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


def remove_stop_words(text):
    # TODO Evaluate effect of https as stop word
    # Inbuilt stopwords here https://github.com/RaRe-Technologies/gensim/blob/d5556ea2700333e07c8605385def94dd96fb2c94/gensim/parsing/preprocessing.py
    pprint("Remove my stopwords...")
    # Leave just name and surname
    stop_phrases = []
    new_sps = ["mon", "tue", "thu", "fri"]
    stop_phrases.extend(new_sps)
    pattern = re.compile(r"\b(" + r"|".join(stop_phrases) + r")\b", re.IGNORECASE)
    text = pattern.sub("", text)
    non_words = ["(:properties:.*)", "(:end:)", "(:created:.*)", "(:id:.*)", r"(mygtd\d*:)"]
    pattern = re.compile(r"(" + r"|".join(non_words) + r")", re.IGNORECASE)
    text = pattern.sub("", text)
    pattern = re.compile(r" +")
    text = pattern.sub(" ", text)
    remove_lines_with_phrases = ["www.google.com", "DONE", "HOLD", "CANCELLED"]
    pattern = re.compile(r"^.*(" + r"|".join(remove_lines_with_phrases) + r")\b.*$", re.IGNORECASE | re.MULTILINE,)
    text = pattern.sub("", text)
    text = remove_empty_lines(text)
    return text


def put_placeholders(args, text):
    placeholder_count = 0

    text = re.sub(r"PERSONAL", " *6 ", text)
    placeholder_count, text = single_placeholder(placeholder_count, text, r"(\.)(\w)", r" *1 \2")

    def replacement(match):
        return "I want to " + match.group(2).lower()

    # if args.mode != "bert":
    #     text = re.sub(r"TODO", " *5 ", text)
    #     placeholder_count, text = single_placeholder(placeholder_count, text, r"\.\s", ". \n *9 ")
    # else:
    #     # text = re.sub(r"(^\*+\sTODO\s)(\w)", replacement, text, flags=re.MULTILINE)
    #     text = re.sub(r"(TODO\s)(\w)", replacement, text, flags=re.MULTILINE)
    text = re.sub(r"TODO", " *5 ", text)
    placeholder_count, text = single_placeholder(placeholder_count, text, r"\.\s", ". \n *9 ")
    placeholder_count, text = single_placeholder_M(placeholder_count, text, r"(\n)", ".\n")
    # Spacy used here in Bert works better e.g with wikipedia which has articles with sentence ending structures like
    # in political terms.[6] Some students define ...
    # quality of a good man."  Specifically ...
    # The sentence tokenizer doesn't work well for list of TODOs
    # GPT-3 is based on predicting next words
    placeholder_count, text = single_placeholder(placeholder_count, text, r"\-", " *2 ")
    placeholder_count, text = single_placeholder(placeholder_count, text, r"\/", " *3 ")
    placeholder_count, text = single_placeholder(placeholder_count, text, r"https", " *4 ")
    # placeholder_count, text = single_placeholder(placeholder_count, text, r"\-", " *2 ")
    placeholder_count, text = single_placeholder(placeholder_count, text, "http", " *7 ")
    placeholder_count, text = single_placeholder(placeholder_count, text, "\bcom\b", "*10 ")
    # placeholder_count, text = single_placeholder(placeholder_count, text, "github", "*11 ")
    return placeholder_count, text


def single_placeholder(placeholder_count, text, real_subtext, replm):
    c4 = len(re.findall(real_subtext, text))
    print("%d %s" % (c4, real_subtext))
    placeholder_count += c4
    text = re.sub(real_subtext, replm, text)
    return placeholder_count, text


def single_placeholder_M(placeholder_count, text, real_subtext, replm):
    c4 = len(re.findall(real_subtext, text, re.MULTILINE))
    print("%d %s" % (c4, real_subtext))
    placeholder_count += c4
    text = re.sub(real_subtext, replm, text, flags=re.MULTILINE)
    return placeholder_count, text


def revert_placeholders(result):
    priority_symbols_re = re.compile(r"\[Priority ")
    result = priority_symbols_re.sub(r"[#", result)
    result = re.sub(r" ?\*1\s", ".", result)
    result = re.sub(r" ?\*2\s", "-", result)
    result = re.sub(r" ?\*3\s", r"/", result)
    result = re.sub(r" ?\*4\s", r"https", result)
    result = re.sub(r" ?\*5\s", r"TODO", result)
    result = re.sub(r" ?\*6\s", r"PERSONAL", result)
    result = re.sub(r" ?\*7\s", r"http", result)
    result = re.sub(r"\.\s\*9\s", r"\n", result)
    result = re.sub(r"\*9\s", r"", result)
    result = re.sub(r"\n\n", r"\n", result)
    result = re.sub(r"\*10\s", r"com", result)
    # result = re.sub(r"\*11\s", r"github", result)
    placeholder_count = 0
    i = 0
    it_SM_STOP_WORDS = iter(SM_STOP_WORDS)
    len_stop_words = len(SM_STOP_WORDS)
    while i < len_stop_words + 1 + 10:
        if i not in range(1, 11):
            try:
                word = next(it_SM_STOP_WORDS)
                placeholder_count, result = single_placeholder(placeholder_count, result, fr"\s\*{i}\s", word)
            except StopIteration:
                print("Finished iterating stop words")
        i += 1
    return result


def summarize_as_keywords(short, medium, total_doc, keyOutF):
    # gensim.summarization.keywords fetching different results as it is non-deterministic
    # e.g with pos_filter=('NP') i.e noun phrase
    # https://github.com/RaRe-Technologies/gensim/issues/2586. Aug 9, 2019
    # https://github.com/DerwenAI/pytextrank probably better here?
    # Default gensim.summarization.keywords
    #   .keywords(text, ratio=0.2, words=None, split=False,
    #               scores=False, pos_filter=('NN', 'JJ'), lemmatize=False, deacc=True)
    # NN is noun, singular or mass. JJ is adjective
    result = ""
    # doc_without_placeholders = re.sub(r"\s\*\d\s", "", total_doc["doc_with_placeholders"])
    # doc_properties = calc_doc_properties(doc_without_placeholders)
    doc_with_placeholders = total_doc["doc_with_placeholders"]
    if short or total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
        print("Summarising to 800 words")
        keyOutF.write("Summarising to 800 words")
        keyOutF.write("\n")
        result = keywords(doc_with_placeholders, words=800, scores=True, split=True, lemmatize=True)
    elif medium or total_doc["word_count"] - total_doc["placeholder_count"] <= 32000:
        print("Summarising to 1800 words")
        keyOutF.write("Summarising to 1800 words")
        keyOutF.write("\n")
        result = keywords(doc_with_placeholders, words=1800, scores=True, split=True, lemmatize=True)
    else:
        print("Summarising by 5% of sentences")
        keyOutF.write("Summarising by 5% of sentences")
        keyOutF.write("\n")
        result = keywords(doc_with_placeholders, ratio=0.05, scores=True, split=True, lemmatize=True)
    print("Summary keywords count: ", len(result))
    keyOutF.write("Summary keywords count: %s" % len(result))
    keyOutF.write("\n")
    print("Trimmed doc count: ", len(doc_with_placeholders))
    keyOutF.write("Trimmed doc count: %s" % len(doc_with_placeholders))
    keyOutF.write("\n")
    return result


@profile
def summarize_as_sentences(args, file_at_limit_reached, outF, reached_limit, text, total_doc, kw_result):
    if args.org:
        # text = add_many_copies_to_add_weight(text)
        priority_re = re.compile(r"\[\#")
        text = priority_re.sub("[Priority ", text)
        if args.debug:
            pprint(text[:1500])
    pprint("Summarizing text...")
    # Aim 1MB each section of text
    results = []
    summarize_ratio = ""
    print("placeholder_count:%d" % total_doc["placeholder_count"])
    c0 = len(re.findall(r"\n", text))
    print("%s ." % c0)
    all_lines = text.split("\n")
    all_text = text
    num_sub_docs = math.ceil(len(all_lines) / 1000)
    docs = summ_utils.split(all_lines, num_sub_docs)
    if args.sh:
        summarize_ratio = "30 lines (short)"
    elif args.me or total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
        summarize_ratio = "60 lines (10%)"
    else:
        summarize_ratio = "5%"

    if args.long:
        if args.mode == "slr":
            results = [apply_sumy_lexrank(args, text, 1, total_doc)]
        elif args.mode == "lr":
            results = [apply_lexrank(args, text, 1, total_doc)]
        elif args.mode == "lsa":
            results = [apply_lsa(args, text, 1, total_doc)]
        elif args.mode == "luhn":
            results = [apply_luhn(args, text, 1, total_doc)]
        elif args.mode == "bert":
            results = [apply_bert(args, text, 1, total_doc)]
    else:
        print("%d subdocuments" % num_sub_docs)
        results = summarize_subdocs(args, docs, num_sub_docs, total_doc)
    # Duplicates still come even after trimming original passages
    results = [trim_sentences_to_max_occur(result) for result in results]
    for result in results:
        if args.append_keywords:
            note_ls = []
            length_of_appended_kw = 4
            length_of_kw = len(kw_result)
            lines = result.split("\n")
            result = ""
            keywords_exclude = set(["com", "github", "windows"])
            kw_result_temp = []
            for kw in kw_result:
                if kw[0] not in keywords_exclude:
                    kw_result_temp.append(kw)
            kw_result = kw_result_temp
            for line in lines:
                note_dict = {"line": line}
                # line_trimmed = line.rstrip()
                for i in range(length_of_kw):
                    note_dict["key" + str(i)] = ""
                kws = []
                lcv = 0
                for i, kw in enumerate(kw_result):
                    if string_found(kw[0], line.lower()):
                        kws.append("%s(%s)" % (kw[0], i))
                        note_dict["key" + str(lcv)] = kw[0]
                        lcv += 1
                    if lcv == length_of_kw:
                        break
                kws = ",".join(kws)
                note_dict["kws"] = kws
                note_ls.append(note_dict)

            srt = {s[0]: i for i, s in enumerate(kw_result)}
            for i in range(length_of_appended_kw - 1, -1, -1):
                note_ls = sorted(note_ls, key=lambda x: (srt.get(x["key" + str(i)], length_of_kw + 1)))

            for note_dict in note_ls:
                line = note_dict["line"]
                line_with_kws = line + "   " + note_dict["kws"] + "\n"
                result += line_with_kws
        print(result)
        outF.write(result)

    print("Original character count: ", total_doc["character_count"])
    outF.write("Original character count: %s\n" % total_doc["character_count"])
    print("Original word count: ", total_doc["word_count"])
    outF.write("Original word count: %s\n" % total_doc["word_count"])
    print("Original line count: ", total_doc["line_count"])
    outF.write("Original line count: %s\n" % total_doc["line_count"])
    total_result = ""
    for result in results:
        total_result += result + "\n\n"
    print("Summary word count: ", len(total_result.split()))
    outF.write("Summary word count: %s\n" % len(total_result.split()))
    print("Summary line count: ", len(total_result.split("\n")))
    outF.write("Summary line count: %s\n" % len(total_result.split("\n")))
    print("Summary ratio: ", summarize_ratio)
    outF.write("Summary ratio: %s\n" % summarize_ratio)
    if reached_limit:
        print("Lines limit was reached at ", file_at_limit_reached)
        outF.write("Lines limit was reached at %s\n" % file_at_limit_reached)
    print("temp original word count: %d" % len(all_text.split()))


def summarize_subdocs(args, docs, num_sub_docs, total_doc):
    with ProcessPoolExecutor() as executor:
        futures = {}
        i = 0
        for doc in docs:
            logging.debug("debug doc:")
            logging.debug(doc)
            text = "\n".join(doc)
            if args.mode == "slr":
                future = executor.submit(apply_sumy_lexrank, args, text, num_sub_docs, total_doc)
            elif args.m == "lr":
                future = executor.submit(apply_lexrank, args, text, num_sub_docs, total_doc)
            elif args.m == "lsa":
                future = executor.submit(apply_lsa, args, text, num_sub_docs, total_doc)
            elif args.m == "luhn":
                future = executor.submit(apply_luhn, args, text, num_sub_docs, total_doc)
            elif args.m == "bert":
                future = executor.submit(apply_bert, args, text, num_sub_docs, total_doc)
            futures[i] = future
            i += 1
        results = []
        for i, future in futures.items():  # So can see exceptions
            a_result = future.result()
            logging.debug(a_result)
            results.append(a_result)
    return results


def apply_sumy_lexrank(args, text, num_sub_docs, total_doc):
    result = ""
    logging.debug("text")
    logging.debug(text)
    lr_parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    # Definition of SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",)). Only sentence returned unfortunately
    if args.ush:
        summary = summarizer(lr_parser.document, max(1, math.ceil(2.0 / num_sub_docs)))
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    elif args.sh:
        summary = summarizer(lr_parser.document, math.ceil(30.0 / num_sub_docs))
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    elif args.me or total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
        summary = summarizer(lr_parser.document, math.ceil(60.0 / num_sub_docs))
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    else:
        no_sentences = math.ceil(5 * total_doc["line_count"] / 100.0 / num_sub_docs)
        summary = summarizer(lr_parser.document, no_sentences)
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    result = revert_placeholders(result)
    return result


def apply_lexrank(args, text, num_sub_docs, total_doc):
    result = ""
    logging.debug("text")
    logging.debug(text)
    document = text.splitlines()
    documents = [document]
    stopwords = STOPWORDS["en"]
    stopwords.update(["a", "b", "c", "d", "e", "f", "like", "better", "blob", "master"])
    lxr = LexRank(documents, stopwords=stopwords)
    # lxr = LexRank(documents)
    if args.ush:
        result = lxr.get_summary(document, summary_size=max(1, math.ceil(2.0 / num_sub_docs)))
    elif args.sh:
        result = lxr.get_summary(document, summary_size=math.ceil(30.0 / num_sub_docs))
    # elif args.me or total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
    #     summary = summarizer(lr_parser.document, math.ceil(60.0 / num_sub_docs))
    #     for sentence in summary:
    #         result += sentence._text + "\n"  # pylint: disable=protected-access
    else:
        no_sentences = math.ceil(5 * total_doc["line_count"] / 100.0 / num_sub_docs)
        result = lxr.get_summary(document, summary_size=no_sentences)

    print(result)

    result = "\n\n".join(result)
    result = revert_placeholders(result)
    return result


def apply_lsa(args, text, num_sub_docs, total_doc):
    from sumy.nlp.stemmers import Stemmer

    result = ""
    logging.debug("text")
    logging.debug(text)
    lr_parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    # summarizer = LsaSummarizer(Stemmer("english"))
    stop_words = get_sumy_stopwords()
    summarizer.stop_words = stop_words
    # Definition of SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",)). Only sentence returned unfortunately
    if args.ush:
        summary = summarizer(lr_parser.document, max(1, math.ceil(2.0 / num_sub_docs)))
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    elif args.sh:
        summary = summarizer(lr_parser.document, math.ceil(30.0 / num_sub_docs))
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    # elif args.me or total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
    #     summary = summarizer(lr_parser.document, math.ceil(60.0 / num_sub_docs))
    #     for sentence in summary:
    #         result += sentence._text + "\n"  # pylint: disable=protected-access
    elif args.auto and total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
        summary = summarizer(lr_parser.document, math.ceil(60.0 / num_sub_docs))
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    else:
        no_sentences = math.ceil(20 * total_doc["line_count"] / 100.0 / num_sub_docs)
        summary = summarizer(lr_parser.document, no_sentences)
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    print("Result before reverting placeholders:", result)
    result = revert_placeholders(result)
    return result


def apply_luhn(args, text, num_sub_docs, total_doc):
    from sumy.nlp.stemmers import Stemmer

    result = ""
    logging.debug("text")
    logging.debug(text)
    lr_parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LuhnSummarizer()
    # summarizer = LuhnSummarizer(Stemmer("english"))
    stop_words = list(get_stop_words("english"))
    STOP_WORDS = ["a", "b", "c", "d", "e", "f", "like", "better", "blob", "master", "look", "into", "try"]
    STOP_WORDS.extend(["probably", "use", "check"])
    python_sws = ["self", "def", "args", "true", "false"]
    STOP_WORDS.extend(python_sws)
    if args.deweb:
        STOP_WORDS.extend(["com", "github", "www"])
    stop_words.extend(STOP_WORDS)
    summarizer.stop_words = stop_words
    # Definition of SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",)). Only sentence returned unfortunately
    if args.ush:
        summary = summarizer(lr_parser.document, max(1, math.ceil(4.0 / num_sub_docs)))
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    elif args.sh:
        summary = summarizer(lr_parser.document, math.ceil(30.0 / num_sub_docs))
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    elif args.me or total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
        summary = summarizer(lr_parser.document, math.ceil(60.0 / num_sub_docs))
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    else:
        no_sentences = math.ceil(5 * total_doc["line_count"] / 100.0 / num_sub_docs)
        summary = summarizer(lr_parser.document, no_sentences)
        for sentence in summary:
            result += sentence._text + "\n"  # pylint: disable=protected-access
    result = revert_placeholders(result)
    return result


@profile
def apply_bert(args, text, num_sub_docs, total_doc):
    # https://medium.com/analytics-vidhya/text-summarization-using-bert-gpt2-xlnet-5ee80608e961
    from summarizer import TransformerSummarizer

    result = ""
    logging.debug("text")
    logging.debug(text)
    # stop_words = list(get_stop_words("english"))
    # python_sws = ["self", "def", "args", "true", "false"]
    # SM_STOP_WORDS.extend(python_sws)
    placeholder_count = 0
    i = 0
    it_SM_STOP_WORDS = iter(SM_STOP_WORDS)
    len_stop_words = len(SM_STOP_WORDS)
    while i < len_stop_words + 1 + 10:
        print(i)
        if i not in range(1, 11):
            try:
                word = next(it_SM_STOP_WORDS)
                placeholder_count, text = single_placeholder(placeholder_count, text, rf"\b{word}\b", f" *{i} ")
            except StopIteration:
                print("Finished iterating stop words")
        i += 1
    logging.debug("text with placeholder: %s", text[0:1000])
    GPT2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    # full = ''.join(GPT2_model(body, min_length=60))
    # print(full)
    # result = model(text, ratio=0.2)  # Specified with ratio
    # result = model(text, num_sentences=4)
    if args.ush:
        sentences = GPT2_model(text, num_sentences=max(1, math.ceil(4.0 / num_sub_docs)), return_as_list=True)
    elif args.sh:
        sentences = GPT2_model(text, num_sentences=math.ceil(30.0 / num_sub_docs), min_length=40, return_as_list=True)
        # result_sents = SentenceHandler().process(result)
        print("Result before reverting placeholders:", sentences)
    elif args.me or total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
        sentences = GPT2_model(text, num_sentences=math.ceil(60.0 / num_sub_docs), return_as_list=True)
    else:
        no_sentences = math.ceil(5 * total_doc["line_count"] / 100.0 / num_sub_docs)
        sentences = GPT2_model(text, num_sentences=no_sentences, return_as_list=True)
    for sentence in sentences:
        result += sentence + "\n"
    result = revert_placeholders(result)
    return result


@profile
# def apply_bert(args, text, num_sub_docs, total_doc):
#     from summarizer import Summarizer
#     from summarizer import Summarizer

#     result = ""
#     logging.debug("text")
#     logging.debug(text)
#     # stop_words = list(get_stop_words("english"))
#     # python_sws = ["self", "def", "args", "true", "false"]
#     # SM_STOP_WORDS.extend(python_sws)
#     placeholder_count = 0
#     i = 0
#     it_SM_STOP_WORDS = iter(SM_STOP_WORDS)
#     len_stop_words = len(SM_STOP_WORDS)
#     while i < len_stop_words + 1 + 10:
#         print(i)
#         if i not in range(1, 11):
#             try:
#                 word = next(it_SM_STOP_WORDS)
#                 placeholder_count, text = single_placeholder(placeholder_count, text, rf"\b{word}\b", f" *{i} ")
#             except StopIteration:
#                 print("Finished iterating stop words")
#         i += 1
#     logging.debug("text with placeholder: %s", text[0:1000])
#     model = Summarizer()
#     # result = model(text, ratio=0.2)  # Specified with ratio
#     # result = model(text, num_sentences=4)
#     if args.ush:
#         result = model(text, num_sentences=max(1, math.ceil(4.0 / num_sub_docs)))
#     elif args.sh:
#         result = model(text, num_sentences=math.ceil(30.0 / num_sub_docs), min_length=40)
#         # result_sents = SentenceHandler().process(result)
#         print("Result before reverting placeholders:", result)
#     elif args.me or total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
#         result = model(text, num_sentences=math.ceil(60.0 / num_sub_docs))
#     else:
#         no_sentences = math.ceil(5 * total_doc["line_count"] / 100.0 / num_sub_docs)
#         result = model(text, num_sentences=no_sentences)
#     result = revert_placeholders(result)
#     return result


def extract_doc(args, file):
    doc = open(file).read()
    logging.debug(file)
    doc = remove_stop_words(doc)
    placeholder_count, doc_with_placeholders = put_placeholders(args, doc)
    # doc_without_placeholders = re.sub(r"\s\*\d\s", "", doc_with_placeholders)
    doc_properties = calc_doc_properties(doc)
    doc_dict = {"doc": doc, "doc_with_placeholders": doc_with_placeholders}
    doc_dict.update(doc_properties)
    doc_dict.update({"placeholder_count": placeholder_count})
    return doc_dict


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


def generate_high_priority(doc):
    increase_weight_phrases = [r"\[#A", r"\[#B"]
    pattern = re.compile(r"^(.*(" + r"|".join(increase_weight_phrases) + r")\b.*)$", re.IGNORECASE | re.MULTILINE,)
    matches = re.findall(pattern, doc)
    high_priority_doc = ""
    for match in matches:
        high_priority_doc += match[0] + "\n"
    logging.debug(high_priority_doc)
    return high_priority_doc


def add_many_copies_to_add_weight(doc):
    lines = doc.split("\n")
    len_lines = len(lines)
    line_dictionary = dict.fromkeys(lines, "")

    increase_weight_phrases = [r"\[#A", r"\[#B"]
    pattern = re.compile(r"^(.*(" + r"|".join(increase_weight_phrases) + r")\b.*)$", re.IGNORECASE | re.MULTILINE,)
    matches = re.findall(pattern, doc)
    len_matches = len(matches)
    # 10 lines is minimum according to gensim src
    logging.debug("len_lines %s len_matches %s" % (len_lines, len_matches))
    iter_random = max((len_lines / len_matches / 10.0), 10)
    for match in matches:
        for i in range(iter_random + 1):
            line_dictionary[lines[rl]] = match[0]

    weighted_doc = ""
    for line, next_line in line_dictionary.items():
        weighted_doc += line + "\n" + next_line + "\n"

    # print("weighted ", weighted_doc)
    return weighted_doc


def trim_sentences_to_max_occur(doc):
    sentence_list = doc.split("\n")
    unique_sentences = []
    sentence_dict = {}
    for sentence in sentence_list:
        if sentence not in sentence_dict:
            sentence_dict[sentence] = 1
            unique_sentences.append(sentence)
        elif sentence_dict[sentence] < 2:
            unique_sentences.append(sentence)
            sentence_dict[sentence] += 1
    return "\n".join(unique_sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarizes Doc")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-f", "--files", help="Glob pattern for files")
    parser.add_argument("-r", "--summarize", action="store_true", help="Summarize")
    parser.add_argument("--kw", action="store_true", help="Find keywords")
    parser.add_argument("--ush", action="store_true", help="Ultra short summary (600 words)")
    parser.add_argument("--sh", action="store_true", help="Short summary (600 words)")
    parser.add_argument("--me", action="store_true", help="Medium summary (1600 words)")
    parser.add_argument(
        "-o",
        "--org",
        action="store_true",
        help="Save summary output as .org, add weights when summarizing org mode docs",
    )
    parser.add_argument("-l", "--output", help="output location")
    parser.add_argument("--lg", "--long", action="store_true", dest="long", help="Take longer time")
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
    parser.add_argument("-m", "--mode", choices=("slr", "lr", "lsa", "luhn", "bert"))
    parser.add_argument(
        "--dw", "--deweb", dest="deweb", action="store_true", help="Deprioritize web specific words like github, www"
    )
    parser.add_argument("--auto", dest="auto", action="store_false", help="Automatically determine summary length")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logging.getLogger("gensim").setLevel(logging.INFO)
    logging.getLogger("binaryornot").setLevel(logging.INFO)
    if args.deweb:
        SM_STOP_WORDS.extend(["com", "github", "www"])

    main(args)
