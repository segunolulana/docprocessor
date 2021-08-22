#!/usr/local/bin/pythons
r"""
Text Summarization
==================

Demonstrates summarizing text by extracting the most important sentences from it.

"""
from collections import Counter
import os
import re
from gensim.summarization import keywords
import glob
from os.path import expanduser
import logging
from pprint import pprint, pformat
from gensim.summarization import summarize
from gensim.summarization import mz_entropy
from gensim.summarization import mz_keywords
import argparse
import datetime
from binaryornot.check import is_binary
import sys
import random
import pandas as pd
import time
from concurrent.futures.process import ProcessPoolExecutor
import spacy
import pytextrank

sys.path.append(expanduser("~/Utilities"))
# import summ_utils


# Note https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/
# ROUGE scores for every summary is the maximum ROUGE score amongst the five (individual gold summary) scores
# For Gensim TextRank, the count of words in the output summary, word_count was set to 75.
# For Sumy-LSA and Sumy-Lex_rank the count of sentences in the output summary(sentence_count) was set to 2.
# Luhn's algorithm had the lowest score as Sumy can't change the word limit
# A future direction was to compare Gensim's TextRank implementation with Paco Nathan's PyTextRank


# Ex 1: run_summarization.py --kw -r -f "ThirtyScreenshots/ss*.txt" --tfs 3 --tfe 13 -t -s 2020-09-21 -e 2020-09-25
# Ex 2: run_summarization.py --kw -r -f "Dropbox/orgnotes/mygtd*.org" -o -p org_sum -l ~/Dropbox/org_summaries -a
# @profile


def main(args):
    ext = "txt"
    if args.org:
        ext = "org"
    if args.output_prefix:
        output_lcn = "%s_%s_ptr_sum.%s" % (args.output_prefix, time.strftime("%Y%m%d_%H%M_%S"), ext,)
        key_output_lcn = "%s_%s_key.%s" % (args.output_prefix, time.strftime("%Y%m%d_%H%M_%S"), ext,)
    else:
        output_lcn = "%s_ptr_sum.%s" % (time.strftime("%Y%m%d_%H%M_%S"), ext)
        key_output_lcn = "%s_key.%s" % (time.strftime("%Y%m%d_%H%M_%S"), ext)
    if args.output:
        output_lcn = "%s/%s" % (os.path.expanduser(args.output), output_lcn)
        key_output_lcn = "%s/%s" % (os.path.expanduser(args.output), key_output_lcn)
    if args.summarize:
        outF = open(output_lcn, "a")
    keyOutF = open(key_output_lcn, "a")
    file_at_limit_reached, reached_limit, total_doc = extract_docs(args)

    pprint("Remove date and time...")
    time_re = re.compile(r".*(([01]\d|2[0-3]):([0-5]\d)|24:00)\]")
    text = time_re.sub("", total_doc["doc_with_placeholders"])
    text = "\n".join(
        [line for line in text.split("\n") if line.strip() != ""]
    )  # pytextrank shows too many empty lines without this
    # text = re.sub(r"\n", r". *9\n", text)

    # TODO Remove text = remove_stop_words(text)
    # text = trim_sentences_to_max_occur(text)
    priority_text = generate_high_priority(text)
    priority_text_total_doc = total_doc.copy()
    priority_text_total_doc["doc_with_placeholders"] = priority_text
    priority_text_total_doc.update(calc_doc_properties(priority_text))

    date_removed_total_doc = total_doc.copy()
    date_removed_total_doc["doc_with_placeholders"] = text
    date_removed_total_doc.update(calc_doc_properties(text))

    # https://github.com/DerwenAI/pytextrank/blob/master/example.py
    nlp = spacy.load("en_core_web_sm")
    # add PyTextRank to the spaCy pipeline
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    # placeholders don't work well here as even * is significant
    # TODO Add stop_words here
    # nlp.Defaults.stop_words -= {"TODO", "*"}
    nlp.vocab["*"].is_stop = True
    nlp.vocab["TODO"].is_stop = True

    tr.load_stopwords(expanduser("~") + "Utilities/stop.json")
    print("Checking stop word implementation")
    print(nlp.vocab["*"].is_stop)
    print(nlp.vocab["TODO"].is_stop)

    nlp_doc = nlp(text)

    # Must be done before customized sentence summarization that emphasizes words in org mode
    if args.kw:
        print("Printing keywords...")
        keyOutF.write("Printing keywords...\n")
        if args.only_high_priority:
            kw_result = summarize_as_keywords(nlp_doc)
        else:
            kw_result = summarize_as_keywords(nlp_doc)
        col_width = max(len(kw_one[0]) for kw_one in kw_result) + 7 + 2
        num_columns = 3
        for count, item in enumerate(kw_result[0:300], 1):
            print((item[0] + " " + format(item[1], ".5f")).ljust(col_width), sep=" ", end="")
            if count % num_columns == 0:
                print()
        print("Truncated for space...")
        keyOutF.write(pformat(kw_result))
        keyOutF.write("\n")

    if args.only_high_priority:
        summarize_as_mz_keywords(priority_text, total_doc["line_count"], keyOutF)
    else:
        summarize_as_mz_keywords(text, total_doc["line_count"], keyOutF)

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
                args, file_at_limit_reached, outF, reached_limit, priority_text, total_doc, kw_result, nlp_doc
            )
        else:
            summarize_as_sentences(
                args, file_at_limit_reached, outF, reached_limit, text, total_doc, kw_result, nlp_doc
            )
        outF.close()
    keyOutF.close()

    # (the default is 20%).
    # pprint(summarize(text, ratio=0.5))
    # pprint(summarize(text, word_count=50))


def extract_docs(args, total_doc):
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
        placeholder_count, doc_with_placeholders = put_placeholders(doc)
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
        files_filtered = sorted(set(glob.glob(args.files, recursive=True)))
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
    # stop_phrases = []
    stop_phrases = []
    new_sps = ["id", "overview"]
    stop_phrases.extend(new_sps)
    new_sps2 = ["try", "probably", "mon", "tue", "thu", "fri", "look into", "use", r"mygtd\d*"]
    stop_phrases.extend(new_sps2)
    pattern = re.compile(r"\b((" + r")|(".join(stop_phrases) + r"))\b", re.IGNORECASE)
    text = pattern.sub("", text)
    phrases_without_boundaries = [":properties:", ":end:", ":created:"]
    pattern = re.compile(r"((" + r")|(".join(phrases_without_boundaries) + r"))", re.IGNORECASE)
    text = pattern.sub("", text)
    # Can't use placeholders as all text is significant
    deprioritize = ["\*", "TODO"]
    pattern = re.compile(r"((" + r")|(".join(deprioritize) + r"))", re.IGNORECASE)
    text = pattern.sub("", text)
    remove_lines_with_phrases = ["www.google.com", "DONE", "HOLD", "CANCELLED", ":id"]
    pattern = re.compile(r"^.*(" + r"|".join(remove_lines_with_phrases) + r")(\b|:).*$", re.IGNORECASE | re.MULTILINE,)
    text = pattern.sub("", text)
    return text


# TODO Remove as placeholders are significant with pytextrank
def put_placeholders(text):
    placeholder_count = 0

    # Needed if you want document to split on new line and not on full stop e.g on TODO Bookmark plus (viz. auto bookmarking)
    # c0 = len(re.findall(r"\.", text))
    # print("%s ." % c0)
    # placeholder_count += c0
    # text = re.sub(r"\.", " *1 ", text)

    # c1 = len(re.findall(r"\-", text))
    # print("%d -" % c1)
    # placeholder_count += c1
    # text = re.sub(r"\-", " *2 ", text)
    # c2 = len(re.findall(r"\/", text))
    # print("%d /" % c2)
    # placeholder_count += c2
    # text = re.sub(r"\/", " *3 ", text)
    # c3 = len(re.findall("https", text))
    # print("%d https" % c3)
    # placeholder_count += c3
    # text = re.sub(r"https", " *4 ", text)
    # text = re.sub(r"TODO", " *5 ", text)
    # text = re.sub(r"PERSONAL", " *6 ", text)
    # c4 = len(re.findall("http", text))
    # print("%d http" % c4)
    # placeholder_count += c4
    # text = re.sub(r"http", " *7 ", text)
    c5 = len(re.findall(r"\*", text))
    print("%s ." % c5)
    placeholder_count += c5
    text = re.sub(r"\*", "*8 ", text)
    return placeholder_count, text


def summarize_as_mz_keywords(text, total_line_count, keyOutF):
    # #
    # # Montemurro and Zanette's entropy based keyword extraction algorithm
    # # -------------------------------------------------------------------
    # #
    # # `This paper <https://arxiv.org/abs/0907.1558>`__ describes a technique to
    # # identify words that play a significant role in the large-scale structure of a
    # # text. These typically correspond to the major themes of the text. The text is
    # # divided into blocks of ~1000 words, and the entropy of each word's
    # # distribution amongst the blocks is caclulated and compared with the expected
    # # entropy if the word were distributed randomly.
    # #
    # text=requests.get("http://www.gutenberg.org/files/49679/49679-0.txt").text
    # print(mz_keywords(text,scores=True,threshold=0.001))
    # ###############################################################################
    # # By default, the algorithm weights the entropy by the overall frequency of the
    # # word in the document. We can remove this weighting by setting weighted=False
    # #
    print("Printing mz_keywords...")
    keyOutF.write("Printing mz_keywords...\n")
    if total_line_count < 3000:
        result_mz_kw = mz_entropy.mz_keywords(text, scores=True, weighted=False, threshold="auto")
    else:
        result_mz_kw = mz_keywords(text, scores=True, weighted=True, threshold=0.001)
    if len(result_mz_kw) == 0:
        print("Re-summarizing mz_keywords as threshold is too small")
        keyOutF.write("Re-summarizing mz_keywords as threshold is too small")
        result_mz_kw = mz_keywords(text, scores=True, weighted=True)
    logging.debug("result_mz_kw: %s" % result_mz_kw)
    result_mz_kw_top = pd.Series(dict(result_mz_kw)).nlargest(300)
    print(result_mz_kw_top.to_string())
    keyOutF.write(result_mz_kw_top.to_string())
    # print("result_mz_kw ", result_mz_kw)
    # result_mz_kw_top.plot.barh()
    # ###############################################################################
    # # When this option is used, it is possible to calculate a threshold
    # # automatically from the number of blocks
    # #
    # print(mz_keywords(text, scores=True, weighted=False, threshold="auto"))
    # ###############################################################################
    # # The complexity of the algorithm is **O**\ (\ *Nw*\ ), where *N* is the number
    # # of words in the document and *w* is the number of unique words.
    # #


def summarize_as_keywords(nlp_doc):
    result = ""
    # examine the top-ranked phrases in the document
    # for p in doc._.phrases[0:200]:
    for p in nlp_doc._.phrases[0:600]:
        result += "{:.4f} {:5d}  {}\n".format(p.rank, p.count, p.text)
        if args.debug:
            result += pformat(p.chunks[0:5]) + "\n"

    result += "\n----\n"
    return result


def summarize_as_sentences(args, file_at_limit_reached, outF, reached_limit, text, total_doc, kw_result, nlp_doc):
    if args.org:
        # text = add_many_copies_to_add_weight(text)
        priority_re = re.compile(r"\[\#")
        text = priority_re.sub("[Priority ", text)
        if args.debug:
            pprint(text[:1500])
    pprint("Summarizing text...")
    # Aim 1MB each section of text
    result = ""
    summarize_ratio = ""
    print("placeholder_count:%d" % total_doc["placeholder_count"])

    # In example, it was to summarize the document based on the top 15 phrases,
    # yielding the top 5 sentences...

    # for sent in doc._.textrank.summary(limit_phrases=180, limit_sentences=60):
    for sent in nlp_doc._.textrank.summary(limit_sentences=60):
        # result += pformat(sent) + "\n"
        print(sent)

    result += "\n----\n"

    # if args.sh or total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
    #     result = summarize(text, split=False, word_count=600)
    #     summarize_ratio = "600 words"
    # elif args.me or total_doc["word_count"] - total_doc["placeholder_count"] <= 32000:
    #     result = summarize(text, split=False, word_count=1600)
    #     summarize_ratio = "1600 words"
    # else:
    #     result = summarize(text, split=False, ratio=0.05)
    #     summarize_ratio = "5%"
    result = revert_placeholders(result)
    # # Duplicates still come even after trimming original passages
    # result = trim_sentences_to_max_occur(result)
    # if args.append_keywords:
    #     lines = result.split("\n")
    #     result = ""
    #     keywords_exclude = set(["com", "github", "windows"])
    #     kw_result_temp = []
    #     for kw in kw_result:
    #         if kw[0] not in keywords_exclude:
    #             kw_result_temp.append(kw)
    #     kw_result = kw_result_temp
    #     for line in lines:
    #         kws = []
    #         lcv = 0
    #         for kw in kw_result:
    #             if kw[0] in line.lower():
    #                 kws.append(kw[0])
    #                 lcv += 1
    #             if lcv == 4:
    #                 break
    #         kws = ",".join(kws)
    #         line_with_kws = line + "   " + kws + "\n"
    #         result += line_with_kws
    print(result)
    outF.write(result)
    print("Original character count: ", total_doc["character_count"])
    outF.write("Original character count: %s\n" % total_doc["character_count"])
    print("Original word count: ", total_doc["word_count"])
    outF.write("Original word count: %s\n" % total_doc["word_count"])
    print("Original line count: ", total_doc["line_count"])
    outF.write("Original line count: %s\n" % total_doc["line_count"])
    print("Summary word count: ", len(result.split()))
    outF.write("Summary word count: %s\n" % len(result.split()))
    print("Summary line count: ", len(result.split("\n")))
    outF.write("Summary line count: %s\n" % len(result.split("\n")))
    print("Summary ratio: ", summarize_ratio)
    outF.write("Summary ratio: %s\n" % summarize_ratio)
    if reached_limit:
        print("Lines limit was reached at ", file_at_limit_reached)
        outF.write("Lines limit was reached at %s\n" % file_at_limit_reached)
    print("temp original word count: %d" % len(text.split()))


def revert_placeholders(result):
    priority_symbols_re = re.compile(r"\[Priority ")
    result = priority_symbols_re.sub(r"[#", result)
    result = re.sub(r"\*1\s", ".", result)
    result = re.sub(r"\s\*2\s", "-", result)
    result = re.sub(r"\s\*3\s", r"/", result)
    result = re.sub(r"\s\*4\s", r"https", result)
    result = re.sub(r"\*5", r"TODO", result)  # pytextrank trims leading spaces
    result = re.sub(r"\s\*6\s", r"PERSONAL", result)
    result = re.sub(r"\s\*7\s", r"http", result)
    return result


def extract_doc(file):
    doc = open(file).read()
    logging.debug(file)
    doc = remove_stop_words(doc)
    placeholder_count, doc_with_placeholders = put_placeholders(doc)
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
            rl = random.randint(0, len_lines - 1)
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
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logging.getLogger("gensim").setLevel(logging.INFO)
    logging.getLogger("binaryornot").setLevel(logging.INFO)

    main(args)
