import re
import argparse
import glob
import logging
from collections import Counter
from string import punctuation
import os
from binaryornot.check import is_binary
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from itertools import chain, combinations
from sci_summ_utils import work_on_csv_drop_consecutive
import datetime
from copy import copy

import builtins

try:
    profile = builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func

# E.g python ~/Utilities/generate_freq_dist.py -f "TimeSink/*" --ct csv --cc 1 --tfs 0 --tfe -4 -t -s "2020-05-03" -e "2020-05-11"
# Took, along with ss, 7 min, without ss, 20s for 16229 words, 3731 lines
# After using combination instead of n*n, took 82s for 18035 words, 3793 lines, though this might be more due to flushing buffer
# TODO Remove speed loss by flushing buffer to shell?


# https://stackoverflow.com/a/48316322
def n_grams(seq, n=1):
    """Returns an iterator over the n-grams given a list_tokens"""
    shift_token = lambda i: (el for j, el in enumerate(seq) if j >= i)
    shifted_tokens = (shift_token(i) for i in range(n))
    tuple_ngrams = zip(*shifted_tokens)
    return tuple_ngrams # if join in generator : (" ".join(i) for i in tuple_ngrams)

def range_ngrams(list_tokens, ngram_range=(1, 5)):
    """Returns an iterator over all n-grams for n in range(ngram_range) given a list_tokens."""
    return chain(*(n_grams(list_tokens, i) for i in range(*ngram_range)))

@profile
def main():
    # os.environ["SPACY_WARNING_IGNORE"] = "W008"
    # nlp = spacy.load('en')
    # nlp = spacy.load('en_vectors_web_lg')
    disable = ['tagger', 'parser', 'ner']
    nlp = spacy.load('en_core_web_md', disable=disable)
    # print(nlp.pipe_names)
    # Can enable icl and split words with e.g gsplit - C 4M - -additional-suffix = .txt keys.txt keys
    if args.icl:
        nlp.max_length = 4200000
    # TODO replace with set
    sws = list(STOP_WORDS)
    new_sws = ['properties', 'end', 'created', 'id', 'https', 'slash']
    sws.extend(new_sws)
    new_sws2 = ['try', 'probably', "mon", "tue", "thu", "fri"]
    sws.extend(new_sws2)

    for w in sws:
        nlp.Defaults.stop_words.add(w)
        nlp.vocab[w].is_stop = True

    i = 0
    doc_contents = [""]
    lcv1 = 0
    doc_lengths = [0]
    total_doc_length = 0
    # TODO Rewrite to use set
    excludes = ['node_modules', 'stats-100919.json', '.svg',
                'themes', 'components', 'schema', 'mobile-ios']
    excludes.extend(['.xcassets', 'assets', 'tmp', 'package-lock.json'])
    files_unfiltered = sorted(glob.glob(os.path.expanduser(args.files), recursive=True))
    excludes = set(excludes)
    files_filtered = []
    for file in files_unfiltered:
        if file not in excludes:
            files_filtered.append(file)
    time_filtered = []
    if args.timestamped:
        for file in files_filtered:
            start_time = "0000-00-00"
            end_time = datetime.datetime.now().strftime('%Y-%m-%d')
            if args.start:
                start_time = args.start
            if args.end:
                end_time = args.end
            date_part = (os.path.basename(file))[int(args.tfs):int(args.tfe)]
            if date_part >= start_time and date_part < end_time:
                time_filtered.append(file)
    else:
        time_filtered = copy(files_filtered)
    for file in time_filtered:   # Iterate over the files
        if os.path.isfile(file) and not is_binary(file):
            try:
                if args.ct in ('tsv', 'csv'):
                    contents = work_on_csv_drop_consecutive(file, args.ct, args.cc)
                else:
                    contents = open(file).read()  # Load file contents
            except UnicodeDecodeError:
                pass

            # Okay to not use 1 million char limit since not using spacy's parser or NER
            content_length = len(contents)
            if args.icl or doc_lengths[i] + content_length <= nlp.max_length:
                total_doc_length = explain_content_length(content_length, contents, doc_contents, doc_lengths, file, i,
                                                          lcv1, total_doc_length)
            else:
                print("%s was ignored" % file)
                # The next file is ignored
                break
        else:
            continue

    # Build Word Frequency
    # word.text is tokenization in spacy

    words = []
    unique_tokens_text = set()
    unique_tokens = []
    all_doc_content = ""
    i = 1
    for doc_content in doc_contents:
        all_doc_content += doc_content
        i += 1
    # for doc_content in doc_contents:
    # docx = nlp(doc_content)
        # print("Finished nlp of doc ", i)

    slash_re = re.compile(r"/")  # urls
    all_doc_content = slash_re.sub(" slash ", all_doc_content)
    docx = nlp(all_doc_content)
    print("Finished nlp of doc")

    for token in docx:
        if token.text not in sws and not token.is_punct \
                and not (token.is_space or token.is_digit or token.is_stop):
            words.append(token.text)
            if token.text not in unique_tokens_text:
                unique_tokens.append(token)
                unique_tokens_text.add(token.text)
        # all_doc_content += doc_content
        # print("Finished nlp of doc ", i)
        # i += 1
    print()
    print("Printing most common phrases...")
    phrases_freq = Counter(words)
    common_phrases = phrases_freq.most_common(200)
    for cw in common_phrases:
        if cw[1] > 1:
            print(cw)

    unique_tokens.sort(key=lambda x: x.text)

    if args.ss:
        show_similar_tokens(phrases_freq, unique_tokens)

    slash_re = re.compile(r"\sslash\s")  # urls
    all_doc_content = slash_re.sub(r"/", all_doc_content)

    input_list = all_doc_content.split("\n")
    min_ngram = int(args.min_ngram)
    ngrams_list = list(range_ngrams(input_list, ngram_range=(min_ngram, 5)))

    print()
    print("Printing ngrams...")
    phrases_freq = Counter(ngrams_list)
    # common_phrases = phrases_freq.most_common(200)
    K = 19
    common_phrases = [(ele, cnt) for ele, cnt in phrases_freq.items() if cnt > K]
    common_phrases.sort(key=lambda x: x[1], reverse=True)
    for cw in common_phrases:
        print(cw)
    print("total_doc_length: ", total_doc_length)


def explain_content_length(content_length, contents, doc_contents, doc_lengths, file, i, lcv1, total_doc_length):
    doc_lengths[i] += content_length
    total_doc_length += content_length
    # All should be lowercase
    doc_contents[i] += contents.lower()
    lcv1 += 1
    print(lcv1, ' ', end='', flush=True)
    if args.debug:
        print(
            "doc_lengths[", str(i), "]: ", doc_lengths[i], ' ', end='', flush=True)
        print(file, ' ', end='', flush=True)
        if lcv1 < 5:
            print("doc_contents[", str(i), "]: ", doc_contents[i],
                  ' ', end='', flush=True)
    return total_doc_length


@profile
def show_similar_tokens(phrases_freq, unique_tokens):
    time_re = re.compile(r'^(([01]\d|2[0-3]):([0-5]\d)|24:00)$')
    exclude_similarity = ['jenkins']
    print()
    print("Printing similar words...")
    cmbs = list(combinations(unique_tokens, 2))
    present_token = ""
    for token1, token2 in cmbs:
        # for token1 in unique_tokens:
        # logging.debug(token1)
        token1_string = str(token1)
        if token1_string in exclude_similarity or bool(time_re.match(token1_string)):
            continue
        # if word_freq[token1.text] > 2:
        if phrases_freq[token1.text] > 4 and token1.vector_norm:
            # for token2 in unique_tokens:
            token2_string = str(token2)
            if token2_string in exclude_similarity or bool(time_re.match(token2_string)):
                continue
            this_similarity = token1.similarity(token2)
            if this_similarity > 0.55 and token2.vector_norm:
                if token1.text != present_token:
                    print()
                    print()
                    present_token = token1.text
                print(token1.text, token2.text, token1.similarity(token2), ' ', end='')


parser = argparse.ArgumentParser(
    description='Shows Freq Dist')
parser.add_argument('-f', '--files', required=True,
                    help='Glob pattern for files')
parser.add_argument(
    '-d', '--debug', action='store_true', help='Debug mode')
parser.add_argument(
    '--ct', help='csvtype, csv or tsv')
parser.add_argument(
    '--cc', help='csv column')
parser.add_argument(
    '--ss', action='store_true', help='show similar words')
parser.add_argument(
    '--icl', action='store_true', help='increase character count limit')
parser.add_argument(
    '--timestamped', '-t', action='store_true', help='timestamped like timesink')
parser.add_argument(
    '--tfs', help='time format substring start')
parser.add_argument(
    '--tfe', help='time format substring end')
parser.add_argument('-s', '--start')
parser.add_argument('-e', '--end')
parser.add_argument('-m', '--min_ngram', default='3')
args = parser.parse_args()
if args.debug:
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

main()

