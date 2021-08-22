# https://www.kaggle.com/hamid3731/keyphrase-extraction-and-graph-analysis
import argparse
import logging
import time
from pke.base import LoadFile
from string import punctuation
from nltk.corpus import stopwords
import string
from collections import defaultdict
import os
import sys
import gzip


try:
    # Works from site but not individually
    from .doc_retriever import DocRetriever
    from .run_sum_helper import Devnull
except ImportError:
    from doc_retriever import DocRetriever
    from run_sum_helper import Devnull


def main():
    args = parse_arguments()
    stoplist = list(stopwords.words('english')) + list(string.punctuation)

    frequencies = defaultdict(int)
    delimiter = '\t'

    # initialize number of documents
    nb_documents = 0

    doc_retriever = DocRetriever(args)
    time_str = time.strftime("%y%m%d_%H%M_%S")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    y = time_str[0:2]
    m = time_str[2:4]
    d = time_str[4:6]
    output_dir = os.path.join(output_dir, y, m, d)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outFs = {"outF": Devnull(), "keyOutF": Devnull(), "keyOutOnlyF": Devnull()}
    file_at_limit_reached, reached_limit, total_doc = doc_retriever.extract_docs(outFs)
    # Note: im limit this item becuase is very time consumed task for my notebook
    # limite_dataset_article = 50
    # if limite_dataset_article is not None:
    #     my_df_articles = articles.head(limite_dataset_article)
    # else:
    #     my_df_articles = articles

    output_file = output_dir + '/df.tsv.gz'
    for article in total_doc["doc"].splitlines():
        content = article
        doc = LoadFile()
        doc.load_document(input=content)
        # n is number of n-gram
        doc.ngram_selection(n=3)
        doc.candidate_filtering(stoplist=stoplist)
        # loop through candidates
        for lexical_form in doc.candidates:
            frequencies[lexical_form] += 1
        nb_documents += 1

        if nb_documents % 10 == 0:
            print("{} docs, memory used: {} mb".format(nb_documents, sys.getsizeof(frequencies) / 1024 / 1024))

    # dump the df container
    with gzip.open(output_file, 'wb') as f:
        # add the number of documents as special token
        first_line = '--NB_DOC--' + delimiter + str(nb_documents)
        f.write(first_line.encode('utf-8') + b'\n')
        for ngram in frequencies:
            line = ngram + delimiter + str(frequencies[ngram])
            f.write(line.encode('utf-8') + b'\n')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-f", "--files", help="Glob pattern for files")
    parser.add_argument("--icl", action="store_true", help="increase character count limit")

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(
            format="%(asctime)s : {%(pathname)s:%(lineno)d} : %(levelname)s : %(message)s", level=logging.DEBUG,
        )
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    # logging.getLogger("gensim").setLevel(logging.INFO)
    logging.getLogger("binaryornot").setLevel(logging.INFO)
    return args


if __name__ == "__main__":
    main()
