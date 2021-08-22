import logging

try:
    from dk.py.app.a.smy.doc_retriever import DocRetriever
except ImportError:
    from run_sum_helper import DocRetriever


from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from gensim.similarities import Similarity
import argparse


# https://towardsdatascience.com/a-laymans-guide-to-fuzzy-document-deduplication-a3b3cf9a05a7


def main():
    args = parse_arguments()
    doc_retriever = DocRetriever(args)
    _, _, total_doc, doc_dict_ls = doc_retriever.extract_docs_get_list(None)
    # documents = [
    #     "Used SpaceX rocket as-is, buyer must transport.",
    #     "Used SpaceX rocket as-is, buyer must transport.",
    #     "For sale: bulk 100lbs pack of spaghetti noodles",
    #     "Spaghetti noodles for sale — 100lbs bulk pack",
    #     "Pale blue tuxedo, used, good condition. Call 555–555–5555",
    #     "Brand new yellow tuxedo in great condition!"
    # ]
    documents = []
    document_files = []
    for doc_dict in doc_dict_ls:
        documents.append(doc_dict["doc"])
        document_files.append(doc_dict["file"])
    texts = [[text for text in simple_preprocess(doc, deacc=True)] for doc in documents]

    # Build a bigram model to capture every pair of words in the texts
    bigram = Phrases(texts, min_count=1)
    bigram_phraser = Phraser(bigram)
    # The min_count argument specifies how many times the phrase must be found in the corpus to be kept. We're keeping it at 1 since our dataset is so small.

    # Reconvert documents to collection of words/bigrams
    texts_bigrams = [[text for text in bigram_phraser[simple_preprocess(doc, deacc=True)]] for doc in documents]

    dictionary = corpora.Dictionary(texts_bigrams)
    corpus = [dictionary.doc2bow(docString) for docString in texts_bigrams]

    # Build similarity index
    index = Similarity(corpus=corpus, num_features=len(dictionary), output_prefix="on_disk_output")

    # Parse similarities from index
    doc_id = 0
    similar_docs = {}
    for similarities in index:
        similar_docs[doc_id] = list(enumerate(similarities))
        doc_id += 1

    # sim_threshold = 90
    sim_threshold = 99
    if args.sim_threshold:
        sim_threshold = int(args.sim_threshold)
    count_of_similar_docs = 0
    for doc_id, sim_doc_tuples in similar_docs.items():
        for sim_doc_tuple in sim_doc_tuples:
            sim_doc_id = sim_doc_tuple[0]
            sim_score = sim_doc_tuple[1]
            if sim_score >= (sim_threshold / 100.0) and doc_id != sim_doc_id:
                count_of_similar_docs += 1
                print(f"Found similar documents, score of {sim_score:.2f}:")
                if args.preview_docs:
                    print("\t", documents[doc_id][-300:])
                print("\t", document_files[doc_id])
                if args.preview_docs:
                    print("\t", documents[sim_doc_id][-300:], "\n")
                print("\t", document_files[sim_doc_id], "\n")
    if count_of_similar_docs > 0:
        print(f"Found {count_of_similar_docs} similar documents")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Get similar documents")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-f", "--files", help="Glob pattern for files")
    parser.add_argument("--icl", action="store_true", help="increase character count limit")
    parser.add_argument("-s", "--sim_threshold")
    parser.add_argument("-p", "--preview_docs", action="store_true")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(
            format="%(asctime)s : {%(pathname)s:%(lineno)d} : %(levelname)s : %(message)s", level=logging.DEBUG,
        )
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logging.getLogger("gensim").setLevel(logging.INFO)
    logging.getLogger("binaryornot").setLevel(logging.INFO)
    return args


if __name__ == "__main__":
    main()
