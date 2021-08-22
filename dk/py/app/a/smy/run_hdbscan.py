import time
import argparse
import logging
import os

from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from icecream import ic
import umap


try:
    # Works from site but not individually
    from .doc_retriever import DocRetriever
    from .run_sum_helper import Devnull
except ImportError:
    from doc_retriever import DocRetriever
    from run_sum_helper import Devnull


def runClustering(args, total_doc, min_samples):
    print('Clustering all documents with min_samples=%d' % min_samples)

    # min_samples could be a somewhat important
    # Default metric='euclidean', cluster_selection_method='eom'
    db = HDBSCAN(min_samples=min_samples, min_cluster_size=4)

    # Time this step.
    t0 = time.time()

    corpus = [line.strip().lower() for line in total_doc["doc"].splitlines()]

    n_neighbors = 15
    n_components = 5
    random_state = 42
    embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    corpus_embeddings = embedder.encode(corpus, show_progress_bar=True, batch_size=8)
    ic(corpus_embeddings.shape)  # shows why we need umap to reduce dimensionality
    umap_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=random_state
    ).fit_transform(corpus_embeddings)

    # Cluster
    db.fit(umap_embeddings)

    # Calculate the elapsed time (in seconds)
    elapsed = time.time() - t0
    print("  done in %.3fsec" % elapsed)

    # Get the set of unique IDs.
    cluster_ids = set(db.labels_)

    # Show the number of clusters (don't include noise label)
    print('Number of clusters (excluding "noise"): %d' % (len(cluster_ids) - 1))

    # For each of the clusters...
    for cluster_id in cluster_ids:

        # Get the list of all doc IDs belonging to this cluster.
        cluster_doc_ids = []
        for doc_id in range(0, len(db.labels_)):
            if db.labels_[doc_id] == cluster_id:
                cluster_doc_ids.append(doc_id)

        # # Get the top words in this cluster
        # if cluster_id != -1:
        #     top_words = ssearch.getTopWordsInCluster(cluster_doc_ids)
        # else:
        #     top_words = ssearch.getTopWordsInCluster(cluster_doc_ids, topn=30)

        # print('  Cluster %d: (%d docs) %s' % (cluster_id, len(cluster_doc_ids), " ".join(top_words)))
        print('  Cluster %d: (%d docs)' % (cluster_id, len(cluster_doc_ids)))
        for cluster_doc_id in cluster_doc_ids:
            print(corpus[cluster_doc_id])
        print("-" * 20)


def main():
    """
    Entry point for the script.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", help="Glob pattern for files")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("--icl", action="store_true", help="increase character count limit")
    parser.add_argument("-m", "--min_pts", dest="min_pts", help="Use min_pts")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    args.mode = "gs"  # influences how placeholders replace text

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
    total_doc = doc_retriever.clean_doc(total_doc["doc"])

    # min_samples = 8
    min_samples = 4
    if args.min_pts:
        min_samples = int(args.min_pts)

    runClustering(args, total_doc, min_samples)


main()
