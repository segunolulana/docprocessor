"""
     Initial Author: Gaetano Rossiello
     Email: gaetano.rossiello@uniba.it
"""
from text_summarizer import base
import numpy as np
import logging
import subprocess
from icecream import ic

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
except ImportError:
    output = subprocess.check_output("uname -o", shell=True).decode('utf-8')
    if "Android" in output:  # Only for termux. PyDroid 3 allows much easier installation
        logging.warning("scikit-learn is not installed")
    else:
        raise


class CentroidBOWSummarizer(base.BaseSummarizer):
    def __init__(
        self,
        language='english',
        preprocess_type='nltk',
        stopwords_remove=True,
        length_limit=10,
        debug=False,
        topic_threshold=0.1,
        sim_threshold=0.6,
        additional_stopwords=None,
    ):
        # https://aclanthology.org/W17-1003.pdf. Using topic and similarity threshold here

        super().__init__(language, preprocess_type, stopwords_remove, length_limit, debug, additional_stopwords)
        self.debug = False
        ic(logging.getLevelName(logger.getEffectiveLevel()))
        if logging.getLevelName(logger.getEffectiveLevel()) == "DEBUG":
            self.debug = True
        self.topic_threshold = topic_threshold
        self.sim_threshold = sim_threshold
        return

    def summarize(self, text, limit_type='word', limit=100, split=True):
        raw_sentences = self.sent_tokenize(text)
        clean_sentences = self.preprocess_text(text)

        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(clean_sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0

        sentences_scores = []
        for i in range(tfidf.shape[0]):
            score = base.similarity(tfidf[i, :], centroid_vector)
            sentences_scores.append((i, raw_sentences[i], score, tfidf[i, :]))

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)

        count = 0
        sentences_summary = []
        for s in sentence_scores_sort:
            if count > limit:
                break
            include_flag = True
            for ps in sentences_summary:
                sim = base.similarity(s[3], ps[3])

                # print(s[0], ps[0], sim, s[1], ps[1])
                # self.debug and ic(s[0], ps[0], sim, s[1], ps[1])
                if sim > self.sim_threshold:
                    include_flag = False
            if include_flag:
                # print(s[0], s[1])
                sentences_summary.append(s)
                if limit_type == 'word':
                    words = s[1].split()
                    if len(words) < 10:
                        logging.warn(f"{s[1]} with score {s[2]} ignored in word count since it is too short")
                    else:
                        count += len(words)
                else:
                    count += len(s[1])

        if split:
            summary = [(s[2], s[1]) for s in sentences_summary]
        else:
            summary = "\n".join([s[1] for s in sentences_summary])
        return summary
