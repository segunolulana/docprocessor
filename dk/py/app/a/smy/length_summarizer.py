import math

try:
    # Works from site but not individually
    from .calc.keywords_calc import keywords
except ImportError:
    from calc.keywords_calc import keywords


class ushSummarizer:
    def __init__(self, text, STOP_WORDS, algorithm):
        self.text = text
        self.STOP_WORDS = STOP_WORDS
        self.algorithm = algorithm

    def length_summarize(self):
        result = self.algorithm.summarize(self.text, split=True, word_count=75, additional_stopwords=self.STOP_WORDS)
        return result

    def length_kw_summarize(self, kw_output, doc_with_placeholders, index):
        print("Summarising keywords for group %s to 75 words" % index)
        kw_output += "Summarising to 75 words\n"
        kw_result = keywords(doc_with_placeholders, words=75, scores=True, split=True)
        return kw_result, kw_output


class ratioSummarizer:
    def __init__(self, text, STOP_WORDS, percent, num_sub_docs, algorithm):
        self.text = text
        self.STOP_WORDS = STOP_WORDS
        self.percent = float(percent)
        self.algorithm = algorithm
        self.num_sub_docs = num_sub_docs

    def length_summarize(self):
        result = self.algorithm.summarize(
            self.text, split=True, ratio=self.percent * self.num_sub_docs / 100.0, additional_stopwords=self.STOP_WORDS
        )
        return result

    def length_kw_summarize(self, kw_output, doc_with_placeholders, index):
        print("Summarising keywords for group %s by 5% of sentences" % index)
        kw_output += "Summarising by 5% of sentences\n"
        kw_result = keywords(
            doc_with_placeholders, ratio=math.ceiling(0.05 * self.num_sub_docs), scores=True, split=True
        )
        return kw_result, kw_output


class twelveKSummarizer:
    def __init__(self, text, STOP_WORDS, algorithm):
        self.text = text
        self.STOP_WORDS = STOP_WORDS
        self.algorithm = algorithm

    def length_summarize(self):
        result = self.algorithm.summarize(self.text, split=True, word_count=1600, additional_stopwords=self.STOP_WORDS,)
        return result

    def length_kw_summarize(self, kw_output, doc_with_placeholders, index):
        print("Summarising keywords for group %s to 600 words" % index)
        kw_output += "Summarising to 600 words\n"
        kw_result = keywords(doc_with_placeholders, words=600, scores=True, split=True)
        return kw_result, kw_output
