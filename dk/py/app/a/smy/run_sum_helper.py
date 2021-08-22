import datetime
from pprint import pprint, pformat
import math


try:
    # Works from site but not individually
    from .summ_utils import RepresentsInt, string_found, remove_empty_lines, split, output_to_file_and_stdout
    from .doc_retriever import DocRetriever
    from .length_summarizer import ushSummarizer, ratioSummarizer, twelveKSummarizer
except ImportError:
    from summ_utils import RepresentsInt, string_found, remove_empty_lines, split, output_to_file_and_stdout
    from doc_retriever import DocRetriever
    from length_summarizer import ushSummarizer, ratioSummarizer, twelveKSummarizer


from enum import Enum
import builtins
import logging
import subprocess
from concurrent.futures.process import ProcessPoolExecutor
import re
import os
import time
from icecream import ic
import io
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
from gensim.summarization.syntactic_unit import SyntacticUnit
import text_summarizer


try:
    import pke
except ImportError:
    logging.warning("pke package not installed")

try:
    from summarizer import TransformerSummarizer
except ImportError:
    logging.warning("bert-extractive-summarizer package not installed")

try:
    import line_profiler
except ImportError:
    logging.warning("line_profiler package not installed")

# from multiprocess import Pool


try:
    # Works from site but not individually
    from .calc.keywords_calc import keywords
    from .calc.summarizer_calc import summarize as gensim_summarize
except ImportError:
    from calc.keywords_calc import keywords
    from calc.summarizer_calc import summarize as gensim_summarize

try:
    # Works from site but not individually
    from .sumy_calc.lex_rank import LexRankSummarizer
    from .sumy_calc.lsa import LsaSummarizer
    from .sumy_calc.luhn import LuhnSummarizer
except ImportError:
    from sumy_calc.lex_rank import LexRankSummarizer
    from sumy_calc.lsa import LsaSummarizer
    from sumy_calc.luhn import LuhnSummarizer

PROFILING = False
try:
    profile = builtins.profile
    PROFILING = True
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


class Summarizer:
    def __init__(self, result_length, args):
        self.result_length = result_length
        self.STOP_WORDS = ["a", "b", "c", "d", "e"]
        self.STOP_WORDS.extend(["like", "better", "blob", "master", "look", "into", "try"])
        self.STOP_WORDS.extend(["probably", "use", "using", "check", "PERSONAL"])
        python_sws = ["self", "def", "args", "true", "false"]
        self.STOP_WORDS.extend(python_sws)
        self.args = args
        if args.deweb:
            self.STOP_WORDS.extend(["com", "github", "www", "t", "co", "bit", "ly", "http", "https"])

        self.STOP_WORDS.extend(["aws", "console"])
        if getattr(args, "stop_words", False):
            doc = open(os.path.expanduser(args.stop_words), 'r').read()
            additional_stopwords = doc.splitlines()
            self.STOP_WORDS.extend(additional_stopwords)

    @profile
    def summarize_all(self, args, file_at_limit_reached, reached_limit, total_doc, outFs):
        kw_output = ""
        kw_large_output = ""
        text = total_doc["doc"]
        print("Time:", datetime.datetime.now())
        if args.divide_and_approximate:
            all_lines, all_text, num_sub_docs, total_doc, word_and_sentence_results = self.divide_and_approximate(
                args, outFs, text, total_doc
            )
        else:
            if total_doc["line_count"] > 10000:
                text, total_doc = self.gsplit_long_doc(args, outFs, text, total_doc)
            algorithm, all_lines, all_text, docs, num_sub_docs, text = self.get_subdocs_and_algorithm(args, text)
            if args.long:
                word_and_sentence_results = [self.apply_summarize(args, text, 1, 1, algorithm)]
            else:
                all_summary_lines = []
                print("%d subdocuments" % num_sub_docs)
                word_and_sentence_results = self.summarize_subdocs(args, docs, num_sub_docs, algorithm)
                for word_and_sentence_result in word_and_sentence_results:
                    for note in word_and_sentence_result[0]:
                        all_summary_lines.append(note[1].text)
                text = "\n".join(all_summary_lines)
                # Final summmarize
                algorithm, all_lines, all_text, docs, num_sub_docs, text = self.get_subdocs_and_algorithm(args, text)
                print("%d subdocuments" % num_sub_docs)
                word_and_sentence_results = self.summarize_subdocs(args, docs, num_sub_docs, algorithm)
                # return all_lines, all_text, num_sub_docs, total_doc, word_and_sentence_results

        if args.complement:
            # TODO Make complement work with sentences (with dot?) when not using gensim
            if args.mode != "gs":
                logging.warning("Only gensim works well with complement presently")
            note_ls_ls = []
            all_summary_lines = []
            for word_and_sentence_result in word_and_sentence_results:
                for note in word_and_sentence_result[0]:
                    all_summary_lines.append(note[1].text)
                kw_output += word_and_sentence_result[2]
                kw_large_output += word_and_sentence_result[3]
            all_summary_lines = set(all_summary_lines)
            note_ls = []
            doc_retriever = DocRetriever(args)
            for line in all_lines:
                line = doc_retriever.remove_org_time(line)
                line = doc_retriever.remove_stop_words(line)
                line = line.strip()
                if line != "" and line not in all_summary_lines:
                    note_ls.append({"line": line, "kws": "", "score": ""})
            note_ls_ls.append(note_ls)
            sent_summ = self.get_sentence_from_note_ls(note_ls)
            print(kw_output)
            outFs["keyOutF"].write(kw_output)
            outFs["keyOutOnlyF"].write(kw_large_output)
            print(sent_summ)
            outFs["outF"].write(sent_summ)
            total_sent_summ = sent_summ + "\n\n"
        else:
            total_sent_summ = ""
            note_ls_ls = []
            for word_and_sentence_result in word_and_sentence_results:
                kw_output += word_and_sentence_result[2]
                kw_large_output += word_and_sentence_result[3]
                if args.append_keywords:
                    note_ls = self.append_keywords_to_sentences(args, word_and_sentence_result)
                else:
                    # TODO: Trim duplicate lines here also
                    note_ls = [
                        {"line": note[1].text, "kws": [], "kw_strs": "", "score": note[0], "keyword_rank": 0}
                        for note in word_and_sentence_result[0]
                    ]  # TODO. Fix. Why tuple?
                note_ls_ls.append(note_ls)
                sent_summ = self.get_sentence_from_note_ls(note_ls)
                print(kw_output)
                outFs["keyOutF"].write(kw_output)
                outFs["keyOutOnlyF"].write(kw_large_output)
                print(sent_summ)
                outFs["outF"].write(sent_summ)
                total_sent_summ += sent_summ + "\n\n"

        self.print_info_one(
            all_text, args, file_at_limit_reached, num_sub_docs, outFs, reached_limit, total_doc, total_sent_summ
        )
        return note_ls_ls

    def print_info_one(
        self, all_text, args, file_at_limit_reached, num_sub_docs, outFs, reached_limit, total_doc, total_sent_summ
    ):
        summarize_ratio = ""
        if self.result_length == RL.sh:
            summarize_ratio = "600 words (short)"
        elif args.me or total_doc["word_count"] <= 12000:
            summarize_ratio = "1600 words (10%?)"
        elif args.percent:
            summarize_ratio = f"{args.percent}%"
        else:
            summarize_ratio = "20%"
        output_to_file_and_stdout("Original character count: %s" % total_doc["character_count"], outFs["outF"])
        output_to_file_and_stdout("Original word count: %s" % total_doc["word_count"], outFs["outF"])
        output_to_file_and_stdout("Original line count: %s" % total_doc["line_count"], outFs["outF"])
        output_to_file_and_stdout("Summary word count: %s" % len(total_sent_summ.split()), outFs["outF"])
        output_to_file_and_stdout("Summary line count: %s" % len(total_sent_summ.split("\n")), outFs["outF"])
        output_to_file_and_stdout("Summary ratio: %s" % summarize_ratio, outFs["outF"])
        output_to_file_and_stdout("Summary line count: %s" % len(total_sent_summ.split("\n")), outFs["outF"])
        if reached_limit:
            output_to_file_and_stdout("Lines limit was reached at %s" % file_at_limit_reached, outFs["outF"])
        output_to_file_and_stdout("Method was %s" % args.mode, outFs["outF"])
        print("temp original word count: %d" % len(all_text.split()))
        if not args.long:
            print("%d subdocuments" % num_sub_docs)

    def divide_and_approximate(self, args, outFs, text, total_doc):
        texts = text.split("\n")
        d_n_a_line_len = 10000
        if args.dal:
            d_n_a_line_len = int(args.dal)
        parts_num = math.ceil(len(texts) / float(d_n_a_line_len))
        slice_start = 0
        all_summary_lines = []
        for part_index in range(parts_num):
            current_slice = texts[slice_start : slice_start + d_n_a_line_len]
            slice_start += d_n_a_line_len
            text = "\n".join(current_slice)
            print(f"Summarizing {d_n_a_line_len} line division {part_index + 1}")

            algorithm, all_lines, all_text, docs, num_sub_docs, text = self.get_subdocs_and_algorithm(args, text)
            if args.long:
                word_and_sentence_results = [self.apply_summarize(args, text, 1, 1, algorithm)]
            else:
                algorithm, all_lines, all_text, docs, num_sub_docs, text = self.get_subdocs_and_algorithm(args, text)
                print("%d subdocuments" % num_sub_docs)
                word_and_sentence_results = self.summarize_subdocs(args, docs, num_sub_docs, algorithm)
            for word_and_sentence_result in word_and_sentence_results:
                for note in word_and_sentence_result[0]:
                    all_summary_lines.append(note[1].text)
        text = "\n".join(all_summary_lines)
        # Final summmarize
        if args.long:
            word_and_sentence_results = [self.apply_summarize(args, text, 1, 1, algorithm)]
        else:
            algorithm, all_lines, all_text, docs, num_sub_docs, text = self.get_subdocs_and_algorithm(args, text)
            print("%d subdocuments" % num_sub_docs)
            word_and_sentence_results = self.summarize_subdocs(args, docs, num_sub_docs, algorithm)
        return all_lines, all_text, num_sub_docs, total_doc, word_and_sentence_results

    def get_subdocs_and_algorithm(self, args, text):
        if args.org:
            # text = add_many_copies_to_add_weight(text)
            priority_re = re.compile(r"\[\#")
            text = priority_re.sub("[Priority ", text)
            if args.debug:
                pprint(text[:1500])
        pprint("Summarizing text...")
        # Aim 1MB each section of text
        text = re.sub(r"\. ", ".\n", text)
        all_lines = text.split("\n")
        all_text = text
        partition_length = 4000
        if args.partition_length:
            partition_length = int(args.partition_length)
        num_sub_docs = math.ceil(len(all_lines) / partition_length)
        docs = split(all_lines, num_sub_docs)
        algorithm = TextRank()
        if args.mode == "luhn":
            algorithm = Luhn()
        elif args.mode == "lsa":
            algorithm = Lsa()
        elif args.mode == "slr":
            algorithm = SumyLexRank()
        elif args.mode == "cwe":
            algorithm = CentroidWordEmbeddings()
        elif args.mode == "cbow":
            algorithm = CentroidBOWEmbeddings()
        elif args.mode == "gpt2m":
            # Took 1024677.96ms for mygtd*.org 4144 lines 19338 words to 34 sentences
            algorithm = GPT2Medium()
        elif args.mode == "bart":
            # Took 249620.70ms for mygtd*.org 4152 lines 19374 words to 35 sentences
            algorithm = Bart()
        return algorithm, all_lines, all_text, docs, num_sub_docs, text

    def gsplit_long_doc(self, args, outFs, text, total_doc):
        filename = self.get_filename_for_split(args)
        logging.info("Splitting document with gnu split...")
        command = ["gsplit", "-l", "10000", "--additional-suffix=.txt", "-", filename]
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = p.communicate(input=text.encode())[0]
        print(filename)
        print(output)
        ic(outFs)
        doc_retriever = DocRetriever(args)
        total_doc = doc_retriever.extract_doc(filename + "aa.txt")
        text = total_doc["doc"]
        return text, total_doc

    @staticmethod
    def get_filename_for_split(args):
        filename = args.files.replace("*", "s")
        filename = filename.replace(".", "d")
        filename = filename.replace(r"/", "_")
        return filename

    def summarize_as_keywords(self, lengthSummarizer, short, medium, total_doc, kw_output, kw_large_output, index):
        # gensim.summarization.keywords fetching different results as it is non-deterministic
        # e.g with pos_filter=('NP') i.e noun phrase
        # https://github.com/RaRe-Technologies/gensim/issues/2586. Aug 9, 2019
        # https://github.com/DerwenAI/pytextrank probably better here?
        # Default gensim.summarization.keywords
        #   .keywords(text, ratio=0.2, words=None, split=False,
        #               scores=False, pos_filter=('NN', 'JJ'), lemmatize=False, deacc=True)
        # NN is noun, singular or mass. JJ is adjective
        kw_result = ""
        # doc_without_placeholders = re.sub(r"\s\*\d\s", "", total_doc["doc_with_placeholders"])
        # doc_properties = calc_doc_properties(doc_without_placeholders)
        doc_with_placeholders = total_doc["doc_with_placeholders"]
        if self.args.kw == "tpr":
            extractor = pke.unsupervised.TopicRank()
            # extractor = pke.unsupervised.TextRank()
            extractor.load_document(doc_with_placeholders)
            extractor.candidate_selection()
            extractor.candidate_weighting()

        if short:
            print("Summarising keywords for group %s to 600 words" % index)
            kw_output += "Summarising to 600 words\n"
            if self.args.kw == "gs":
                # lemmatize is very important else you get something like "firefox addons tested" which occurs
                # only once in text instead of "firefox"?
                kw_result = keywords(doc_with_placeholders, words=600, scores=True, split=True, lemmatize=True)
            elif self.args.kw == "tpr":
                kw_result = extractor.get_n_best(n=600)
        elif medium or total_doc["word_count"] - total_doc["placeholder_count"] <= 32000:
            print("Summarising keywords for group %s to 1600 words" % index)
            kw_output += "Summarising to 1600 words\n"
            if self.args.kw == "gs":
                kw_result = keywords(doc_with_placeholders, words=1600, scores=True, split=True, lemmatize=True)
            elif self.args.kw == "tpr":
                kw_result = extractor.get_n_best(n=1600)
        else:
            kw_result, kw_output = lengthSummarizer.length_kw_summarize(kw_output, doc_with_placeholders, index)
        print("Summary keywords count for group %s: %s" % (index, len(kw_result)))
        kw_output += "Summary keywords count: %s\n" % len(kw_result)
        print("Trimmed doc count: ", len(doc_with_placeholders))
        kw_output += "Trimmed doc count: %s\n" % len(doc_with_placeholders)
        if len(kw_result) > 0:
            for word in self.STOP_WORDS:
                try:
                    self.remove_tuple(kw_result, word)
                except ValueError:
                    pass
            kw_output = self.output_kw_summary(kw_result, kw_output)
            kw_large_output += "%s\n" % pformat(kw_result)
        return kw_result, kw_output, kw_large_output

    def remove_tuple(self, kw_result, word):
        for i, a_kw_result_tuple in enumerate(kw_result):
            if word == a_kw_result_tuple[0]:
                del kw_result[i]

    def output_kw_summary(self, kw_result, kw_output):
        col_width = max(len(kw_one[0]) for kw_one in kw_result) + 7 + 2
        num_columns = 3
        for count, item in enumerate(kw_result[0:60], 1):
            adjusted_kw_result = (item[0] + " " + format(item[1], ".5f")).ljust(col_width)
            kw_output += adjusted_kw_result
            # print(adjusted_kw_result, sep=" ", end="")
            if count % num_columns == 0:
                # print()
                kw_output += "\n"
        # print("Truncated for space...")
        kw_output += "Truncated for space...\n"
        return kw_output

    def append_keywords_to_sentences(self, args, word_and_sentence_result):
        kw_result = word_and_sentence_result[1]
        note_ls = []
        length_of_appended_kw = 4
        length_of_kw = len(kw_result)
        # keywords_exclude = set("windows")
        keywords_exclude = set("")
        kw_result_temp = []
        for kw in kw_result:
            if kw[0] not in keywords_exclude:
                kw_result_temp.append(kw)
        kw_result = kw_result_temp

        sentence_dict = {}
        for sentence_result in word_and_sentence_result[0]:
            # Duplicates still come even after trimming original passages
            line = sentence_result[1].text
            if line not in sentence_dict:
                sentence_dict[line] = 1
            elif sentence_dict[line] == 1:
                sentence_dict[line] += 1
                # TODO Add to database
                continue
            else:
                continue
            note_dict = {"line": line}
            for i in range(length_of_appended_kw):
                note_dict["key" + str(i)] = ""
            kws = []
            kw_strs = []
            lcv = 0
            for i, kw in enumerate(kw_result):
                if string_found(kw[0], line.lower()):
                    kws.append(kw)
                    kw_strs.append("%s(%s)" % (kw[0], i))
                    note_dict["key" + str(lcv)] = kw[0]
                    lcv += 1
                if lcv == length_of_appended_kw:
                    break
            kw_strs = ",".join(kw_strs)
            note_dict["kws"] = kws
            note_dict["kw_strs"] = kw_strs
            note_dict["score"] = sentence_result[0]
            note_ls.append(note_dict)
        # if args.sort_mode == "k":
        srt = {s[0]: i for i, s in enumerate(kw_result)}
        # reverse = not args.reverse
        for i in range(length_of_appended_kw - 1, -1, -1):
            note_ls = sorted(note_ls, key=lambda x: (srt.get(x["key" + str(i)], length_of_kw + 1)))
        # Increasing order for easier view on terminal
        # elif args.sort_mode == "s":
        #     reverse = True if args.reverse else False
        #     note_ls = sorted(
        #         note_ls, key=lambda x: x["score"], reverse=reverse
        #     )  # Increasing order for easier view on terminal
        for rank, note in enumerate(note_ls):
            note.update({"keyword_rank": rank})
        return note_ls

    def get_sentence_from_note_ls(self, note_ls):
        sent_summ = ""
        for note_dict in note_ls:
            line = note_dict["line"]
            line_with_kws = line + " - " + note_dict.get("kw_strs", "") + " - " + str(note_dict["score"]) + "\n"
            sent_summ += line_with_kws
        return sent_summ

    @profile
    def summarize_subdocs(self, args, docs, num_sub_docs, algorithm):
        # Slower but more versatile multiprocess
        # p = Pool(4)
        # i = 0

        # # futures = {}
        # TASKS = []
        # for doc in docs:
        #     logging.debug("debug doc:")
        #     logging.debug(doc[0:1000])
        #     text = "\n".join(doc)
        #     TASKS += (args, text, num_sub_docs, i)
        #     i += 1
        # results = []
        # for future in p.map(self.apply_summarize, TASKS):
        # # for i, future in futures.items():  # So can see exceptions
        #     a_result = future.result()
        #     logging.debug(a_result)
        #     results.append(a_result)
        # return results
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {}
            i = 0
            for doc in docs:
                logging.debug("debug doc:")
                logging.debug(doc[0:200])
                text = "\n".join(doc)
                future = executor.submit(self.apply_summarize, args, text, num_sub_docs, i, algorithm)
                futures[i] = future
                i += 1
            results = []
            for i, future in futures.items():  # So can see exceptions
                a_result = future.result()
                logging.debug(a_result)
                results.append(a_result)
        return results

    def apply_summarize(self, args, text, num_sub_docs, index, algorithm):
        """
            Returns result, kw_result, kw_output, kw_large_output
        """
        if PROFILING:
            p = line_profiler.LineProfiler(self.apply_summarize_work)
            p.enable_by_count()

        kw_large_output, kw_output, kw_result, result, s = self.apply_summarize_work(
            args, index, num_sub_docs, text, algorithm
        )
        if PROFILING:
            p.disable_by_count()
            p.print_stats(stream=s)
            print(s.getvalue())
        return result, kw_result, kw_output, kw_large_output

    def apply_summarize_work(self, args, index, num_sub_docs, text, algorithm):
        result = ""
        doc_retriever = DocRetriever(args)
        total_doc = doc_retriever.clean_doc(text)
        text = total_doc["doc_with_placeholders"]
        print("doc %s placeholder_count:%d" % (index, total_doc["placeholder_count"]))
        if args.percent:
            lengthSummarizer = ratioSummarizer(text, self.STOP_WORDS, args.percent, num_sub_docs, algorithm)
        elif args.ush:  # ush takes priority and order is important
            lengthSummarizer = ushSummarizer(text, self.STOP_WORDS, algorithm)
        elif total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
            lengthSummarizer = twelveKSummarizer(text, self.STOP_WORDS, algorithm)
        else:
            # (the default is 20%).
            lengthSummarizer = ratioSummarizer(text, self.STOP_WORDS, 20, num_sub_docs, algorithm)
        kw_large_output = kw_output = kw_result = ""
        # Must be done before customized sentence summarization that emphasizes words in org mode
        if args.kw and num_sub_docs == 1:
            print("Getting keywords...")
            kw_output = "Printing keywords...\n"
            kw_large_output = ""
            kw_result, kw_output, kw_large_output = self.summarize_as_keywords(
                lengthSummarizer, args.sh, args.me, total_doc, kw_output, kw_large_output, index
            )
        if args.sh:
            short_num_words = 600
            if num_sub_docs != 1:
                short_num_words = short_num_words * 1.8 / num_sub_docs
            result = algorithm.summarize(
                text, split=True, word_count=short_num_words, additional_stopwords=self.STOP_WORDS
            )
        elif args.me:  # TODO Remove. Ush, sh and ratio is sufficient
            result = algorithm.summarize(text, split=True, word_count=1600, additional_stopwords=self.STOP_WORDS)
        else:
            result = lengthSummarizer.length_summarize()
        logging.debug("Result before reverting placeholders:%s", result)
        result = doc_retriever.revert_placeholders(result)
        s = io.StringIO()
        return kw_large_output, kw_output, kw_result, result, s

    def trim_sentences_to_max_occur(self, doc):
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


def generate_summary(args):
    ts = time.time()
    result_length = RL.fp
    if args.sh:
        result_length = RL.sh
    elif args.ush:
        result_length = RL.ush
    summarizer = Summarizer(result_length, args)
    outF, keyOutF, keyOutOnlyF, outFs = prepare_output_variables(args)
    doc_retriever = DocRetriever(args)
    file_at_limit_reached, reached_limit, total_doc = doc_retriever.extract_docs(outFs)
    # TODO Remove text = remove_stop_words(text)
    # text = trim_sentences_to_max_occur(text)
    text = ""
    if args.only_high_priority:
        text = doc_retriever.generate_high_priority(text)
        total_doc["doc"] = text
        total_doc.update(doc_retriever.calc_doc_properties(text))
    # Without partitioning, summarizing "The Matrix" synopsis(about 36, 000 characters) takes about
    # 3.1 seconds, while summarizing 35, 000 characters of "Honest Abe" by Alonzo Rothschild takes about 8.5 seconds
    summary = ""
    if args.summarize:
        summary = summarizer.summarize_all(args, file_at_limit_reached, reached_limit, total_doc, outFs,)
        outF.close()
    keyOutF.close()
    keyOutOnlyF.close()
    te = time.time()
    print("%r  %2.2f ms" % ("generate_summary took", (te - ts) * 1000))
    timing = (te - ts) * 1000
    return summary, timing


def prepare_output_variables(args):
    key_large_output_lcn, key_output_lcn, output_lcn = create_text_file_output_names(args)
    if args.no_file_output:
        outF = Devnull()
        keyOutF = Devnull()
        keyOutOnlyF = Devnull()
    else:
        outF = open(output_lcn, "a")
        keyOutF = open(key_output_lcn, "a")
        keyOutOnlyF = open(key_large_output_lcn, "a")
    outFs = {"outF": outF, "keyOutF": keyOutF, "keyOutOnlyF": keyOutOnlyF}
    return outF, keyOutF, keyOutOnlyF, outFs


class RL(Enum):
    sh = 1
    me = 2
    ush = 3
    fp = 4  # 5 %


# RL = Enum("RL", "sh, me, ush, fp")


class Devnull(object):
    def write(self, *_):
        pass

    def close(self, *_):
        pass


class TextRank:
    def summarize(self, text, ratio=0.2, word_count=None, split=True, additional_stopwords=None):
        result = gensim_summarize(
            text, ratio=ratio, split=True, word_count=word_count, additional_stopwords=additional_stopwords
        )
        return result


class SummarizerAlgorithm:
    def apply_summarizer_algorithm(self, additional_stopwords, ratio, summarizer, text, word_count):
        lr_parser = PlaintextParser.from_string(text, Tokenizer("english"))
        stop_words = list(get_stop_words("english"))
        stop_words.extend(additional_stopwords)
        summarizer.stop_words = stop_words
        num_sentences = get_num_sentences_to_retrieve(ratio, word_count, text)
        summary = summarizer(lr_parser.document, num_sentences)
        result = []
        # Definition of SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",)). Only sentence returned unfortunately
        for sentence in summary:
            result.append(
                (sentence.rating, SyntacticUnit(sentence.sentence._text, token=""))
            )  # pylint: disable=protected-access
        return result


class Luhn(SummarizerAlgorithm):
    def summarize(self, text, ratio=0.2, word_count=None, split=True, additional_stopwords=None):
        summarizer = LuhnSummarizer()
        return self.apply_summarizer_algorithm(additional_stopwords, ratio, summarizer, text, word_count)


class SumyLexRank(SummarizerAlgorithm):
    def summarize(self, text, ratio=0.2, word_count=None, split=True, additional_stopwords=None):
        summarizer = LexRankSummarizer()
        return self.apply_summarizer_algorithm(additional_stopwords, ratio, summarizer, text, word_count)


class Lsa(SummarizerAlgorithm):
    def summarize(self, text, ratio=0.2, word_count=None, split=True, additional_stopwords=None):
        summarizer = LsaSummarizer()
        return self.apply_summarizer_algorithm(additional_stopwords, ratio, summarizer, text, word_count)


class CentroidWordEmbeddings:
    def summarize(self, text, ratio=0.2, word_count=None, split=True, additional_stopwords=None):
        # https://github.com/gaetangate/text-summarizer with improvements from olivier-compilatio
        # E.g limit = 500 for 500 words
        twenty_sec_longer = False  # TODO Add argument. Though this doesn't seem necessary according to https://github.com/lambdaofgod/examples-counterexamples/blob/master/notebooks/text_mining/Extractive_Summarization.ipynb ?
        embedding_model = text_summarizer.centroid_word_embeddings.load_gensim_embedding_model('glove-wiki-gigaword-50')
        centroid_we_summarizer = text_summarizer.CentroidWordEmbeddingsSummarizer(
            embedding_model, preprocess_type='nltk', additional_stopwords=additional_stopwords
        )
        centroid_we_summary = centroid_we_summarizer.summarize(text, limit=word_count)
        result = []
        for sentence in centroid_we_summary:
            result.append((sentence[0], SyntacticUnit(sentence[1], token="")))
        return result


class CentroidBOWEmbeddings:
    def summarize(self, text, ratio=0.2, word_count=None, split=True, additional_stopwords=None):
        # https://github.com/gaetangate/text-summarizer with improvements from olivier-compilatio
        centroid_bow_summarizer = text_summarizer.CentroidBOWSummarizer(
            preprocess_type='nltk', additional_stopwords=additional_stopwords
        )
        centroid_bow_summary = centroid_bow_summarizer.summarize(text, limit=word_count)
        result = []
        for sentence in centroid_bow_summary:
            result.append((sentence[0], SyntacticUnit(sentence[1], token="")))
        return result


class GPT2Medium:
    def summarize(self, text, ratio=0.2, word_count=None, split=True, additional_stopwords=None):
        # https://medium.com/analytics-vidhya/text-summarization-using-bert-gpt2-xlnet-5ee80608e961
        GPT2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")

        sentences = GPT2_model(text, num_sentences=30, min_length=40, return_as_list=True)
        result = []
        for sentence in sentences:
            result.append((0, SyntacticUnit(sentence, token="")))
        return result


class Bart:
    def summarize(self, text, ratio=0.2, word_count=None, split=True, additional_stopwords=None):
        # https://huggingface.co/sshleifer/distilbart-cnn-12-6
        num_sentences = get_num_sentences_to_retrieve(ratio, word_count, text)
        # model = TransformerSummarizer(transformer_type="DistilBART", transformer_model_key="sshleifer/distilbart-xsum-1-1")
        model = TransformerSummarizer(transformer_type="Bart", transformer_model_key="sshleifer/distilbart-xsum-1-1")

        sentences = model(text, num_sentences=num_sentences, min_length=20, return_as_list=True)
        result = []
        for sentence in sentences:
            result.append((0, SyntacticUnit(sentence, token="")))
        return result


def get_num_sentences_to_retrieve(ratio, word_count, text):
    total_num_sentences = len(text.split("\n"))
    if word_count:
        # TODO Get divisor from average number of words per sentence in this particular text
        num_sentences = word_count // 20
    else:
        num_sentences = math.ceil(ratio * total_num_sentences)
    return num_sentences


def create_text_file_output_names(args):
    ext = "txt"
    if args.org:
        ext = "org"
    time_str = time.strftime("%y%m%d_%H%M_%S")
    if args.output_prefix:
        output_lcn = "%s_%s_sum.%s" % (args.output_prefix, time_str, ext,)
        key_output_lcn = "%s_%s_key.%s" % (args.output_prefix, time_str, ext,)
        key_large_output_lcn = "%s_%s_key_only.%s" % (args.output_prefix, time_str, ext,)
    else:
        output_lcn = "%s_sum.%s" % (time_str, ext)
        key_output_lcn = "%s_key.%s" % (time_str, ext)
        key_large_output_lcn = "%s_key_only.%s" % (time_str, ext)
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if args.output:
        output_dir = os.path.expanduser(args.output)
    y = time_str[0:2]
    m = time_str[2:4]
    d = time_str[4:6]
    output_dir = os.path.join(output_dir, y, m, d)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)  # TODO Remove?
    output_lcn = "%s/%s" % (output_dir, output_lcn)
    key_output_lcn = "%s/%s" % (output_dir, key_output_lcn)
    key_large_output_lcn = "%s/%s" % (output_dir, key_large_output_lcn,)
    return key_large_output_lcn, key_output_lcn, output_lcn
