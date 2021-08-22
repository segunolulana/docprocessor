import datetime
import glob
import logging
import math
import os
import random
import re
import sys
from fnmatch import fnmatch
from pprint import pprint

import numpy as np
from binaryornot.check import is_binary

try:
    # Works from site but not individually
    from .summ_utils import remove_empty_lines, output_to_file_and_stdout
except ImportError:
    from summ_utils import remove_empty_lines, output_to_file_and_stdout
from dateutil.parser import parse


class DocRetriever:
    def __init__(self, args=None):
        self.args = args

    def extract_docs(self, outFs):
        reached_limit, file_at_limit_reached, total_doc, _ = self.extract_docs_base(outFs)
        return file_at_limit_reached, reached_limit, total_doc

    def extract_docs_get_list(self, outFs):
        reached_limit, file_at_limit_reached, total_doc, doc_dict_ls = self.extract_docs_base(outFs)
        return file_at_limit_reached, reached_limit, total_doc, doc_dict_ls

    def extract_docs_base(self, outFs):
        total_doc = {}
        doc_dict_ls = []
        total_doc["line_count"] = 0
        total_doc["character_count"] = 0
        total_doc["word_count"] = 0
        total_doc["doc"] = ""
        total_doc["doc_with_placeholders"] = ""
        total_doc["placeholder_count"] = 0
        reached_limit = False
        file_at_limit_reached = ""
        # not args.files needed else will be stuck while debugging input through files in Pycharm
        if not sys.stdin.isatty() and not self.args.files:
            doc = "".join(sys.stdin)
            doc_dict = {
                "doc": doc,
            }
            doc_properties = self.calc_doc_properties(doc)
            doc_dict.update(doc_properties)
            total_doc["line_count"] += doc_dict["line_count"]
            total_doc["word_count"] += doc_dict["word_count"]
            total_doc["character_count"] += doc_dict["character_count"]
            total_doc["doc"] += doc_dict["doc"]
            doc_dict_ls = [doc_dict]
        else:
            # Using sorted(glob.glob('...')) so it follows file name order and not arbitrary order
            files_filtered = sorted(set(glob.glob(os.path.expanduser(self.args.files), recursive=True)))
            files_filtered = self.filter_by_time(files_filtered)
            files_filtered = self.filter_by_work_filename(files_filtered)
            ignore_list = []
            files_filtered = self.filter_by_ignore_list(ignore_list, files_filtered)
            line_count_dicts = {}
            line_counts = []
            len_files = len(files_filtered)
            for index, file in enumerate(files_filtered):
                if os.path.isfile(file) and not is_binary(file) and not reached_limit:
                    doc_dict = self.extract_doc(file)
                    line_count_dicts[doc_dict["line_count"]] = file
                    line_counts.append(doc_dict["line_count"])
                    if self.args.icl or total_doc["character_count"] + doc_dict["character_count"] < 4200000:
                        total_doc["line_count"] += doc_dict["line_count"]
                        total_doc["word_count"] += doc_dict["word_count"]
                        total_doc["character_count"] += doc_dict["character_count"]
                        total_doc["doc"] += doc_dict["doc"]
                        doc_dict_ls.append(doc_dict)
                    else:
                        reached_limit = True
                        file_at_limit_reached = file
                    if len_files < 10:
                        print("%s. %s %s" % (file, "lines:", doc_dict["line_count"]))
                    else:
                        if (
                            index in [*range(0, 3)]
                            or index in [*range(len_files // 2 - 1, len_files // 2 + 1)]
                            or index in [*range(len_files - 3, len_files - 1)]
                        ):
                            print("%s. %s %s" % (file, "lines:", doc_dict["line_count"]))
                        elif index == 10 or index == len_files // 2 + 1:
                            print("...")
            print(f"Line count statistics of {len_files} documents")
            mean = np.mean(line_counts)
            print(f"Mean is {mean} e.g {line_count_dicts.get(mean, 'na')}")
            median = np.median(line_counts)
            print(f"Median is {median} e.g {line_count_dicts.get(mean, 'na')}")
            percentile25 = np.percentile(line_counts, 25)
            print(f"25th percentile is {percentile25} e.g {line_count_dicts.get(math.floor(mean), 'na')}")
            percentile90 = np.percentile(line_counts, 90)
            print(f"90th percentile is {percentile90} e.g {line_count_dicts.get(math.floor(percentile90), 'na')}")
            percentile95 = np.percentile(line_counts, 95)
            print(f"95th percentile is {percentile95} e.g {line_count_dicts.get(math.floor(percentile95), 'na')}")
            percentile99 = np.percentile(line_counts, 99)
            print(f"99th percentile is {percentile99} e.g {line_count_dicts.get(math.floor(percentile99), 'na')}")
            print(f"Standard deviation is {np.std(line_counts)}")
            if outFs:
                output_to_file_and_stdout("Original character count: %s" % total_doc["character_count"], outFs["outF"])
                output_to_file_and_stdout("Original word count: %s" % total_doc["word_count"], outFs["outF"])
                output_to_file_and_stdout("Original line count: %s" % total_doc["line_count"], outFs["outF"])
        return reached_limit, file_at_limit_reached, total_doc, doc_dict_ls

    def filter_by_work_filename(self, files_filtered):
        if getattr(self.args, "work", False):
            work_filtered = []
            for file in files_filtered:
                if "Waterfox56Research" in file:
                    continue
                work_filtered.append(file)
            files_filtered = work_filtered
        return files_filtered

    def filter_by_time(self, files_filtered):
        if getattr(self.args, "timestamped", False):
            time_filtered = []
            for file in files_filtered:
                start_time = parse("0001-01-01")
                # end_time = datetime.datetime.now().strftime("%Y-%m-%d")
                end_time = datetime.datetime.now()
                if self.args.start:
                    start_time = parse(self.args.start)
                if self.args.end:
                    end_time = parse(self.args.end)
                tfe = int(self.args.tfs) + 10  # Using xxxx-xx-xx default
                if self.args.tfe:
                    tfe = int(self.args.tfe)
                date_part = (os.path.basename(file))[int(self.args.tfs) : tfe]
                if start_time <= parse(date_part) < end_time:
                    time_filtered.append(file)
            files_filtered = time_filtered
        return files_filtered

    def filter_by_ignore_list(self, ignore_list, files_filtered):
        if getattr(self.args, "exclude_from", False):
            doc = open(os.path.expanduser(self.args.exclude_from), "r").read()
            lines = doc.split("\n")
            ignore_list = [line for line in lines if line.strip() != ""]
        # E.g ignore_list = ["*.git*", "ansible.log", "*docs.seg*", "*workspace", "*.sublime-project"]
        if ignore_list:
            files_to_ignore = []
            for file in files_filtered:
                if any(fnmatch(file, "*" + pattern) for pattern in ignore_list):
                    logging.debug("Ignored %s", file)
                    files_to_ignore.append(file)
            s = set(files_to_ignore)
            excludes_filtered_out = [x for x in files_filtered if x not in s]
            files_filtered = excludes_filtered_out
        return files_filtered

    @staticmethod
    def remove_stop_words(text):
        # TODO Evaluate effect of https as stop word
        stop_phrases = ["Volumes"]
        pattern = re.compile(r"\b(" + r"|".join(stop_phrases) + r")\b\s*", re.IGNORECASE)
        text = pattern.sub("", text)
        text = DocRetriever.remove_unneeded_lines_1(text)
        non_words = ["(:properties:.*)", "(:end:)", "(:created:.*)", "(:id:.*)", r"(mygtd\d*:)", r"(:[a-zA-Z]+:)"]
        pattern = re.compile(r"(" + r"|".join(non_words) + r")", re.IGNORECASE)
        text = pattern.sub("", text)
        text = re.sub("^[^a-zA-Z]+$", "", text, flags=re.MULTILINE)
        text = re.sub("^[a-zA-Z_]+:\s.*$", "", text, flags=re.MULTILINE)  # E.g Joplin metadata
        pattern = re.compile(r" +")
        text = pattern.sub(" ", text)
        text = remove_empty_lines(text)
        return text

    @staticmethod
    def remove_unneeded_lines_1(text):
        remove_lines_with_phrases = ["www.google.com", "DONE", "HOLD", "SOMEDAY", "CANCELLED"]
        ocr_remove_lines_with_phrases = ["UTF-8 LF", "File Edit", "INSERT MODE", "Most Visited"]
        remove_lines_with_phrases.extend(ocr_remove_lines_with_phrases)
        file_list_phrases = [r"\d+\.\d+\.\d+", r"\.srt"]
        remove_lines_with_phrases.extend(file_list_phrases)
        pattern = re.compile(r"^.*(" + r"|".join(remove_lines_with_phrases) + r")\b.*$", re.IGNORECASE | re.MULTILINE,)
        text = pattern.sub("", text)
        return text

    def put_placeholders(self, text):
        placeholder_count = 0
        text = re.sub(r"TODO", " *5 ", text)
        placeholder_count, text = self.single_placeholder(placeholder_count, text, r"(\.)(org)", r" *6 ")
        # Needed if you want document to split on new line and not on full stop e.g on TODO Bookmark plus (viz. auto bookmarking)
        placeholder_count, text = self.single_placeholder(placeholder_count, text, r"(\.)(\w)", r" *1 \2")

        if self.args.mode != "gs":
            placeholder_count, text = self.single_placeholder(placeholder_count, text, r"\.\s", ". \n *9 ")
            placeholder_count, text = self.single_placeholder_M(placeholder_count, text, r"(\n)", ".\n*9 ")
        placeholder_count, text = self.single_placeholder(placeholder_count, text, r"\-", " *2 ")
        placeholder_count, text = self.single_placeholder(placeholder_count, text, r"(\w)(\/)", r"\1 *3 ")
        placeholder_count, text = self.single_placeholder(placeholder_count, text, r"(\/)", " *3 ")
        placeholder_count, text = self.single_placeholder(placeholder_count, text, r"\-", " *2 ")

        if self.args.mode not in ("cbow", "cwe"):
            text, placeholder_count = self.put_placeholders_for_words(placeholder_count, text)

        # placeholder_count, text = self.single_placeholder(placeholder_count, text, r"e\.g", "*11 ")
        logging.debug("text with placeholder: %s", text[0:1000])
        return placeholder_count, text

    def put_placeholders_for_words(self, placeholder_count, text):
        placeholder_count, text = self.single_placeholder(placeholder_count, text, r"https", " *4 ")
        placeholder_count, text = self.single_placeholder(placeholder_count, text, "http", " *7 ")
        placeholder_count, text = self.single_placeholder(placeholder_count, text, r"\bcom\b", "*10 ")
        return text, placeholder_count

    def single_placeholder(self, placeholder_count, text, placeholder, replm):
        c4 = len(re.findall(placeholder, text))
        print("%d %s" % (c4, placeholder))
        placeholder_count += c4
        text = re.sub(placeholder, replm, text)
        return placeholder_count, text

    def single_placeholder_M(self, placeholder_count, text, placeholder, replm):
        c4 = len(re.findall(placeholder, text, re.MULTILINE))
        print("%d %s" % (c4, placeholder))
        placeholder_count += c4
        text = re.sub(placeholder, replm, text, flags=re.MULTILINE)
        return placeholder_count, text

    def revert_placeholders(self, sentence_and_scores):
        for sentence_full_form in sentence_and_scores:
            temp_text = sentence_full_form[1].text
            priority_symbols_re = re.compile(r"\[Priority ")
            temp_text = priority_symbols_re.sub(r"[#", temp_text)
            temp_text = re.sub(r"\s\*1\s", ".", temp_text)
            temp_text = re.sub(r"\s\*2\s", "-", temp_text)
            temp_text = re.sub(r" ?\*3 ?", r"/", temp_text)  # Directories can start with /
            temp_text = re.sub(r" ?\*4\s", r"https", temp_text)
            temp_text = re.sub(r"\s\*5\s", r"TODO", temp_text)
            temp_text = re.sub(r"\s\*6\s", r".org", temp_text)
            temp_text = re.sub(r" ?\*7\s", r"http", temp_text)

            if self.args.mode != "gs":
                temp_text = re.sub(r"\.\s\*9\s", r"\n", temp_text)
                temp_text = re.sub(r"\.$", r"", temp_text)  # TODO This dot removal seems outrageous
                temp_text = re.sub(r"\*9\s", r"", temp_text)
            temp_text = re.sub(r"\n\n", r"\n", temp_text)

            # temp_text = re.sub(r"\*9\s", r"", temp_text)
            temp_text = re.sub(r"\*10 ?", r"com", temp_text)
            temp_text = re.sub(r"\*11\s", r"e.g", temp_text)
            temp_text = re.sub(r" ?\*12\s", r".", temp_text)
            sentence_full_form[1].text = temp_text
        return sentence_and_scores

    def extract_doc(self, file):
        doc = open(file).read()
        logging.debug(file)
        doc_dict = self.calc_doc_properties(doc)
        doc_dict["file"] = file
        return doc_dict

    # @pysnooper.snoop()
    def clean_doc(self, doc):
        pprint("Remove date and time...")
        doc = self.remove_org_time(doc)
        pprint("Remove my stopwords...")
        doc = self.remove_stop_words(doc)
        placeholder_count, doc_with_placeholders = self.put_placeholders(doc)
        doc_with_placeholders = remove_empty_lines(doc_with_placeholders)
        # doc_without_placeholders = re.sub(r"\s\*\d\s", "", doc_with_placeholders)
        doc_properties = self.calc_doc_properties(doc)
        doc_dict = {"doc_with_placeholders": doc_with_placeholders}
        doc_dict.update(doc_properties)
        doc_dict.update({"placeholder_count": placeholder_count})
        return doc_dict

    @staticmethod
    def remove_org_time(doc):
        state_str = r"( *\- State.*)"  # - State "DONE"       from "TODO"
        org_with_date_str = r"(:*[a-zA-Z]+:.*)"  # SCHEDULED: <2020-06-17 Wed .+28d>
        time_str = r"(([01]\d|2[0-3]):([0-5]\d)|24:00)"
        date_str = r"\d{4}-\d{2}-\d{2} "
        at_least_date_str = f"({date_str}.*)({time_str})*"
        time_re_1 = re.compile("^" + state_str + time_str, re.MULTILINE)
        doc = time_re_1.sub("", doc)
        time_re_2 = re.compile("^" + org_with_date_str + at_least_date_str, re.MULTILINE)
        doc = time_re_2.sub("", doc)
        brackets_or_ang = re.compile(".*" + time_str + r">|\]\n", re.MULTILINE)
        doc = brackets_or_ang.sub("", doc)
        starts_with_date = re.compile("^ *" + at_least_date_str, re.MULTILINE)
        doc = starts_with_date.sub("", doc)
        return doc

    def calc_doc_properties(self, doc):
        lines = doc.split("\n")
        non_empty_lines = [line for line in lines if line.strip() != ""]
        no_empty_line_doc = ""
        count = len(non_empty_lines)
        word_count = 0
        character_count = 0
        for line in non_empty_lines:
            word_count += len(line.split())
            character_count += len(line)
            no_empty_line_doc += line + "\n"
        return {
            "doc": no_empty_line_doc,
            "line_count": count,
            "word_count": word_count,
            "character_count": character_count,
        }

    def generate_high_priority(self, doc):
        increase_weight_phrases = [r"\[#A", r"\[#B"]
        pattern = re.compile(r"^(.*(" + r"|".join(increase_weight_phrases) + r")\b.*)$", re.IGNORECASE | re.MULTILINE,)
        matches = re.findall(pattern, doc)
        high_priority_doc = ""
        for match in matches:
            high_priority_doc += match[0] + "\n"
        logging.debug(high_priority_doc)
        return high_priority_doc

    def add_many_copies_to_add_weight(self, doc):
        lines = doc.split("\n")
        len_lines = len(lines)
        line_dictionary = dict.fromkeys(lines, "")

        increase_weight_phrases = [r"\[#A", r"\[#B"]
        pattern = re.compile(r"^(.*(" + r"|".join(increase_weight_phrases) + r")\b.*)$", re.IGNORECASE | re.MULTILINE,)
        matches = re.findall(pattern, doc)
        len_matches = len(matches)
        # 10 lines is minimum according to gensim src
        logging.debug("len_lines %s len_matches %s", len_lines, len_matches)
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
