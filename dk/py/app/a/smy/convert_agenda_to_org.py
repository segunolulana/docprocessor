r"""
Convert agenda to org form to view in orgzly
==================



"""
import re
import logging
import argparse
from summ_utils import string_found
import operator


try:
    from dk.py.app.a.smy.doc_retriever import DocRetriever
except ImportError:
    from run_sum_helper import DocRetriever

try:
    # Works from site but not individually
    from .calc.keywords_calc import keywords
except ImportError:
    from calc.keywords_calc import keywords

STOP_WORDS = ["a", "b", "c", "d", "e", "f", "like", "better", "blob", "master", "look", "into", "try"]
STOP_WORDS.extend(["probably", "use", "check"])


# @profile
def main():
    args = parse_arguments()
    text = ""
    args.icl = True
    doc_retriever = DocRetriever(args)
    file_at_limit_reached, reached_limit, total_doc = doc_retriever.extract_docs()
    text = re.sub(r"::", ":", total_doc["doc"])
    total_doc = doc_retriever.clean_doc(text)
    summary = summarize_as_keywords(short=None, medium=None, total_doc=total_doc)
    lines = text.split("\n")
    keywords_exclude = set(["com", "github", "windows", "todo", "personal"])
    summary_temp = []
    for kw in summary:
        if kw[0] not in keywords_exclude:
            summary_temp.append(kw)
    summary = summary_temp
    note_ls = []
    length_of_appended_kw = 4
    length_of_kw = len(summary)
    section = "Calendar"
    section_title = False
    for line in lines:
        note_dict = {}
        line_trimmed = line.rstrip()
        if line_trimmed == "Tasks to Refile":
            section = "Tasks to Refile"
            section_title = True
            kws = "a0"
        elif line_trimmed == "Stuck Projects":
            section = "Stuck Projects"
            section_title = True
            kws = "a2"
        elif line_trimmed == "Projects":
            section = "Projects"
            section_title = True
            kws = "a3"
        elif line_trimmed == "Project Next Tasks":
            section = "Project Next Tasks"
            section_title = True
            kws = "a1"
        elif line_trimmed == "Project Subtasks":
            section = "Project Subtasks"
            section_title = True
            kws = "a4"
        elif line_trimmed == "Standalone Tasks":
            section = "Standalone Tasks"
            section_title = True
            kws = "a5"
        elif line_trimmed == "Waiting and Postponed Tasks":
            section = "Waiting and Postponed Tasks"
            section_title = True
            kws = "a6"
        elif line_trimmed == "Tasks to Archive":
            section = "Tasks to Archive"
            section_title = True
            kws = "a7"
        note_dict["section"] = section
        note_dict["line"] = line
        for i in range(length_of_appended_kw):
            note_dict["key" + str(i)] = ""
        if not section_title:
            kws = []
            lcv = 0
            # Fix performance by using split(). A worst case scenario for example occurs if line is empty
            for i, kw in enumerate(summary):
                if string_found(kw[0], line.lower()):
                    kws.append("%s(%s)" % (kw[0], i))
                    note_dict["key" + str(lcv)] = kw[0]
                    lcv += 1
                if lcv == length_of_appended_kw:
                    break
            kws = ",".join(kws)
        else:
            note_dict["key0"] = kws
            section_title = False

        note_dict["kws"] = kws
        note_ls.append(note_dict)

    if args.mode == "a":
        sort_alphabetically(note_ls, length_of_appended_kw)
    elif args.mode == "r":
        sort_by_rank(summary, note_ls, length_of_appended_kw, length_of_kw)


def sort_by_rank(summary, note_ls, length_of_appended_kw, length_of_kw):
    note_ls_kw_sorted = note_ls.copy()
    srt = {s[0]: i for i, s in enumerate(summary)}
    section_headings = {("a%d" % i): -1 for i in range(8)}
    section_headings.update(srt)
    srt = section_headings
    # print(srt)
    for i in range(length_of_appended_kw - 1, -1, -1):
        note_ls_kw_sorted = sorted(note_ls_kw_sorted, key=lambda x: (srt.get(x["key" + str(i)], length_of_kw + 1)))
    note_ls_kw_sorted.sort(key=lambda x: x["section"])
    print("Sorted by keywords rank...")
    new_text = ""
    # len_note_ls_kw_sorted = len(note_ls_kw_sorted)
    for i, note_dict in enumerate(note_ls_kw_sorted):
        # TODO Look into splitting into regions by keywords. Might not be necessary as too many keywords
        # and too small regions
        # previous_kw = len_note_ls_kw_sorted[i-1]['kws']
        # if
        # if 0 < i < len_note_ls_kw_sorted and len_note_ls_kw_sorted[i-1]['kws']
        line = note_dict["line"]
        line_with_kws = line + "   " + note_dict["kws"] + "\n"
        if "TODO" in line:
            new_text += "* TODO " + line_with_kws
        elif "NEXT" in line:
            new_text += "* NEXT " + line_with_kws
        elif "SOMEDAY" in line:
            new_text += "* SOMEDAY " + line_with_kws
        elif "DONE" in line or "CANCELLED" in line:
            continue
        elif "HOLD" in line or "STARTED" in line:
            new_text += line_with_kws
        elif re.match(r"a[0-9]+", note_dict["kws"]):
            new_text += "\n" + line_with_kws  # Creates regions so it works better with editor folding
        else:
            new_text += line_with_kws
    print(new_text)


def sort_alphabetically(note_ls, length_of_appended_kw):
    note_ls_alpha_sorted = note_ls.copy()
    # So empty string is last not first
    for i in range(length_of_appended_kw - 1, -1, -1):
        note_ls_alpha_sorted = sorted(note_ls_alpha_sorted, key=lambda x: (x["key" + str(i)] == "", x["key" + str(i)]))
    note_ls_alpha_sorted.sort(key=lambda x: x["section"])
    # for i in range(length_of_appended_kw - 1, -1, -1):
    #     note_ls.sort(key=operator.itemgetter('key' + str(i)))
    # note_ls.sort(key=operator.itemgetter('section'))

    print("Sorted by keywords alphabetically...")
    new_text = ""
    for note_dict in note_ls_alpha_sorted:
        line = note_dict["line"]
        line_with_kws = line + "   " + note_dict["kws"] + "\n"
        if "TODO" in line:
            new_text += "* TODO " + line_with_kws
        elif "NEXT" in line:
            new_text += "* NEXT " + line_with_kws
        elif "SOMEDAY" in line:
            new_text += "* SOMEDAY " + line_with_kws
        elif "DONE" in line or "CANCELLED" in line:
            continue
        elif "HOLD" in line or "STARTED" in line:
            new_text += line_with_kws
        else:
            new_text += line_with_kws
    print(new_text)


def summarize_as_keywords(short, medium, total_doc):
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
        print("Summarising to 600 words")
        result = keywords(doc_with_placeholders, words=600, scores=True, split=True)
    elif medium or total_doc["word_count"] - total_doc["placeholder_count"] <= 32000:
        print("Summarising to 1600 words")
        result = keywords(doc_with_placeholders, words=1600, scores=True, split=True)
    else:
        print("Summarising by 5% of sentences")
        result = keywords(doc_with_placeholders, ratio=0.05, scores=True, split=True)
    print("Summary keywords count: ", len(result))
    if len(result) > 0:
        for word in STOP_WORDS:
            try:
                remove_tuple(result, word)
            except ValueError:
                pass
    return result


def remove_tuple(kw_result, word):
    for i, a_kw_result_tuple in enumerate(kw_result):
        if word == a_kw_result_tuple[0]:
            del kw_result[i]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-f", "--files", help="Glob pattern for files")
    parser.add_argument("-m", "--mode", choices=("a", "r"), help="sort mode, a alphabetically, r by rank")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logging.getLogger("binaryornot").setLevel(logging.INFO)
    return args


if __name__ == "__main__":
    main()
