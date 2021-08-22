r"""
Sorting and viewing org agenda
==================

E.g sort_agenda.py -f "~/Dropbox/agenda/bulk_1.txt" -m r -o -g 50

"""
import re
import logging
import argparse
from summ_utils import string_found
import operator
import os
import sqlite3
import sys
import subprocess
import glob
from icecream import ic
import pysnooper
import pke


try:
    from dk.py.app.a.smy.doc_retriever import DocRetriever
except ImportError:
    from run_sum_helper import DocRetriever

try:
    # Works from site but not individually
    from .calc.keywords_calc import keywords
except ImportError:
    from calc.keywords_calc import keywords

try:
    # Works from site but not individually
    from .summ_utils import RepresentsInt, string_found, remove_empty_lines, split
except ImportError:
    from summ_utils import RepresentsInt, string_found, remove_empty_lines, split

STOP_WORDS = ["a", "b", "c", "d", "e", "f", "like", "better", "blob", "master", "look", "into", "try"]
STOP_WORDS.extend(["probably", "use", "check"])


# @pysnooper.snoop(depth=5)
# @profile
def main():
    args = parse_arguments()
    output = subprocess.check_output("uname -s", shell=True).decode("utf-8")
    dropbox_str = "~/Dropbox/orgnotes"
    if "Linux" in output:
        output = subprocess.check_output("uname -o", shell=True).decode("utf-8")
        if "Android" in output:
            dropbox_str = "~/storage/shared/orgnotes"
    if args.open or args.post:
        data_path = os.path.expanduser("~/data/")
        filename = "agendas.db"
        os.makedirs(data_path, exist_ok=True)
        conn = sqlite3.connect(data_path + filename)
        if args.post:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS FileGlob (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT UNIQUE)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS FileGlobAgenda (fileGlobId INTEGER, agendaLineId INTEGER, agendaLine TEXT, keywords TEXT)"
            )
        cursor = conn.cursor()
    if args.post:
        rows = cursor.execute("SELECT id FROM FileGlob WHERE value = ?", (args.files,)).fetchall()
        if rows:
            fileGlobId = rows[0][0]
            cursor.execute("DELETE FROM FileGlobAgenda WHERE fileglobId = ?", (fileGlobId,))
            cursor.execute("DELETE FROM FileGlob WHERE id = ?", (fileGlobId,))
            # conn.commit()
        agenda_ls = generate_sorted_agenda(args)
        ic_in_debug_mode(args, agenda_ls)
        cursor.execute("INSERT INTO FileGlob (value) VALUES (?)", (args.files,))
        fileGlobId = cursor.lastrowid
        for agenda in agenda_ls:
            for i, agenda_line in enumerate(agenda):
                # ic_in_debug_mode(args, agenda_line)
                cursor.execute(
                    "INSERT INTO FileGlobAgenda (agendaLine, keywords, agendaLineId, fileGlobId) VALUES (?, ?, ?, ?)",
                    (agenda_line["line"], agenda_line["kws"], i, fileGlobId),
                )
                print("%d: %s - %s" % (i, agenda_line["line"], agenda_line["kws"]))
        conn.commit()
    elif args.open:
        rows = cursor.execute(
            """SELECT agendaLineId, agendaLine, keywords, fileGlobId FROM FileGlobAgenda
                join FileGlob on FileGlobAgenda.fileGlobId = FileGlob.Id WHERE value = ?
                order by FileGlobAgenda.agendaLineId""",
            (args.files,),
        ).fetchall()
        fileGlobId = rows[0][3]
        number_of_agenda_lines = len(rows)
        doc_type = "summary"
        show_rows(args, number_of_agenda_lines, rows)
        while True:
            sys.stdout.write("Input id of %s line to open, -1 to list all, non integer to quit: " % (doc_type))
            choice = input()

            if RepresentsInt(choice):
                i = int(choice)
                if i == -1:
                    rows = cursor.execute(
                        """SELECT agendaLineId, agendaLine, keywords, fileGlobId FROM FileGlobAgenda
                            join FileGlob on FileGlobAgenda.fileGlobId = FileGlob.Id WHERE value = ?
                            order by FileGlobAgenda.agendaLineId""",
                        (args.files,),
                    ).fetchall()
                    show_rows(args, number_of_agenda_lines, rows)
                elif i in range(number_of_agenda_lines):
                    rows = cursor.execute(
                        "SELECT agendaLine, keywords FROM FileGlobAgenda WHERE fileGlobId = ? and agendaLineId = ?",
                        (fileGlobId, i),
                    ).fetchall()
                    print(rows[0][0])
                    note = rows[0][0]
                    ic_in_debug_mode(args, note)
                    p = re.compile(r"(.+)\s{2,}((TODO|NEXT|SOMEDAY).+)(:PERSONAL:)(.*)")
                    note = p.search(note).group(2).rstrip()
                    ic_in_debug_mode(args, note)
                    location = [os.path.expanduser(dropbox_str)]
                    sack_command = ["sack", "-ag", "-Q", "-G", "mygtd\\d*.org$", note] + location
                    args.debug and ic(sack_command)
                    output = subprocess.check_output(sack_command).decode("utf-8")
                    print(output)
                    # TODO Search subset of string if full string not found
                    try:
                        output = subprocess.check_output(
                            "export SACK_EDITOR='emacsclient -n' && F 1", shell=True
                        ).decode("utf-8")
                        print(output)
                    except Exception:
                        print("\nSearching failed")
                else:
                    sys.stdout.write(
                        "Please respond with valid integer between 0 and %s.\n" % number_of_agenda_lines - 1
                    )
            else:
                break
    else:
        agenda_ls = generate_sorted_agenda(args)
    # elif args.get_all:
    #     pass
    if args.open or args.post:
        conn.close()


def show_rows(args, number_of_agenda_lines, rows):
    if args.neighbour:
        neighbour = int(args.neighbour)
        start_index = max(0, neighbour - 25)
        end_index = min(neighbour + 26, number_of_agenda_lines)
        rows = rows[start_index:end_index]
    for row in rows:
        print(row[0], row[1], " - ", row[2])


def generate_sorted_agenda(args):
    text = ""
    args.icl = True
    doc_retriever = DocRetriever(args)
    _, _, total_doc = doc_retriever.extract_docs(None)
    text = re.sub(r"::", ":", total_doc["doc"])
    total_doc = doc_retriever.clean_doc(text)
    summary = summarize_as_keywords(args, short=None, medium=None, total_doc=total_doc)
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
        # Creates regions so it works better with editor folding
        if line_trimmed == "Tasks to Refile":
            section, section_title, kws, line = "Tasks to Refile", True, "a0", "\n" + line
        elif line_trimmed == "Stuck Projects":
            section, section_title, kws, line = "Stuck Projects", True, "a2", "\n" + line
        elif line_trimmed == "Projects":
            section, section_title, kws, line = "Projects", True, "a3", "\n" + line
        elif line_trimmed == "Project Next Tasks":
            section, section_title, kws, line = "Project Next Tasks", True, "a1", "\n" + line
        elif line_trimmed == "Project Subtasks":
            section, section_title, kws, line = "Project Subtasks", True, "a4", "\n" + line
        elif line_trimmed == "Standalone Tasks":
            section, section_title, kws, line = "Standalone Tasks", True, "a5", "\n" + line
        elif line_trimmed == "Waiting and Postponed Tasks":
            section, section_title, kws, line = "Waiting and Postponed Tasks", True, "a6", "\n" + line
        elif line_trimmed == "Tasks to Archive":
            section, section_title, kws, line = "Tasks to Archive", True, "a7", "\n" + line
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
        note_ls = sort_alphabetically(note_ls, length_of_appended_kw)
    elif args.mode == "r":
        note_ls = sort_by_rank(summary, note_ls, length_of_appended_kw, length_of_kw)
    return [note_ls]


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
        new_text += line_with_kws
    print(new_text)
    return note_ls_kw_sorted


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
    return note_ls_alpha_sorted


def summarize_as_keywords(args, short, medium, total_doc):
    # gensim.summarization.keywords fetching different results as it is non-deterministic
    # e.g with pos_filter=('NP') i.e noun phrase
    # https://github.com/RaRe-Technologies/gensim/issues/2586. Aug 9, 2019
    # https://github.com/DerwenAI/pytextrank probably better here?
    # Default gensim.summarization.keywords
    #   .keywords(text, ratio=0.2, words=None, split=False,
    #               scores=False, pos_filter=('NN', 'JJ'), lemmatize=False, deacc=True)
    # NN is noun, singular or mass. JJ is adjective
    kw_result = []
    # doc_without_placeholders = re.sub(r"\s\*\d\s", "", total_doc["doc_with_placeholders"])
    # doc_properties = calc_doc_properties(doc_without_placeholders)
    doc_with_placeholders = total_doc["doc_with_placeholders"]
    if args.kw == "tpr":
        extractor = pke.unsupervised.TopicRank()
    elif args.kw == "kpm":
        extractor = pke.unsupervised.KPMiner()
    if args.kw in ["tpr", "kpm"]:
        # extractor = pke.unsupervised.TextRank()
        extractor.load_document(doc_with_placeholders)
        extractor.candidate_selection()
        extractor.candidate_weighting()
    if short or total_doc["word_count"] - total_doc["placeholder_count"] <= 12000:
        print("Summarising to 600 words")
        if args.kw == "gs":
            kw_result = keywords(doc_with_placeholders, words=600, scores=True, split=True)
        elif args.kw in ["tpr", "kpm"]:
            kw_result = extractor.get_n_best(n=600)
    elif medium or total_doc["word_count"] - total_doc["placeholder_count"] <= 32000:
        print("Summarising to 1600 words")
        if args.kw == "gs":
            kw_result = keywords(doc_with_placeholders, words=1600, scores=True, split=True)
        elif args.kw in ["tpr", "kpm"]:
            kw_result = extractor.get_n_best(n=1600)
    else:
        print("Summarising by 5%")
        if args.kw == "gs":
            kw_result = keywords(doc_with_placeholders, ratio=0.05, scores=True, split=True)
        elif args.kw in ["tpr", "kpm"]:
            no_w = ratio * total_doc["word_count"]
            kw_result = extractor.get_n_best(n=no_w)
    print("Summary keywords count: ", len(kw_result))
    print("Original character count: %s" % total_doc["character_count"])
    print("Original word count: %s" % total_doc["word_count"])
    print("Original line count: %s" % total_doc["line_count"])
    if len(kw_result) > 0:
        for word in STOP_WORDS:
            try:
                remove_tuple(kw_result, word)
            except ValueError:
                pass
    return kw_result


def ic_in_debug_mode(args, note):
    if args.debug:
        ic(note)


def remove_tuple(kw_result, word):
    for i, a_kw_result_tuple in enumerate(kw_result):
        if word == a_kw_result_tuple[0]:
            del kw_result[i]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-f", "--files", help="Glob pattern for files")
    parser.add_argument("-m", "--mode", choices=("a", "r"), help="sort mode, a alphabetically, r by rank")
    parser.add_argument("-o", "--open", action="store_true", help="Open list index with sack or list all")
    parser.add_argument("-p", "--post", action="store_true", help="Post to database")
    parser.add_argument(
        "-g", "--neighbour", help="Index of central item from which you want to list indices preceding and after"
    )
    parser.add_argument(
        "--kw", choices=("gs", "tpr", "kpm"), default="gs", help="keyword extraction method e.g gensim, kp-miner"
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logging.getLogger("binaryornot").setLevel(logging.INFO)
    return args


if __name__ == "__main__":
    main()
