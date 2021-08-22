r"""
Text Summarization
==================

Demonstrates summarizing text by extracting the most important sentences from it.

"""
import collections
from concurrent.futures.process import ProcessPoolExecutor
import os
import re
from summ_utils import RepresentsInt
import sys
import glob
import logging
from pprint import pprint, pformat
import argparse
from binaryornot.check import is_binary
from random import sample
from orgparse import load
import sqlite3
import subprocess
from icecream import ic

try:
    # Works from site but not individually
    from .summ_utils import bcolors, string_found
except ImportError:
    from summ_utils import bcolors, string_found


def main():
    output = subprocess.check_output("uname -s", shell=True).decode("utf-8")
    dropbox_str = "~/Dropbox/orgnotes"
    os_type = "Darwin"
    if "Linux" in output:
        output = subprocess.check_output("uname -o", shell=True).decode("utf-8")
        if "Android" in output:
            os_type = "Android"
            dropbox_str = "~/storage/shared/orgnotes"
    args = parse_arguments()
    total_doc = {}
    total_doc = extract_docs(dropbox_str, args, total_doc)
    print("Total org count is %d" % total_doc["count"])
    org_candidates = sample(total_doc["todos"], 15)

    words = load_orgnotes_words(args)
    org_candidates = append_keywords_to_sentences(words, org_candidates)

    data_path = os.path.expanduser("~/data/")
    filename = "random_org.db"
    os.makedirs(data_path, exist_ok=True)
    conn = sqlite3.connect(data_path + filename)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS List (id INTEGER PRIMARY KEY, note TEXT, keywords TEXT, keywordRank INTEGER)"
    )
    cursor = conn.cursor()
    if args.post:
        cursor.execute("DELETE FROM List")
        for i, org_candidate in enumerate(org_candidates):
            # ic(org_candidate)
            print("%d: %s - %s" % (i, org_candidate["line"], org_candidate["kw_strs"]))
            cursor.execute(
                "INSERT INTO List (id, note, keywords, keywordRank) VALUES (?, ?, ?, ?)",
                (i, org_candidate["line"], org_candidate["kw_strs"], org_candidate["keyword_rank"]),
            )
        conn.commit()
    elif args.open:
        notes_dict = print_all_notes(cursor)
        while True:
            sys.stdout.write(
                f"{bcolors.OKGREEN}{bcolors.BOLD}Input letter then id of note as needed, o (Open in Editor), d (Done), x (Cancel without note), "
                f"dl (Delete locally from db), t (Todo), -1 to list all: {bcolors.ENDC}"
            )
            choice = input()
            choice_split = choice.split()
            c_verb = c_object = ""
            if len(choice_split) > 1:
                c_verb = choice_split[0]
                c_object = choice_split[1]
            elif len(choice_split) == 1:
                c_object = choice_split[0]
                open_note_in_editor(c_object, cursor, notes_dict, dropbox_str, os_type)
            if c_verb == "d":  # TODO
                location = [os.path.expanduser(dropbox_str)]
                sack_command = ["ag", "-Q", "-G", "mygtd\\d*.org$", note] + location
                output = subprocess.check_output(sack_command).decode("utf-8")
                print(output)
            elif c_verb == "dl":
                id = notes_dict[int(c_object)]["id"]
                cursor.execute("DELETE FROM List WHERE id = ?", (id,))
                conn.commit()
                print(f"Deleted index {c_object} from db")
                notes_dict = print_all_notes(cursor)
    else:
        for i, org_candidate in enumerate(org_candidates[::-1]):
            # ic(org_candidate)
            print("%d: %s - %s" % (i, org_candidate["line"], org_candidate["kw_strs"]))
    conn.close()


def load_orgnotes_words(args):
    """
    Loads orgnotes_words
    """
    words = []
    with open(os.path.join(os.path.dirname(__file__), 'saved', '211210_1227_02_key_only.txt')) as f:
        text = f.read()
        # To match "('github', 0.19736095001235374)"
        pattern = re.compile(r"^.*'(.*)'", re.MULTILINE,)
        matches = pattern.findall(text)
        total = 0
        for index, match in enumerate(matches):
            if total <= 300:
                if args.debug:
                    print(match)
                stop_words = set(load_stop_words())
                nigeria_stop_words = ['league']
                stop_words.update(nigeria_stop_words)
                if match in stop_words:
                    continue
                words.append({"value": match, "index": index})
                total += 1
        if args.debug:
            print()
    return words


def load_stop_words():
    # From nltk and other additions
    stop_words_file = os.path.join(os.path.dirname(__file__), "stop_words.txt")
    with open(stop_words_file) as file_io:
        valid_words = set(file_io.read().split())
    return valid_words


def append_keywords_to_sentences(kws, sentence_results):
    note_ls = []
    length_of_appended_kw = 4
    length_of_kw = len(kws)
    # keywords_exclude = set("windows")
    keywords_exclude = set("")
    kws_temp = []
    for kw in kws:
        if kw["value"] not in keywords_exclude:
            kws_temp.append(kw)
    kws = kws_temp

    sentence_dict = {}
    for sentence_result in sentence_results:
        sentence_result = str(sentence_result)  # Convert from orgnode
        # Duplicates still come even after trimming original passages
        line = sentence_result
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
            note_dict["key" + str(i)] = {}
        kw_strs_1 = []
        kw_strs = []
        lcv = 0
        for i, kw in enumerate(kws):
            if string_found(kw["value"], line.lower()):
                kw_strs_1.append(kw)
                kw_strs.append("%s(%s)" % (kw, i))
                note_dict["key" + str(lcv)] = kw
                lcv += 1
            if lcv == length_of_appended_kw:
                break
        kw_strs = ",".join(kw_strs)
        note_dict["kws"] = kw_strs_1
        note_dict["kw_strs"] = kw_strs
        note_ls.append(note_dict)
    # ic(kws)
    # ic(note_ls)
    for i in range(length_of_appended_kw - 1, -1, -1):
        note_ls = sorted(note_ls, key=lambda n: n["key" + str(i)].get("index", length_of_kw + 1))
    for rank, note in enumerate(note_ls):
        note.update({"keyword_rank": rank})
    return note_ls


# NoteNamedTuple = collections.namedtuple('NoteNamedTuple',['line','keyword_rank','kws', 'kw_strs'])


def open_note_in_editor(c_object, cursor, notes_dict, dropbox_str, os_type):
    if RepresentsInt(c_object):
        i = int(c_object)
        if i == -1:
            rows = cursor.execute("SELECT note, keywords FROM List ORDER BY keywordRank DESC",).fetchall()
            for j, row in enumerate(rows):
                print(j, row[0])
        elif i in range(15):
            id = notes_dict[int(c_object)]["id"]
            rows = cursor.execute("SELECT note FROM List WHERE id = ?", (id,),).fetchall()
            print(rows[0][0])
            note = rows[0][0]

            note = re.sub(r"^\*+", "", note)  # stars don't work well with sack?
            location = [os.path.expanduser(dropbox_str)]
            sack_command = ["sack", "-ag", "-Q", "-G", "mygtd\\d*.org$", note] + location
            try:
                output = subprocess.check_output(sack_command).decode("utf-8")
                print(output)
                emacsclient_cmd = "emacsclient -n"
                if os_type == "Android":
                    emacsclient_cmd = (
                        "emacsclient -s /data/data/com.termux/files/usr/var/run/emacs%d/server -n" % os.getuid()
                    )
                output = subprocess.check_output("export SACK_EDITOR='%s' && F 1" % emacsclient_cmd, shell=True).decode(
                    "utf-8"
                )
                print(output)
            except subprocess.CalledProcessError:
                print("\nSearching failed")
        else:
            sys.stdout.write(
                f"{bcolors.OKGREEN}{bcolors.BOLD}Please respond with valid integer between 0 and 14.{bcolors.ENDC}\n"
            )
    else:
        sys.stdout.write(f"{bcolors.OKGREEN}{bcolors.BOLD}Please respond with -1 or valid integer.{bcolors.ENDC}\n")


def print_all_notes(cursor):
    rows = cursor.execute("SELECT note, id, keywords FROM List ORDER BY keywordRank DESC",).fetchall()
    notes_dict = {}
    for j, row in enumerate(rows):
        print(j, row[0], row[2])
        notes_dict[j] = {"id": row[1]}
    return notes_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generates random org notes")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    # parser.add_argument("-a", "--get_all", action="store_true", help="Get stored random data")
    parser.add_argument("-f", "--files", help="Glob pattern for files")
    parser.add_argument("-o", "--open", action="store_true", help="Open list index with sack or list all")
    parser.add_argument("-p", "--post", action="store_true", help="Get new random list and post to database")
    parser.add_argument("--ns", action="store_true", help="Exclude notes with STARTED todo state")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(
            format="%(asctime)s : {%(pathname)s:%(lineno)d} : %(levelname)s : %(message)s", level=logging.DEBUG
        )
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logging.getLogger("binaryornot").setLevel(logging.INFO)
    return args


def extract_docs(dropbox_str, args, total_doc):
    total_doc["todos"] = []
    total_doc["count"] = 0
    # Using sorted(glob.glob('...')) so it follows file name order and not arbitrary order
    org_files = dropbox_str + "/mygtd*.org"
    if args.files:
        org_files = args.files
    files_filtered = sorted(set(glob.glob(os.path.expanduser(org_files), recursive=True)))
    for file in files_filtered:
        if os.path.isfile(file) and not is_binary(file):
            org_doc_dict = extract_doc(args, file)
            total_doc["todos"].extend(org_doc_dict["todos"])
            total_doc["count"] += org_doc_dict["count"]
            print(file)
    return total_doc


def extract_doc(args, file):
    org_doc_dict = {}
    root = load(file)
    nodes = []
    count = 0
    for node in root[1:]:
        logging.debug(node)
        # ic(node.todo)
        tags = "HOLD|SOMEDAY|CANCELLED|DONE"
        if args.ns:
            tags += "|STARTED"
        if not re.match(rf"\*+ ({tags})", str(node)):
            nodes.append(node)
            count += 1
    logging.debug(file)
    org_doc_dict["todos"] = nodes
    org_doc_dict["count"] = count
    # doc_dict = calc_doc_properties(doc)
    return org_doc_dict


if __name__ == "__main__":
    main()
