r"""
Text Summarization
==================

Demonstrates summarizing text by extracting the most important sentences from it.

"""
import os
import re
import sqlite3

import glob
import logging

import argparse
import datetime
import sys
import subprocess
from icecream import ic


sys.path.append(os.path.dirname(__file__))
if __name__ == "__main__":
    from summ_utils import RepresentsInt, bcolors
else:
    # I.e works from site but not individually
    # TODO Why does .summ_utils not work when using multiprocess now
    from summ_utils import RepresentsInt, bcolors


import builtins
from pathlib import Path

try:
    profile = builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


import snoop


debug_1_lg = logging.getLogger("debug_1")


@profile
@snoop(depth=3)
def main(args):
    # args = parse_arguments()
    surroundings = Surroundings()
    # termux data folder needs to be symlinked to main home data folder
    data_path = os.path.expanduser("~/data/")
    filename = "summaries.db"
    os.makedirs(data_path, exist_ok=True)
    full_data_path = os.path.join(data_path, filename)
    if args.summary_db:
        print(args.summary_db)
        full_data_path = os.path.expanduser(args.summary_db)
    conn = sqlite3.connect(full_data_path)
    cursor = conn.cursor()
    opener = Opener(args)

    start_time = end_time = ""
    if args.start:
        start_time = args.start
    if args.end:
        end_time = args.end
    method = args.mode
    if args.mode == "gs":
        method = "gensim"
    if args.open or args.open_agenda or args.open_hdbscan:
        if args.mode == "cm":
            rows = cursor.execute(
                "SELECT id FROM FileGlobCommonMethod WHERE value = ? AND SummarySize = ? AND startTime = ? AND endTime = ?",
                (args.files, opener.summary_size, start_time, end_time),
            ).fetchall()
        else:
            rows = cursor.execute(
                "SELECT id FROM FileGlob WHERE value = ? AND SummarySize = ? AND method = ? AND startTime = ? AND endTime = ?",
                (args.files, opener.summary_size, method, start_time, end_time),
            ).fetchall()
        if rows:
            file_glob_id = rows[0][0]
        if args.mode != "cm":
            rows = cursor.execute("SELECT timestamp, timing FROM FileGlob WHERE id = ?", (file_glob_id,),).fetchall()
            epoch_time = rows[0][0]
            datetime_time = datetime.datetime.fromtimestamp(epoch_time)
            run_time = rows[0][1]
            print("Done " + str(datetime_time))
            print(f"Summary had taken {run_time}ms")
        number_of_summary_lines = show_all_summary_for_db(opener, cursor, file_glob_id, surroundings.dropbox_agenda_str)
        operate_on_summary_items(opener, cursor, conn, surroundings, file_glob_id, number_of_summary_lines)
    # elif args.get_all:
    #     pass
    if args.open or args.post or args.open_agenda or args.open_hdbscan:
        conn.close()


class Opener:
    def __init__(self, args):
        self.args = args
        self.files = self.args.files
        termux_files = "/data/data/com.termux/files"
        if os.environ['SHELL'] == f"{termux_files}/usr/bin/bash":
            self.files = self.args.files.replace("/storage/emulated/0/", f"/{termux_files}/home/storage/shared/")
        self.summary_size = "prop"  # proportional
        if args.sh:
            self.summary_size = "sh"
        elif args.ush:
            self.summary_size = "ush"
        elif args.me:
            self.summary_size = "me"


class Surroundings:
    def __init__(self):
        output = subprocess.check_output("uname -s", shell=True).decode("utf-8")
        self.dropbox_agenda_str = "~/Dropbox/agenda"
        self.os_type = "Darwin"
        if "Linux" in output:
            output = subprocess.check_output("uname -o", shell=True).decode("utf-8")
            if "Android" in output:
                # Termux agenda folder in main home agenda (used in PyDroid and synced with Dropbox)
                self.dropbox_agenda_str = "/storage/emulated/0/agenda"
                self.os_type = "Android"


@snoop(depth=3)
def operate_on_summary_items(opener, cursor, conn, surroundings, fileGlobId, number_of_summary_lines):
    args = opener.args
    while True and not args.stop:
        sys.stdout.write(
            f"{bcolors.OKGREEN}{bcolors.BOLD}Input letter then id of note as needed, o (Open in Editor), d (Done), x (Cancel without note),"
            f" t (Todo), s (Simsearch), i (Open Image), -1 to list all: {bcolors.ENDC}"
        )
        choice = input()
        choice_split = choice.split()
        c_verb = c_object = ""
        if len(choice_split) > 1:
            c_verb = choice_split[0]
            c_object = choice_split[1]
        elif len(choice_split) == 1:
            c_object = choice_split[0]
        if c_verb == "d":  # TODO
            cursor.execute(
                "DELETE FROM FileGlobSummary WHERE fileglobId = ? AND summaryLineId = ?", (fileGlobId, c_object)
            )
            conn.commit()
        # if c_verb == "s":
        #     sack_command = ["ag", "-Q", "-G", "mygtd\\d*.org$", note] + location
        #     output = subprocess.check_output(sack_command).decode("utf-8")
        #     print(output)

        elif RepresentsInt(c_object):
            i = int(c_object)
            if i == -1:
                show_all_summary_for_db(args, cursor, fileGlobId, surroundings.dropbox_agenda_str)
            elif i in range(number_of_summary_lines):
                process_chosen_index(opener, cursor, surroundings, fileGlobId, c_verb, i)
            else:
                sys.stdout.write("Please respond with valid integer between 0 and %s.\n" % number_of_summary_lines - 1)
        else:
            break


def process_chosen_index(opener, cursor, surroundings, fileGlobId, c_verb, i):
    args = opener.args
    if args.mode != "cm":
        rows = cursor.execute(
            "SELECT summaryLine FROM FileGlobSummary WHERE fileGlobId = ? and summaryLineId = ?", (fileGlobId, i),
        ).fetchall()
    else:
        rows = cursor.execute(
            "SELECT summaryLine FROM FileGlobSummaryCommonMethod WHERE fileGlobId = ? and summaryLineId = ?",
            (fileGlobId, i),
        ).fetchall()
    print(rows[0][0])
    note = rows[0][0]
    args.debug and ic(note)  # pylint: disable=expression-not-assigned
    # ic_in_debug_mode(args, note)
    location = []
    location_pattern = os.path.expanduser(opener.files)
    if "Dropbox_ThirtyScreenshots" in os.path.expanduser(opener.files):
        location_pattern = os.path.expanduser("~/Dropbox/ThirtyScreenshots_ocr/ss*.txt")
    ss = location_pattern.find("**")
    sl = location_pattern.rfind("/")
    if ss > 0:
        location = [location_pattern[0 : ss - 1]]
    elif sl > 0:
        location = [location_pattern[0:sl]]
    file_search_pattern = location_pattern[sl + 1 :]
    file_search_filter = ["-g", file_search_pattern]
    # TODO Add ripgrep options --max-columns=500 --max-columns-preview e.g when searching screenshot ocr's
    sack_command = ["sack", "-rg", "-iun", "-F"] + file_search_filter + [note] + location
    if args.open_agenda or args.open_hdbscan:
        latest_file, location, note = get_location(surroundings.dropbox_agenda_str, note, args)
        file_search_filter = ["-g", Path(latest_file).name]
        sack_command = ["sack", "-rg", "-iun", "-F"] + file_search_filter + [note] + location
    args.debug and ic(sack_command)  # pylint: disable=expression-not-assigned
    output = subprocess.check_output(sack_command).decode("utf-8")
    show_cluster_info(args, note, latest_file)
    print(output)
    if c_verb == "i":
        with open(os.path.expanduser("~/.sack_shortcuts")) as input_file:
            line1 = input_file.readline().rstrip()
        if line1:
            file = re.sub("^\d*\s+", "", line1)
            filepath = Path(file)
            # This removes "_ocr" from end of folder
            ocr_fileimagepath = (
                filepath.parents[1] / (filepath.parent.name[:-4]) / filepath.name.replace(".txt", ".jpg")
            )
            if not ocr_fileimagepath.is_file():
                ocr_fileimagepath = (
                    filepath.parents[1] / (filepath.parent.name[:-4]) / filepath.name.replace(".txt", ".png")
                )
            output = subprocess.check_output(["code", str(ocr_fileimagepath)]).decode("utf-8")
    else:
        open_in_editor(args, surroundings.os_type)


def show_cluster_info(args, note, latest_file):
    if args.open_hdbscan:
        with open(latest_file) as location_file:
            for line in location_file:
                if re.search(r"Cluster \-*\d+: \(\d+ docs\)$", line):
                    cluster = line
                    continue
                if f"{note.lower()}" in line:
                    break
            if cluster:
                print(cluster)


def open_in_editor(args, os_type):
    # TODO Search subset of string if full string not found e.g with ugrep
    try:
        emacsclient_cmd = "emacsclient -n"
        if os_type == "Android":
            emacsclient_cmd = "emacsclient -s /data/data/com.termux/files/usr/var/run/emacs%d/server -n" % os.getuid()
        se = emacsclient_cmd
        if args.open_with == "s":
            se = "subl"
        output = subprocess.check_output("export SACK_EDITOR='%s' && F 1" % se, shell=True).decode("utf-8")
        print(output)
    except subprocess.CalledProcessError:
        print("\nSearching failed")


def get_location(dropbox_agenda_str, note, args):
    location = [os.path.expanduser(dropbox_agenda_str)]
    note = re.sub(r"^\*+", "", note)
    if args.open_agenda:
        file_locations = f"{location[0]}/bulk*.org"
    elif args.open_hdbscan:
        file_locations = f"{location[0]}/hdbscan*.txt"
    latest_file = max(glob.glob(os.path.expanduser(file_locations)))
    return latest_file, location, note


def show_all_summary_for_db(opener, cursor, file_glob_id, dropbox_agenda_str):
    args = opener.args
    if args.mode != "cm":
        sort_order = "FileGlobSummary.summaryLineId"
        if args.sort_mode == "k":
            sort_order = "FileGlobSummary.keywordRank"
            if not args.reverse:
                sort_order += " DESC"
        elif args.sort_mode == "s":
            sort_order = "FileGlobSummary.score"
            if args.reverse:
                sort_order += " DESC"
        rows = cursor.execute(
            # "SELECT summaryLine, keywords FROM FileGlobSummary "
            # "INNER JOIN SummaryKeywords ON FileGlobSummary.Id = SummaryKeywords.FileGlobSummaryId "
            # "WHERE fileGlobId = ? and summaryLineId = ?"
            """SELECT summaryLineId, summaryLine, keywords, FileGlobSummary.score FROM FileGlobSummary
                join FileGlob on FileGlobSummary.fileGlobId = FileGlob.Id
                WHERE fileGlobId = ?
                order by """
            + sort_order,
            (file_glob_id,),
        ).fetchall()
    else:
        sort_order = "FileGlobSummaryCommonMethod.count"
        if not args.reverse:
            sort_order += " DESC"
        if args.sort_mode == "k":
            sort_order = "FileGlobSummaryCommonMethod.keywordRank"
            if not args.reverse:
                sort_order += " DESC"
        rows = cursor.execute(
            """SELECT summaryLineId, summaryLine, keywords, methods, count, FileGlobSummaryCommonMethod.scores, timestamps FROM FileGlobSummaryCommonMethod
                join FileGlobCommonMethod on FileGlobSummaryCommonMethod.fileGlobId = FileGlobCommonMethod.Id
                WHERE fileGlobId = ?
                order by """
            + sort_order,
            (file_glob_id,),
        ).fetchall()
    location = []
    location_pattern = os.path.expanduser(opener.files)
    if "Dropbox_ThirtyScreenshots" in os.path.expanduser(opener.files):
        location_pattern = os.path.expanduser("~/Dropbox/ThirtyScreenshots_ocr/ss*.txt")
    ss = location_pattern.find("/**/")
    sl = location_pattern.rfind("/")
    if ss > 0:
        location = [location_pattern[0:ss]]
    elif sl > 0:
        location = [location_pattern[0:sl]]
    file_search_pattern = location_pattern[sl + 1 :]
    file_search_filter = ["-g", file_search_pattern]
    # debug_1_lg.debug(ic.format(file_search_pattern))
    for row in rows:
        if re.search("^\**\s*STARTED", row[1]) and not args.include_started:
            continue
        note = row[1]
        if args.mode == "cm":
            output_a = f"{row[0]} {row[1]}  -  {row[2]}  -  {row[3]} -  {row[4]} - {row[5]} - {row[6]}"
        else:
            output_a = f"{row[0]} {row[1]}  -  {row[2]}  -  {row[3]}"
        if args.open_agenda or args.open_hdbscan:
            latest_file, location, note = get_location(dropbox_agenda_str, note, args)
            file_search_filter = ["-g", Path(latest_file).name]
        rg_command = ["rg", "-B1", "-A2", "--pretty", "-iun", "-F"] + file_search_filter + [note] + location
        if args.minimal:
            rg_command = ["rg", "--pretty", "-iun", "-F"] + file_search_filter + [note] + location
        else:
            print(output_a)
        args.debug and ic(rg_command)  # pylint: disable=expression-not-assigned
        try:
            output = subprocess.check_output(rg_command, stderr=subprocess.STDOUT).decode("utf-8")
            output = output.replace(os.path.expanduser("~"), "~")
            output = "\n".join(output.split("\n", 30)[:30])
            if args.minimal:
                output = output_a + " " + output
                output = output.split(":[0m[1m[31m")[0].rstrip()
            print(" ".join(output.split("\n")))
            show_cluster_info(args, note, latest_file)
        except subprocess.CalledProcessError:
            print("\nSearching failed")
        print()
    return len(rows)


def ic_in_debug_mode(args, note):
    if args.debug:
        ic(note)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Summarizes Doc")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("--dd", action="store_true", help="Debug+1 mode")
    parser.add_argument("--sn", dest="snoop", action="store_true", help="Use Snooper to trace")
    parser.add_argument("-f", "--files", help="Glob pattern for files")
    parser.add_argument("-r", "--summarize", action="store_true", help="Summarize")
    parser.add_argument("-w", "--words")
    parser.add_argument("--kw", help="Find keywords", default="gs", choices=("gs", "tpr"))
    parser.add_argument("--sh", action="store_true", help="Short summary (600 words)")
    parser.add_argument("--ush", action="store_true", help="Ultra short summary (e.g 75 words or 2 lines)")
    parser.add_argument("--me", action="store_true", help="Medium summary (1600 words)")
    parser.add_argument(
        "-g",
        "--org",
        action="store_true",
        help="Save summary output as .org, add weights when summarizing org mode docs",
    )
    parser.add_argument("-l", "--output", help="output location")
    parser.add_argument("--op", "--output_prefix", dest="output_prefix", help="output prefix")
    parser.add_argument(
        "--hp",
        "--only_high_priority",
        action="store_true",
        help="Only high priority org mode docs",
        dest="only_high_priority",
    )
    parser.add_argument("--timestamped", "-t", action="store_true", help="timestamped like timesink")
    parser.add_argument("--tfs", help="time format substring start")
    parser.add_argument("--tfe", help="time format substring end")
    parser.add_argument("-s", "--start")
    parser.add_argument("-e", "--end")
    parser.add_argument("--wk", "--work", dest="work")
    parser.add_argument(
        "-a", "--append_keywords", action="store_true",
    )
    parser.add_argument(
        "--sm", "--sort_mode", dest="sort_mode", choices=("k", "s"), help="sort mode - keys, score",
    )
    parser.add_argument("--lg", "--long", action="store_true", dest="long", help="Take longer time")
    parser.add_argument("--icl", action="store_true", help="increase character count limit")
    parser.add_argument(
        "--pl",
        "--partition_length",
        dest="partition_length",
        help="length of document partitions. Default is 4000. 1500 (800?) recommended for gpt2m",
    )
    parser.add_argument("-o", "--open", action="store_true", help="Open list index with sack or list all")
    parser.add_argument(
        "--oa", "--open_agenda", dest="open_agenda", action="store_true", help="Open location in org agenda"
    )
    parser.add_argument(
        "--oh", "--open_hdbscan", dest="open_hdbscan", action="store_true", help="Open location in hdbscan"
    )
    parser.add_argument(
        "--cm", "--common", dest="common", action="store_true", help="Get summary lines common to many methods"
    )
    parser.add_argument("-p", "--post", action="store_true", help="Post to database")
    parser.add_argument("--no", "--no_file_output", dest="no_file_output", action="store_true", help="No file output")
    parser.add_argument(
        "--dw", "--deweb", dest="deweb", action="store_true", help="Deprioritize web specific words like github, www"
    )
    parser.add_argument("--ef", "--exclude_from", dest="exclude_from", help="Pattern of files to exclude")
    parser.add_argument(
        "--ow", "--open_with", dest="open_with", help="Editor to open found file with", default="e", choices=("e", "s")
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=("gs", "slr", "luhn", "lsa", "cbow", "cwe", "gpt2m", "bart", "cm"),
        default="gs",
        help="Options are gensim, slr, luhn, lsa, centroid with bow, centroid with embeddings, bart"
        ", cm (common i.e intersection of results from all methods)",
    )
    parser.add_argument("--pc", "--percent", dest="percent", help="Percentage of number of lines to summarize to")
    parser.add_argument(
        "-c", "--complement", dest="complement", action="store_true", help="Find sentences not in summary"
    )
    parser.add_argument("-x", "--stop", action="store_true", help="Don't wait for input when getting existing summary")
    parser.add_argument("--mn", "--minimal", action="store_true", dest="minimal", help="Don't show all details")
    # parser.add_argument("--ft", '--file_type', dest="file_type", choices=('s', 'g'), default="gs", help="Options are screenshot_ocr, general")
    parser.add_argument("--da", action="store_true", dest="divide_and_approximate")
    parser.add_argument(
        "--dal", help="Number of lines to split docs for divide and approximate method. Default if not passed is 10000",
    )
    parser.add_argument("--sb", "--summary_db", dest="summary_db", help="Location of db files")
    # parser.add_argument("--fd", "--from_disk", dest="from_disk", help="Get already processed doc")
    # parser.add_argument("--td", "--to_disk", dest="to_disk", help="Save processed doc to disk")
    parser.add_argument(
        "--is", "--include_started", dest="include_started", action="store_true", help="Include STARTED TODOs"
    )
    parser.add_argument(
        "--rs", "--reverse", dest="reverse", help="Terminal-friendly reverse sort of keywords depending on mode"
    )
    args = parser.parse_args()
    configure_log_levels(args)
    if args.snoop:
        snoop.install(enabled=True)
    else:
        snoop.install(enabled=False)
    return args


def configure_log_levels(args):
    if args.debug or args.dd:
        logging.basicConfig(
            format="%(asctime)s : {%(pathname)s:%(lineno)d} : %(levelname)s : %(message)s", level=logging.DEBUG,
        )
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    # logging.getLogger("gensim").setLevel(logging.INFO)
    logging.getLogger("binaryornot").setLevel(logging.INFO)
    logging.getLogger("debug_1").setLevel(logging.INFO)

    if args.dd:
        logging.getLogger("debug_1").setLevel(logging.DEBUG)


if __name__ == "__main__":
    main_args = parse_arguments()
    main(main_args)
