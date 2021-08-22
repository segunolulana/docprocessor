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
from run_sum_helper import generate_summary


# try:
#     import pysnooper
# except ImportError:
#     logging.warning("pysnooper package not installed")
import snoop


# Ex 0(Using Sublime Filter Pipes): python ...run_sum.py --kw -r -a --sm k
# Ex 1: run_sum.py --kw -r -f "ThirtyScreenshots/ss*.txt" --tfs 3 --tfe 13 -t -s 2020-09-21 -e 2020-09-25
# Ex 2: run_sum.py --kw -r -f "Dropbox/orgnotes/mygtd*.org" --op org_sum -l ~/Dropbox/org_summaries -a
# Ex 3: kernprof -v -l ~/Utilities/run_sum.py --kw -r -f "~/Dropbox/orgnotes/mygtd*.org" --sh -a --sm k --lg -p
# Ex 4: ~/.pyenv/shims/python run_sum.py --kw -r --pc 99 -c

from fnmatch import fnmatch

# from dk.py.app.a.smy.keywords import keywords

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


def main_before_snoop():
    args = parse_arguments()
    main(args)


@profile
@snoop(depth=3)
def main(args):
    output = subprocess.check_output("uname -s", shell=True).decode("utf-8")
    dropbox_agenda_str = "~/Dropbox/agenda"
    os_type = "Darwin"
    if "Linux" in output:
        output = subprocess.check_output("uname -o", shell=True).decode("utf-8")
        if "Android" in output:
            dropbox_agenda_str = "~/storage/shared/agenda"
            os_type = "Android"
    summaryPersister = None
    if args.post:
        data_path = os.path.expanduser("~/data/")
        filename = "summaries.db"
        os.makedirs(data_path, exist_ok=True)
        full_data_path = os.path.join(data_path, filename)
        if args.summary_db:
            print(args.summary_db)
            full_data_path = os.path.expanduser(args.summary_db)
        conn = sqlite3.connect(full_data_path)
        if args.save:
            summaryPersister = NonUniqueSummaryPersister()
        elif args.mode == "cm":
            summaryPersister = CommonSummaryPersister()
        else:
            summaryPersister = UniqueSummaryPersister()

        if args.post:
            summaryPersister.ensure_tables(conn)
        cursor = conn.cursor()
    summary_size = "prop"  # proportional
    if args.sh:
        summary_size = "sh"
    elif args.ush:
        summary_size = "ush"
    elif args.me:
        summary_size = "me"
    startTime = endTime = ""
    if args.start:
        startTime = args.start
    if args.end:
        endTime = args.end
    method = args.mode
    if args.mode == "gs":
        method = "gensim"
    if args.post:
        # TODO

        if args.mode == "cm":
            rows = cursor.execute(
                """select min(fileGlobId),
                summaryLine,
                count(summaryLine),
                group_concat(method),
                group_concat(datetime(timestamp, 'unixepoch')),
                min(keywords),
                avg(keywordRank),
                group_concat(score)
                from FileGlobSummary
                INNER JOIN FileGlob on FileGlobSummary.fileGlobId = FileGlob.id where
                FileGlob.value = ?
                and SummarySize = ?
                GROUP BY summaryLine
                HAVING count(summaryLine) > 1
                ORDER BY count(summaryLine) desc
                """,
                (args.files, summary_size),
            ).fetchall()
            delete_old_common_summary(args, cursor, summary_size, startTime, endTime)
            summaryPersister.insert_summary_gen_info(args, cursor, summary_size, startTime, endTime, method, 0)
            file_glob_id = cursor.lastrowid
            i = 0
            for row in rows:
                summary_line = {}
                summary_line["line"] = row[1]
                summary_line["count"] = row[2]
                summary_line["kws"] = row[5]
                summary_line["methods"] = row[3]
                summary_line["timestamps"] = row[4]
                summary_line["keyword_rank"] = row[6]
                summary_line["scores"] = row[7]
                insert_common_summary(
                    cursor, file_glob_id, i, summary_line,
                )
                print(
                    "%d: %s - %s - %s - %s - %s - %s"
                    % (
                        i,
                        summary_line["line"],
                        summary_line["count"],
                        summary_line["kws"],
                        summary_line["methods"],
                        summary_line["timestamps"],
                        summary_line["scores"],
                    )
                )
                i += 1
            conn.commit()
        else:
            if not args.save:
                delete_old_summary(args, cursor, summary_size, startTime, endTime, method)
            # conn.commit()
            summary_ls, timing = generate_summary(args)
            args.debug and ic(summary_ls[:7])  # pylint: disable=expression-not-assigned
            # ic_in_debug_mode(args, summary_ls)
            summaryPersister.insert_summary_gen_info(args, cursor, summary_size, startTime, endTime, method, timing)
            file_glob_id = cursor.lastrowid
            i = 0
            for summary in summary_ls:
                for summary_line in summary:
                    if (
                        summary_line["line"].strip() == ""
                    ):  # Some summarizers e.g Luhn can return whitespace as summary line
                        continue
                    insert_summary(cursor, file_glob_id, i, summary_line, summaryPersister.fileGlobSummaryTable)
                    print("%d: %s - %s" % (i, summary_line["line"], summary_line["kws"]))
                    i += 1
            conn.commit()
    else:
        summary_ls, _ = generate_summary(args)
    # elif args.get_all:
    #     pass
    if args.post:
        conn.close()


def insert_summary(
    cursor, file_glob_id, i, summary_line, FileGlobSummaryTable,
):
    cursor.execute(
        f"INSERT INTO {FileGlobSummaryTable} (summaryLine, keywords, summaryLineId, score, fileGlobId, keywordRank) VALUES (?, ?, ?, ?, ?, ?)",
        (
            summary_line["line"],
            summary_line["kw_strs"],
            i,
            summary_line["score"],
            file_glob_id,
            summary_line["keyword_rank"],
        ),
    )
    # fileGlobSummaryId = cursor.lastrowid
    # for kw in summary_line["kws"]:
    #     cursor.execute(
    #         f"INSERT INTO SummaryKeywords ({FileGlobSummaryTable}Id, keyword, score, RANK) VALUES (?, ?, ?, ?)",
    #         (fileGlobSummaryId, kw[0], kw[1], 0),
    #     )


def insert_common_summary(cursor, file_glob_id, i, summary_line):
    cursor.execute(
        f"INSERT INTO FileGlobSummaryCommonMethod (summaryLine, count, keywords, methods, timestamps, summaryLineId, scores, keywordRank, fileGlobId) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            summary_line["line"],
            summary_line["count"],
            summary_line["kws"],
            summary_line["methods"],
            summary_line["timestamps"],
            i,
            summary_line["scores"],
            summary_line["keyword_rank"],
            file_glob_id,
            # summary_line["keyword_rank"],
        ),
    )


class SummaryPersister:
    def ensure_tables(conn):
        pass


class UniqueSummaryPersister:
    def __init__(self):
        self.fileGlobSummaryTable = "FileGlobSummary"

    def ensure_tables(self, conn):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS FileGlob (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "value TEXT, timestamp INTEGER, method TEXT, SummarySize TEXT, startTime TEXT, endTime TEXT, timing REAL, "
            "UNIQUE(value, method, SummarySize, startTime, endTime) )"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS FileGlobSummary (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "fileGlobId INTEGER, summaryLineId INTEGER, "
            "summaryLine TEXT, keywords TEXT, score REAL, keywordRank INTEGER)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS SummaryKeywords (id INTEGER PRIMARY KEY AUTOINCREMENT, FileGlobSummaryId INTEGER, keyword TEXT, score REAL, rank INTEGER)"
        )

    def insert_summary_gen_info(self, args, cursor, summary_size, startTime, endTime, method, timing):
        cursor.execute(
            "INSERT INTO FileGlob (value, method, SummarySize, timestamp, startTime, endTime, timing) VALUES (?,?,?,?,?,?,?)",
            (args.files, method, summary_size, datetime.datetime.now().timestamp(), startTime, endTime, timing),
        )


class CommonSummaryPersister:
    def __init__(self):
        self.fileGlobSummaryTable = "FileGlobSummaryCommonMethod"

    def ensure_tables(self, conn):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS FileGlobCommonMethod (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "value TEXT, timestamp TEXT, SummarySize TEXT, startTime TEXT, endTime TEXT, timing REAL, "
            "UNIQUE(value, SummarySize, startTime, endTime) )"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS FileGlobSummaryCommonMethod (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "fileGlobId INTEGER, summaryLineId INTEGER, count INTEGER, methods TEXT, timestamps TEXT, "
            "summaryLine TEXT, keywords TEXT, scores TEXT, keywordRank INTEGER)"
        )
        # conn.execute(
        #     "CREATE TABLE IF NOT EXISTS SummaryKeywords (id INTEGER PRIMARY KEY AUTOINCREMENT, FileGlobSummaryId INTEGER, keyword TEXT, score REAL, rank INTEGER)"
        # )

    def insert_summary_gen_info(self, args, cursor, summary_size, startTime, endTime, method, timing):
        cursor.execute(
            "INSERT INTO FileGlobCommonMethod (value, SummarySize, timestamp, startTime, endTime, timing) VALUES (?,?,?,?,?,?)",
            (args.files, summary_size, datetime.datetime.now().timestamp(), startTime, endTime, timing),
        )


class NonUniqueSummaryPersister:
    def __init__(self):
        self.fileGlobSummaryTable = "FileGlobSummarySaved"

    def ensure_tables(self, conn):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS FileGlobSaved (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "value TEXT, timestamp INTEGER, method TEXT, SummarySize TEXT, startTime TEXT, endTime TEXT, timing REAL, savedName TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS FileGlobSummarySaved (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "fileGlobId INTEGER, summaryLineId INTEGER, "
            "summaryLine TEXT, keywords TEXT, score REAL, keywordRank INTEGER)"
        )

    def insert_summary_gen_info(self, args, cursor, summary_size, startTime, endTime, method, timing):
        cursor.execute(
            "INSERT INTO FileGlobSaved (value, method, SummarySize, timestamp, startTime, endTime, timing) VALUES (?,?,?,?,?,?,?)",
            (args.files, method, summary_size, datetime.datetime.now().timestamp(), startTime, endTime, timing),
        )


def delete_old_summary(args, cursor, summary_size, startTime, endTime, method):
    rows = cursor.execute(
        "SELECT id FROM FileGlob WHERE value = ? AND SummarySize = ? AND method = ? AND startTime = ? AND endTime = ?",
        (args.files, summary_size, method, startTime, endTime),
    ).fetchall()
    if rows:
        fileGlobId = rows[0][0]
        cursor.execute("DELETE FROM FileGlobSummary WHERE fileglobId = ?", (fileGlobId,))
        cursor.execute("DELETE FROM FileGlob WHERE id = ?", (fileGlobId,))


def delete_old_common_summary(args, cursor, summary_size, startTime, endTime):
    rows = cursor.execute(
        "SELECT id FROM FileGlobCommonMethod WHERE value = ? AND SummarySize = ? AND startTime = ? AND endTime = ?",
        (args.files, summary_size, startTime, endTime),
    ).fetchall()
    if rows:
        fileGlobId = rows[0][0]
        cursor.execute("DELETE FROM FileGlobSummaryCommonMethod WHERE fileglobId = ?", (fileGlobId,))
        cursor.execute("DELETE FROM FileGlobCommonMethod WHERE id = ?", (fileGlobId,))


def get_location(dropbox_agenda_str, file_search_filter, location, note):
    location = [os.path.expanduser(dropbox_agenda_str)]
    note = re.sub(r"^\*+", "", note)
    bulk_location = f"{location[0]}/bulk*.org"
    file = max(glob.glob(os.path.expanduser(bulk_location)))
    file_search_filter = ["-g", Path(file).name]
    return file_search_filter, location, note


def ic_in_debug_mode(args, note):
    if args.debug:
        ic(note)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Summarizes Doc")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
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
    parser.add_argument("-p", "--post", action="store_true", help="Post to database")
    parser.add_argument("--no", "--no_file_output", dest="no_file_output", action="store_true", help="No file output")
    parser.add_argument(
        "--dw", "--deweb", dest="deweb", action="store_true", help="Deprioritize web specific words like github, www"
    )
    parser.add_argument("--ef", "--exclude_from", dest="exclude_from", help="Pattern of files to exclude")
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
        "--rs", "--reverse", dest="reverse", help="Terminal-friendly reverse sort of keywords depending on mode"
    )
    parser.add_argument("--sv", action="store_true", dest="save")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(
            format="%(asctime)s : {%(pathname)s:%(lineno)d} : %(levelname)s : %(message)s", level=logging.DEBUG,
        )
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    # logging.getLogger("gensim").setLevel(logging.INFO)
    logging.getLogger("binaryornot").setLevel(logging.INFO)
    if args.snoop:
        snoop.install(enabled=True)
    else:
        snoop.install(enabled=False)
    return args


if __name__ == "__main__":
    main_before_snoop()
