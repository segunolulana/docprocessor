import os
import re
import sys
import glob
import logging
from pprint import pprint, pformat
import argparse
from binaryornot.check import is_binary
from orgparse import load
import sqlite3
import subprocess
from icecream import ic
import datetime


def main():
    output = subprocess.check_output("uname -s", shell=True).decode('utf-8')
    dropbox_str = "~/Dropbox/orgnotes"
    if "Linux" in output:
        output = subprocess.check_output(
            "uname -o", shell=True).decode('utf-8')
        if "Android" in output:
            dropbox_str = "~/storage/shared/orgnotes"
    args = parse_arguments()
    total_doc = {}
    total_doc = extract_docs(dropbox_str, args, total_doc)
    print("Total org count is %d" % total_doc["count"])
    data_path = os.path.expanduser(dropbox_str + '/')
    filename = 'todos_log.db'
    os.makedirs(data_path, exist_ok=True)
    conn = sqlite3.connect(data_path + filename)
    conn.execute(
        'CREATE TABLE IF NOT EXISTS TallyTime (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp INTEGER, tally INTEGER, UNIQUE (timestamp))')
    cursor = conn.cursor()
    if args.post:
        cursor.execute(
            "INSERT INTO TallyTime (timestamp, tally) VALUES (?, ?)", (datetime.datetime.now().timestamp(), total_doc["count"]))
        conn.commit()
    elif args.open:
        rows = cursor.execute(
            "SELECT timestamp, tally FROM TallyTime",
        ).fetchall()
        for j, row in enumerate(rows):
            print(j, datetime.datetime.fromtimestamp(row[0]), row[1])
    conn.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-f", "--files", help="Glob pattern for files")
    parser.add_argument("-o", "--open", action="store_true", help="list all")
    parser.add_argument("-p", "--post", action="store_true", help="Get total todo count and post to database")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(
            format="%(asctime)s : {%(pathname)s:%(lineno)d} : %(levelname)s : %(message)s", level=logging.DEBUG
        )
    else:
        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )
    logging.getLogger("binaryornot").setLevel(logging.INFO)
    return args


def extract_docs(dropbox_str, args, total_doc):
    total_doc["todos"] = []
    total_doc["count"] = 0
    # Using sorted(glob.glob('...')) so it follows file name order and not arbitrary order
    org_files = dropbox_str + "/mygtd*.org"
    if args.files:
        org_files = args.files
    files_filtered = sorted(
        set(glob.glob(os.path.expanduser(org_files), recursive=True))
    )
    for file in files_filtered:
        if os.path.isfile(file) and not is_binary(file):
            org_doc_dict = extract_doc(file)
            total_doc["todos"].extend(org_doc_dict["todos"])
            total_doc["count"] += org_doc_dict["count"]
            print(file)
    return total_doc


def extract_doc(file):
    org_doc_dict = {}
    root = load(file)
    nodes = []
    count = 0
    for node in root[1:]:
        logging.debug(node)
        # ic(node.todo)
        if not re.match(r"\*+ (HOLD|SOMEDAY)", str(node)):
            nodes.append(node)
            count += 1
    logging.debug(file)
    org_doc_dict["todos"] = nodes
    org_doc_dict["count"] = count
    # doc_dict = calc_doc_properties(doc)
    return org_doc_dict


if __name__ == "__main__":
    main()
