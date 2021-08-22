#! /usr/bin/env python

import cv2
import glob
import os
import pytesseract
from concurrent.futures import ProcessPoolExecutor
import time
import argparse
import logging
from datetime import datetime
from summ_utils import split
from pathlib import Path
import sqlite3
import subprocess
import shutil
import shlex
import re


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def ocr_files(args, files, group_id):
    logging.debug("files:%s", tuple(files))
    # Uncomment the line below to provide path to tesseract manually
    # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' for using LSTM OCR Engine
    config = "-l eng --oem 1 --psm 3"

    data_path = os.path.expanduser("~/data/")
    data_filename = "ocr.db"
    os.makedirs(data_path, exist_ok=True)
    full_data_path = os.path.join(data_path, data_filename)
    conn = sqlite3.connect(full_data_path)
    cursor = conn.cursor()
    for file in files:
        ocr_filepath = get_ocr_filepath(file)
        logging.debug("Parsed %s", file)
        if os.path.isfile(file):
            rows = cursor.execute("SELECT id FROM OCR WHERE file = ?", (file,)).fetchall()
            if rows:
                continue
            if not re.search("Waterfox57", file) and os.path.exists(ocr_filepath):
                continue
            filepath = Path(file)
            screenshot_b4_fixed_filepath = str(filepath.parents[1] / (filepath.parent.name + "_b4_fixed"))
            if re.search("Waterfox57", file):
                # Important to sync database
                shutil.copy(file, screenshot_b4_fixed_filepath)
                cursor.execute("INSERT INTO OCR (file) VALUES (?)", (file,))
                if not args.min_file or args.min_file <= filepath.name:
                    # Assumes bookmark tabbar not visible
                    output = subprocess.check_output(
                        shlex.split(f"convert '{file}' -gravity NorthWest -chop 16%x9% '{file}'")
                    ).decode("utf-8")
                    print(output)
            elif re.search("(Sublime Text)", file):
                # Important to sync database
                shutil.copy(file, screenshot_b4_fixed_filepath)
                cursor.execute("INSERT INTO OCR (file) VALUES (?)", (file,))
                if not args.min_file or args.min_file <= filepath.name:
                    output = subprocess.check_output(
                        shlex.split(f"convert '{file}' -gravity NorthWest -chop 16%x8.5% '{file}'")
                    ).decode("utf-8")
                    print(output)

            # Read image from disk
            im = cv2.imread(file, cv2.IMREAD_COLOR)

            # TODO? Dark background doesn't work well. Invert images one is sure are dark themed  e.g iterm?
            # https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#inverting-images

            # Run tesseract OCR on image
            text = pytesseract.image_to_string(im, config=config)

            # Print recognized text
            with open(ocr_filepath, "w") as text_file:
                print(f"Ocred {file}")
                text_file.write(text)
            conn.commit()
    logging.info("Finished group %s", group_id)
    # TODO Immediately summarize text?
    # process = subprocess.run(['echo', 'Even more output'],
    #                  stdout=subprocess.PIPE,
    #                  universal_newlines=True)
    # process


def get_ocr_filepath(file):
    filepath = Path(file)
    ocr_filepath = str(filepath.parents[1] / (filepath.parent.name + "_ocr") / filepath.name)
    ocr_filepath = ocr_filepath.replace(".png", ".txt")
    ocr_filepath = ocr_filepath.replace(".jpg", ".txt")
    return ocr_filepath


# @timeit
def main(main_args):
    print(datetime.now())
    # Read image path from command line
    imPath = main_args.files
    logging.debug("imPath:%s", imPath)
    file_list = sorted(glob.glob(os.path.expanduser(imPath), recursive=True))

    data_path = os.path.expanduser("~/data/")
    data_filename = "ocr.db"
    os.makedirs(data_path, exist_ok=True)
    full_data_path = os.path.join(data_path, data_filename)
    conn = sqlite3.connect(full_data_path)
    cursor = conn.cursor()

    count = 0
    outstanding_file_list = []
    for file in file_list:
        ocr_filepath = get_ocr_filepath(file)
        logging.debug("Checked if ocred: %s", file)
        if os.path.isfile(file):
            rows = cursor.execute("SELECT id FROM OCR WHERE File = ?", (file,)).fetchall()
            if re.search("Waterfox57|(Sublime Text)", file) and not rows:
                outstanding_file_list.append(file)
                count += 1
            elif not re.search("Waterfox57|(Sublime Text)", file) and not os.path.exists(ocr_filepath):
                outstanding_file_list.append(file)
                count += 1
        if count == main_args.number:
            break

    # count = 0
    # outstanding_file_list = []
    # for file in file_list:
    #     ocr_filepath = get_ocr_filepath(file)

    #     logging.debug("Checked if ocred: " + file)
    #     if (
    #         os.path.isfile(file)
    #         and not os.path.exists(ocr_filepath)
    #         and (count < main_args.number or main_args.number == -1)
    #     ):
    #         outstanding_file_list.append(file)
    #         count += 1
    # print("Performing ocr for %d documents" % count)

    if main_args.parallel:
        split_file_lists = split(outstanding_file_list, 4)
        logging.debug("split_file_lists:%s", split_file_lists)
        with ProcessPoolExecutor() as executor:
            futures = {}
            i = 0
            for group_id, files in enumerate(split_file_lists):
                logging.debug(files)
                future = executor.submit(ocr_files, main_args, files, group_id)
                futures[i] = future
                i += 1
            for i, future in futures.items():  # So can see exceptions
                print(future.result())
    else:
        ocr_files(main_args, outstanding_file_list, 0)


# Necessary for concurrency
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR files")
    parser.add_argument("-f", "--files", required=True, help="Glob pattern for files")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-n", "--number", default=-1, type=int, help="Number of files to ocr")
    parser.add_argument(
        "-m",
        "--min_file",
        help="Files with name after this should not be trimmed, probably because already trimmed before",
    )
    parser.add_argument("-p", "--parallel", action="store_true", help="Run in parallel")
    main_args = parser.parse_args()
    if main_args.debug:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    main(main_args)
