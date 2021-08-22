#! /usr/bin/env python

import cv2
import glob
import os
# import pytesseract
from concurrent.futures import ProcessPoolExecutor
import time
import argparse
import logging
from datetime import datetime
from summ_utils import split
from pathlib import Path

try:
    import keras_ocr  # On Mac Mavericks, install tensorflow, opencv-python==4.2.0.34 (for the wheel) then keras_ocr
except ImportError:
    logging.warning("keras_ocr not installed")

import easyocr

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


# def ocr_files(args, files, reader):
def ocr_files(args, files):
    logging.debug("files:%s", tuple(files))
    if args.mode == "e":
        reader = easyocr.Reader(["en"])
    for file in files:
        ocr_filepath = get_ocr_filepath(args, file)
        logging.debug("Parsed ", file)
        if os.path.isfile(file) and not os.path.exists(ocr_filepath):
            # # Uncomment the line below to provide path to tesseract manually
            # # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

            # # Define config parameters.
            # # '-l eng'  for using the English language
            # # '--oem 1' for using LSTM OCR Engine
            # config = "-l eng --oem 1 --psm 3"

            # # Read image from disk
            # im = cv2.imread(file, cv2.IMREAD_COLOR)

            # # TODO? Dark background doesn't work well. Invert images one is sure are dark themed  e.g iterm?
            # # https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#inverting-images

            # # Run tesseract OCR on image
            # text = pytesseract.image_to_string(im, config=config)
            if args.mode == "k":
                im = keras_ocr.tools.read(file)
                pipeline = keras_ocr.pipeline.Pipeline()
                prediction_groups = pipeline.recognize([im])
                x_max = 0
                temp_str = ""
                # Print recognized text
                with open(ocr_filepath, "w") as text_file:
                    for i in prediction_groups[0]:
                        x_max_local = i[1][:, 0].max()
                        if x_max_local > x_max:
                            x_max = x_max_local
                            temp_str  += " " + i[0].ljust(15)
                        else:
                            x_max = 0
                            temp_str += "\n"
                            text_file.write(temp_str)
                            temp_str = ""
                    print("Ocred " + file)
            elif args.mode == "e":
                temp_str = ""
                prediction_groups = reader.readtext(file)
                if prediction_groups:
                    with open(ocr_filepath, "w") as text_file:
                        for group in prediction_groups:
                            temp_str += group[1] + "\n"
                        text_file.write(temp_str)
                print("Ocred " + file)


    # TODO Immediately summarize text?
    # process = subprocess.run(['echo', 'Even more output'],
    #                  stdout=subprocess.PIPE,
    #                  universal_newlines=True)
    # process


def get_ocr_filepath(args, file):
    filepath = Path(file)
    path_suffix = "_easyocr"
    if args.mode == "k":
        path_suffix = "_keras_ocr"
    ocr_dir_path = filepath.parents[1] / (filepath.parent.name + path_suffix)
    os.makedirs(str(ocr_dir_path), exist_ok=True)
    ocr_filepath = str(ocr_dir_path / filepath.name)
    ocr_filepath = ocr_filepath.replace(".png", ".txt")
    ocr_filepath = ocr_filepath.replace(".jpg", ".txt")
    return ocr_filepath


@timeit
def main():
    args = parse_arguments()
    print(datetime.now())
    # Read image path from command line
    imPath = os.path.expanduser(args.files)
    logging.debug("imPath:%s", imPath)
    file_list = sorted(glob.glob(imPath, recursive=True))

    count = 0
    outstanding_file_list = []
    for file in file_list:
        ocr_filepath = get_ocr_filepath(args, file)

        logging.debug("Checked if ocred: " + file)
        if (
            os.path.isfile(file)
            and not os.path.exists(ocr_filepath)
            and (count < args.number or args.number == -1)
        ):
            outstanding_file_list.append(file)
            count += 1
    print("Performing ocr for %d documents" % count)

    split_file_lists = split(outstanding_file_list, 4)
    logging.debug("split_file_lists:%s", split_file_lists)
    with ProcessPoolExecutor() as executor:
        futures = {}
        i = 0
        # if args.mode == "e":
        #     reader = easyocr.Reader(["en"])
        for files in split_file_lists:
            logging.debug(files)
            # future = executor.submit(ocr_files, args, files, reader)
            future = executor.submit(ocr_files, args, files)
            futures[i] = future
            i += 1
        for i, future in futures.items():  # So can see exceptions
            print(future.result())


def parse_arguments():
    parser = argparse.ArgumentParser(description="OCR files")
    parser.add_argument("-f", "--files", required=True, help="Glob pattern for files")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-n", "--number", default=-1, type=int, help="Number of files to ocr")
    parser.add_argument(
        "-m",
        "--mode",
        choices=("e", "k"),
        default="e",
        help="Options are easyocr, keras_ocr",
    )
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    return args

# Necessary for concurrency
if __name__ == "__main__":
    main()
