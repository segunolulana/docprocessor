import logging
import re

import itertools


def string_found(needle, haystack):
    if re.search(r"\b" + re.escape(needle) + r"\b", haystack):
        return True
    return False


# https://stackoverflow.com/a/2135920
def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def remove_empty_lines(doc):
    lines = doc.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    no_empty_line_doc = ""
    for line in non_empty_lines:
        no_empty_line_doc += line + "\n"
    return no_empty_line_doc


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def output_to_file_and_stdout(val, outF):
    print(val)
    outF.write(val)
    outF.write("\n")


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
