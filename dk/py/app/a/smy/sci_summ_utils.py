import logging
import pandas as pd
import itertools
from sumy.utils import get_stop_words


def work_on_csv(file, csv_type, csv_column, drop_duplicates):
    try:
        csv_column = int(csv_column)
    except ValueError:
        pass
    if csv_type == "tsv":
        df = pd.read_csv(file, header=None, sep='\t')
    else:
        df = pd.read_csv(file, header=None)
    logging.debug(file)
    if drop_duplicates:
        doc = df[csv_column].drop_duplicates().to_string(index=False)
    else:
        doc = df[csv_column].to_string(index=False)
    logging.debug(doc)
    return doc


def work_on_csv_drop_consecutive(file, csv_type, csv_column):
    try:
        csv_column = int(csv_column)
    except ValueError:
        pass
    if csv_type == "tsv":
        df = pd.read_csv(file, header=None, sep='\t')
    else:
        df = pd.read_csv(file, header=None)
    logging.debug(file)
    lines = list(df[csv_column])
    logging.debug(lines)
    doc = ""
    for (key, group) in itertools.groupby(lines):
        doc += key + "\n"
    return doc


def get_sumy_stopwords():
    stop_words = list(get_stop_words("english"))
    STOP_WORDS = ["a", "b", "c", "d", "e", "f", "like", "better", "blob", "master", "look", "into", "try"]
    STOP_WORDS.extend(["probably", "use", "check"])
    python_sws = ["self", "def", "args", "true", "false"]
    STOP_WORDS.extend(python_sws)
    # if args.deweb:
    #     STOP_WORDS.extend(["com", "github", "www"])
    stop_words.extend(STOP_WORDS)
    return stop_words
