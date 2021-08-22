import argparse
import glob
import os
import time
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from binaryornot.check import is_binary
import datetime
from copy import copy
from wordmesh import Wordmesh

try:
    # Works from site but not individually
    from .doc_retriever import DocRetriever
except ImportError:
    from doc_retriever import DocRetriever


import builtins

try:
    profile = builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


stopwords = set(STOPWORDS)
stopwords.update(['properties', 'todo', 'end', 'created', 'id', 'https'])
stopwords.update(["try", "probably", "mon", "tue", "thu", "fri"])
stopwords.update(['google', 'search', 'ng', 'safe', 'active'])


def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        min_font_size=10,
        # max_words=2000,
        max_words=600,
        width=1024,
        height=720,
        max_font_size=40,
        random_state=1,  # chosen at random by flipping a coin; it was heads
    ).generate(str(data))
    output_lcn = "%s_wc.png" % time.strftime("%Y%m%d%H%M%S")
    if args.output:
        output_lcn = "%s/%s" % (args.output, output_lcn)
    wordcloud.to_file(output_lcn)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


def generate_wordmesh(data, title=None):
    wm = Wordmesh(data, keyword_extractor="sgrank", num_keywords=60)
    # wm = Wordmesh(data, keyword_extractor="sgrank", num_keywords=2000)
    # wm.set_clustering_criteria('cooccurence')
    # wm.set_fontcolor('clustering_criteria')
    wm.plot()
    output_lcn = "%s_wm.png" % time.strftime("%Y%m%d%H%M%S")
    if args.output:
        output_lcn = "%s/%s" % (os.path.expanduser(args.output), output_lcn)
    wm.save_as_html(filename=output_lcn)


parser = argparse.ArgumentParser(description='Creates a Word Cloud')
parser.add_argument('-f', '--files', required=True, help='Glob pattern for files')
parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
parser.add_argument('-o', '--output', help='Output directory')
parser.add_argument('--timestamped', '-t', action='store_true', help='timestamped like timesink')
parser.add_argument('--tfs', help='time format substring start')
parser.add_argument('--tfe', help='time format substring end')
parser.add_argument('-s', '--start')
parser.add_argument('-e', '--end')
parser.add_argument("--sw", "--stop_words", dest="stop_words", help="File of additional stop words")

args = parser.parse_args()
data = ""
lcv1 = 0
reached_word_limit = False
excludes = ['node_modules', 'stats-100919.json', '.svg', 'themes', 'components', 'schema', 'mobile-ios']
excludes.extend(['.xcassets', 'assets', 'tmp'])
files_filtered = sorted(glob.glob(os.path.expanduser(args.files), recursive=True))
time_filtered = []
if args.timestamped:
    for file in files_filtered:
        start_time = "0000-00-00"
        end_time = datetime.datetime.now().strftime('%Y-%m-%d')
        if args.start:
            start_time = args.start
        if args.end:
            end_time = args.end
        date_part = (os.path.basename(file))[int(args.tfs) : int(args.tfe)]
        if date_part >= start_time and date_part < end_time:
            time_filtered.append(file)
else:
    time_filtered = copy(files_filtered)
for file in time_filtered:  # Iterate over the files
    if not reached_word_limit:
        allowed_word = True
        if os.path.isfile(file) and not is_binary(file):
            for exclude in excludes:
                if allowed_word and not reached_word_limit:
                    if exclude in file:
                        allowed_word = False
                        break
                    elif exclude != excludes[-1]:
                        continue
                    try:
                        contents = open(file).read().lower()  # Load file contents
                        lcv1 += 1
                        print(lcv1, ' ', end='', flush=True)
                        if args.debug:
                            print(file, ' ', end='', flush=True)
                            if lcv1 < 5:
                                print("data: ", data, ' ', end='', flush=True)
                        # if lcv1 == 15000:
                        if lcv1 == 7500:
                            reached_word_limit = True
                            break
                    except UnicodeDecodeError:
                        allowed_word = False
                        continue
                    data += contents
                else:
                    break
    else:
        break
print()
if getattr(args, "stop_words", False):
    doc = open(os.path.expanduser(args.stop_words), 'r').read()
    lines = doc.split("\n")
    additional_stopwords = [line for line in lines if line.strip() != ""]
    stopwords.update(additional_stopwords)

print("Remove date and time...")
doc_retriever = DocRetriever(args)
data = doc_retriever.remove_org_time(data)
print("Remove my stopwords...")
data = doc_retriever.remove_stop_words(data)
print(len(data))
show_wordcloud(data)
# generate_wordmesh(data)
