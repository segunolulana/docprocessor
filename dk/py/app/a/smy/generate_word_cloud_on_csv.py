import argparse
import glob
import os
import time
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from binaryornot.check import is_binary
from sci_summ_utils import work_on_csv


stopwords = set(STOPWORDS)
stopwords.update(
    ['properties', 'todo', 'end', 'created', 'id', 'https'])
stopwords.update(
    ['google', 'search', 'ng', 'safe', 'active'])


def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        min_font_size=10,
        max_words=2000,
        width=1024,
        height=720,
        random_state=1  # chosen at random by flipping a coin; it was heads
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


parser = argparse.ArgumentParser(
    description='Creates a Word Cloud')
parser.add_argument('-f', '--files', required=True,
                    help='Glob pattern for files')
parser.add_argument(
    '-d', '--debug', action='store_true', help='Debug mode')
parser.add_argument(
    '-o', '--output', help='Output directory')
parser.add_argument(
    '--ct', help='csvtype, csv or tsv')
parser.add_argument(
    '--cc', help='csv column')

args = parser.parse_args()
data = ""
lcv1 = 0
reached_word_limit = False
excludes = ['node_modules', 'stats-100919.json', '.svg',
            'themes', 'components', 'schema', 'mobile-ios']
excludes.extend(['.xcassets', 'assets', 'tmp'])
text = ""
for file in glob.glob(args.files, recursive=True):
    doc = work_on_csv(file, args.ct, args.cc, False)
    text += doc

show_wordcloud(text)
