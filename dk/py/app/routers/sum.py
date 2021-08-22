from fastapi import APIRouter, Depends

# import sys
# from os.path import expanduser
from enum import Enum

# sys.path.append(expanduser('~/Utilities'))
# from run_sum import Summarizer, DocRetriever, Devnull
from ..a.smy.run_sum import Summarizer, DocRetriever, Devnull


fake_text = """ ** TODO Hard Day's Night (Beatles)
** TODO Leon (The Professional), Young Frankenstein, Children's movies, Streetcar Named Desire, Cool Hand Luke
** TODO The Big Lebowski, 40 year old virgin, Some Like It Hot, Brazil, The Truman Show, Team America: World Police
* TODO [#D] https://www.reddit.com/r/spacemacs/comments/7gvlz6/tips_for_managing_buffers_and_windows_in_spacemacs/
  :PROPERTIES:
  :CREATED:  [2020-02-20 Thu 19:29]
  :END:
"""


STOP_WORDS = ["a", "b", "c", "d", "e", "f", "like", "better", "blob", "master", "look", "into", "try", "probably"]


class RL(Enum):
    sh = 1
    me = 2
    ush = 3
    fp = 4  # 5 %


class Args:
    def __init__(
        self,
        files,
        r,
        org,
        kw,
        debug=False,
        pl=3000,
        me=False,
        long=False,
        ush=False,
        sh=False,
        a=False,
        icl=False,
        skw=False,
    ):
        self.files = files
        self.summarize = r
        self.org = org
        self.debug = debug
        self.partition_length = pl
        self.me = me
        self.long = long
        self.ush = ush
        self.kw = kw
        self.sh = sh
        self.append_keywords = a
        self.icl = icl
        self.skw = skw


router = APIRouter(
    prefix="/summa",
    tags=["summa"],
    responses={404: {"description": "Not found"}},
    # dependencies=[Depends(Summarizer)],
)


@router.get("/{item_id}")
def read_item(item_id: int, q: str = None):
    args = Args(files="~/Dropbox/orgnotes/mygtd.org", r=True, org=True, long=True, kw=True, sh=True, debug=True)
    result_length = RL.fp
    # if args.sh:
    #     result_length = RL.sh
    # elif args.ush:
    #     result_length = RL.ush
    summarizer = Summarizer(result_length, args)
    total_doc = {}
    doc_retriever = DocRetriever(args)
    file_at_limit_reached, reached_limit, total_doc = doc_retriever.extract_docs(args, total_doc)
    outFs = {"outF": Devnull(), "keyOutF": Devnull(), "keyOutOnlyF": Devnull()}
    summary = ""
    if args.summarize:
        summary = summarizer.summarize_all(args, file_at_limit_reached, reached_limit, total_doc, outFs)
    return {"item_id": item_id, "q": q, "summary": summary}
