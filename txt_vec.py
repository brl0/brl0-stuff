#!/usr/bin/env python
"""Snippets to process text in a dataframe with numpy vectorized functions."""

# standard library imports
import os, sys, math
from pathlib import Path
import ntpath
import pprint as pp
import itertools
import functools
import operator
import re
import collections
import unicodedata
import string
from IPython.core.display import HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# commonly installed imports
import numpy as np
import pandas as pd
import tqdm
import toolz
import pdvega

# other third-party package imports
import validators
import tldextract

print(sys.executable)
print(sys.version)


pd_np_str = lambda d: d.values.astype(dtype=np.str_)


def make_trans_table(tolower=True, toupper=False, repl_num=False, repl_punc=True, bad_chars=''):
    """create string translation table for translate string function"""
    good_chars = string.whitespace
    repl_chars = len(string.whitespace) * ' '
    if tolower:
        good_chars += string.ascii_uppercase
        repl_chars += string.ascii_lowercase
    if toupper:
        good_chars += string.ascii_lowercase
        repl_chars += string.ascii_uppercase
    if repl_num:
        good_chars += string.digits
        repl_chars += len(string.digits) * ' '
    if repl_punc:
        good_chars += string.punctuation
        repl_chars += len(string.punctuation) * ' '
        good_chars += '–' + '—'
        repl_chars += ' ' + ' '
        bad_chars += '®'
    return str.maketrans(good_chars, repl_chars, bad_chars)
s_trans_table = make_trans_table(True, False, True, True, "'-`")
TRANS_TABLE = make_trans_table(True, False, False, True, '')


def clean_np_str(a):
    return np.char.translate(a, TRANS_TABLE)


camelEx = re.compile(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))')
camelSplitStr = lambda s: camelEx.sub(r' \1', s)
camelNP = np.vectorize(camelSplitStr, otypes=[np.str_])
uni_norm = lambda s: unicodedata.normalize('NFKD', s)
uni_norm_np = np.vectorize(uni_norm, otypes=[np.str_])

WORD_LENGTH = 3


def filter_words(s):
    return (s and
            len(s) >= WORD_LENGTH and
            not s.isnumeric() and
            s not in stop_all)


clean_words = lambda l: list(filter(filter_words, l))
clean_words_np = np.vectorize(clean_words, otypes=[object])  #[np.str_])
rejoin_np = lambda a: np.char.join(' ', a)


procs = reversed([pd_np_str,
                  #camelNP,
                  uni_norm_np,
                  clean_np_str,
                  np.char.split,
                  clean_words_np,
                  np.vectorize(set, otypes=[object]),
                  rejoin_np
                  ])
pd_proc = toolz.functoolz.compose(*procs)


s_proc_list = reversed([pd_np_str,
                        uni_norm_np,
                        lambda a: np.char.split(a, '/'),
                        lambda l: list(map(filter_toks, l)),
                        rejoin_np,
                        #camelNP,
                        lambda a: np.char.translate(a, s_trans_table),
                        np.char.split,
                        np.vectorize(clean_words, otypes=[object])
                       ])
s_proc = toolz.functoolz.compose(*s_proc_list)


