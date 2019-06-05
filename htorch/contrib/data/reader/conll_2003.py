# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/5 上午10:57
"""
from htorch.contrib.data.reader.base import BaseCharLabeledReader


class CoNLL2003Reader(BaseCharLabeledReader):
    def __init__(self):
        super(CoNLL2003Reader, self).__init__()

    @staticmethod
    def read_line(line):
        token, pos_tag, phrase, ner_tag = line.split(' ')
        return dict(token=token, pos_tag=pos_tag, phrase_tag=phrase, ner_tag=ner_tag)
