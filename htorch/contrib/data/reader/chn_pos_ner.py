# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/12 下午3:11
"""
from htorch.contrib.data.reader.base import BaseCharLabeledReader


class CHNPOSNERReader(BaseCharLabeledReader):
    def __init__(self):
        super(CHNPOSNERReader, self).__init__()

    @staticmethod
    def read_line(line):
        token, pos_tag, ner_tag = line.split('	')
        return dict(token=token, pos_tag=pos_tag, ner_tag=ner_tag)
