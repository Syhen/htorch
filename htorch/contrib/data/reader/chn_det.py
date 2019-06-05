# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/5 上午11:19
"""
from htorch.contrib.data.reader.base import BaseReader, BaseCharLabeledReader


class CHNDetReader(BaseCharLabeledReader):
    def __init__(self):
        super(CHNDetReader, self).__init__()

    @staticmethod
    def read_line(line):
        token, ner_tag = line.split('	')
        return dict(token=token, ner_tag=ner_tag)


class CHNDetOriginalReader(BaseReader):
    def __init__(self):
        super(CHNDetOriginalReader, self).__init__()

    def read(self, filename):
        raise NotImplementedError()
