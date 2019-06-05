# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/5 上午11:11
"""
from htorch.contrib.data.reader.base import BaseCharLabeledReader


class CHNReader(BaseCharLabeledReader):
    def __init__(self):
        super(CHNReader, self).__init__()
