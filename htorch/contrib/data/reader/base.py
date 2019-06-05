# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/4 下午1:45
"""
import pandas as pd


class BaseReader(object):
    def __init__(self):
        pass

    def read(self, filename):
        raise NotImplementedError()


class BaseCharLabeledReader(BaseReader):
    def __init__(self):
        super(BaseCharLabeledReader, self).__init__()

    @staticmethod
    def read_line(line):
        token, ner_tag = line.split(' ')
        return dict(token=token, ner_tag=ner_tag)

    def read(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            lines = f.readlines()
        data = []
        sentence_id = 0
        has_data = True
        for line in lines:
            line = line.strip()
            if not line:
                sentence_id += int(has_data)
                has_data = False
                continue
            has_data = True
            raw_data = self.read_line(line)
            raw_data["sentence_id"] = sentence_id
            data.append(raw_data)
        return pd.DataFrame(data)
