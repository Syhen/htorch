# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/18 下午3:35
"""
import json

import pandas as pd

from htorch.contrib.data.reader.base import BaseReader


class SKE2019Reader(BaseReader):
    def __init__(self):
        super(SKE2019Reader, self).__init__()

    @staticmethod
    def read_line(line):
        line = json.loads(line)
        text = line["text"]
        pos_tag = line['postag']
        relations = line['spo_list']
        word_segs = "|".join([i['word'] for i in pos_tag])
        tags = "|".join([i['pos'] for i in pos_tag])
        return text, word_segs, tags, relations

    def read(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            text, word_segs, pos_tags, relations = self.read_line(line)
            for relation in relations:
                tmp = {
                    'text': text,
                    'word_segs': word_segs,
                    'pos_tags': pos_tags
                }
                tmp.update(relation)
                data.append(tmp)
        df = pd.DataFrame(data)
        df = df[['text', 'word_segs', 'pos_tags', 'subject', 'subject_type', 'object', 'object_type', 'predicate']]
        return df
