# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/4 下午1:53
"""
import json

import pandas as pd

from htorch.contrib.data.reader.base import BaseReader


class FewRelReader(BaseReader):
    """FewRel data loader
    load FewRel data json file
    """
    def __init__(self):
        super(FewRelReader, self).__init__()

    @staticmethod
    def _load_sentence(sentence):
        tokens = sentence["tokens"]
        head = sentence["h"][0]
        head_pos = sentence["h"][-1][0]
        tail = sentence["t"][0]
        tail_pos = sentence["t"][-1][0]
        return dict(tokens=tokens, head_entity=head, head_position=head_pos, tail_entity=tail, tail_position=tail_pos)

    def read(self, filename):
        """read data from json file
        :param filename: str. json filename
        :return: `pd.DataFrame`
        """
        with open(filename, "r", encoding="utf8") as f:
            org_data = json.load(f)
        data = []
        for relation in org_data:
            for sentence in org_data[relation]:
                fmt_sentence = self._load_sentence(sentence)
                fmt_sentence["relation"] = relation
                data.append(fmt_sentence)
        df_data = pd.DataFrame(data)
        return df_data
