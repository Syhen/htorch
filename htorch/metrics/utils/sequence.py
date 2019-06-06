# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/6 下午4:30
"""


def tag_ids2entities(tag_ids, id2tag_mapping, padding_id=0):
    """将tagid转换成entity列表
    :param tag_ids: list. id列表
    :param id2tag_mapping: dict. id到tag的转换表
    :param padding_id: int. default 0
    :return: "%s$%s$%s" % (start_index, entity_category, entity_len)
    """
    BEGIN_TAG = "B"
    MIDDLE_TAG = "I"
    ORDINAL_TAG = "O"
    tag_names = [id2tag_mapping.get(i) for i in tag_ids if i != padding_id]
    has_entity = False
    entity = None
    start_idx = 0
    for idx, tag_name in enumerate(tag_names):
        if "-" in tag_name:
            tag_pos, tag_cls = tag_name.split("-")
        else:
            tag_pos = ORDINAL_TAG
            tag_cls = None
        if tag_pos == ORDINAL_TAG:
            if not has_entity:
                continue
            yield f"{start_idx}{entity}{idx}"
            has_entity = False
            continue
        if tag_pos == BEGIN_TAG:
            has_entity = True
            start_idx = idx
            entity = tag_cls


if __name__ == '__main__':
    print(list(tag_ids2entities([3, 3, 3, 1, 2, 2, 2, 3], {1: "B-TOC", 2: "I-TOC", 3: "O"})))
