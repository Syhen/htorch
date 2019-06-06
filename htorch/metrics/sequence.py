# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/6 下午5:08
"""
from htorch.metrics.utils.sequence import tag_ids2entities


def sequence_f1(y_true, y_pred, id2tag_mapping):
    """
    :param y_true:
    :param y_pred:
    :param id2tag_mapping: dict.
    :return:
    """
    tp = 0
    fp = 0
    fn = 0
    for y_t, y_p in zip(y_true, y_pred):
        tag_true = set(tag_ids2entities(y_t, id2tag_mapping))
        tag_pred = set(tag_ids2entities(y_p, id2tag_mapping))
        tp += len(tag_true & tag_pred)
        fp += len(tag_pred - tag_true)
        fn += len(tag_true - tag_pred)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * recall * precision / (recall + precision)
    return recall, precision, f1
