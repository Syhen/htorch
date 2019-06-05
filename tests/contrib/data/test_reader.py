import unittest

import numpy as np
import pandas as pd
import pytest

from htorch.contrib.data.reader.base import BaseReader
from htorch.contrib.data.reader import FewRelReader, CoNLL2003Reader, CHNReader, CHNDetReader, CHNDetOriginalReader


class TestReader(unittest.TestCase):
    def test_base(self):
        base_reader = BaseReader()
        with pytest.raises(NotImplementedError):
            base_reader.read("")

    def test_few_rel(self):
        # ===== test FewRel =====
        reader = FewRelReader()
        data = reader.read("../data/FewRel/val.json")
        assert isinstance(data, pd.DataFrame)
        expect_output = np.array([['cape girardeau bridge', [26, 27, 28], 'P177',
                                   'mississippi river', [19, 20],
                                   ['In', 'June', '1987', ',', 'the', 'Missouri', 'Highway',
                                    'and', 'Transportation', 'Department', 'approved',
                                    'design', 'location', 'of', 'a', 'new', 'four', '-',
                                    'lane', 'Mississippi', 'River', 'bridge', 'to', 'replace',
                                    'the', 'deteriorating', 'Cape', 'Girardeau', 'Bridge',
                                    '.']]])
        assert data.head(1).values.tolist() == pd.DataFrame(expect_output).values.tolist()

        # ===== test CoNLL 2003 =====
        reader = CoNLL2003Reader()
        data = reader.read("../data/CoNLL-2003/dev.txt")
        assert isinstance(data, pd.DataFrame)
        expect_output = [['O', 'B-NP', 'NNP', 0, 'CRICKET']]
        assert data.head(1).values.tolist() == expect_output

        # ===== test chinese dataset reader =====
        reader = CHNReader()
        data = reader.read("../data/CHN-NER/example.dev")
        assert isinstance(data, pd.DataFrame)
        expect_output = [['O', 0, '在']]
        assert data.head(1).values.tolist() == expect_output

        # ===== test chinese dataset det reader =====
        reader = CHNDetReader()
        data = reader.read("../data/CHN-NER-det/test_data")
        assert isinstance(data, pd.DataFrame)
        expect_output = [['B-ORG', 0, '中']]
        assert data.head(1).values.tolist() == expect_output


if __name__ == '__main__':
    unittest.main()
