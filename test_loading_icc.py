import unittest
import numpy as np
import pandas as pd
import pingouin as pg

from utils import compute_icc


class LoadingICCTest(unittest.TestCase):
    def setUp(self):
        self.patterns = np.array([[1., -2., 3., .5],
                                  [1.2, -1.8, 2.4, .8],
                                  [.8, -2.5, 2.9, .3]])

    def test_independent_fold_amplitudes_do_not_change_stability(self):
        scaled = self.patterns * np.array([.01, 4., 100.])[:, None]
        self.assertEqual(compute_icc(self.patterns), compute_icc(scaled))

    def test_same_direction_has_perfect_agreement(self):
        same = np.tile(self.patterns[0], (3, 1))
        self.assertEqual(compute_icc(same * np.array([.1, 1., 20.])[:, None]), 1.)

    def test_direction_change_is_not_removed(self):
        changed = self.patterns.copy()
        changed[2] = -changed[0]
        self.assertLess(compute_icc(changed), compute_icc(self.patterns))

    def test_matches_pingouin_on_normalized_patterns(self):
        normalized = self.patterns / np.linalg.norm(self.patterns, axis=1, keepdims=True)
        table = pd.DataFrame(normalized.T).melt(var_name='fold', value_name='value')
        table['feature'] = table.index % normalized.shape[1]
        expected = pg.intraclass_corr(data=table, targets='feature', raters='fold', ratings='value')
        expected = round(float(expected.loc[expected.Type == 'ICC1', 'ICC'].iloc[0]), 4)
        self.assertEqual(compute_icc(self.patterns), expected)


if __name__ == '__main__':
    unittest.main()
