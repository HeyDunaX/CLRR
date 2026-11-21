"""Small analytical checks for frozen source-grounding reductions."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
import numpy as np
import pandas as pd
from measure_source_grounding import add_metrics, summarize, block_intervals


class GroundingStatisticsGate(unittest.TestCase):
    def test_score_signs_and_exclusion(self):
        frame=add_metrics(pd.DataFrame(dict(on_match=[1.,99.,2.],on_mismatch=[4.,99.,5.],
            off_match=[2.,99.,3.],off_mismatch=[4.,99.,6.],
            valid_negative=[True,False,True],block=[0,0,1])))
        np.testing.assert_allclose(frame.interaction,[1.,0.,0.])
        result=summarize(frame,1000)
        self.assertEqual(result['valid_pairs'],2)
        self.assertEqual(result['metrics_valid_pairs']['interaction']['mean'],.5)

    def test_block_bootstrap_retains_row_weights_and_pairing(self):
        values=np.array([[1.,3.],[1.,3.],[2.,4.]])
        cis=block_intervals(values,[0,0,1],1000)
        np.testing.assert_allclose(cis[1]-cis[0],2.)
        np.testing.assert_allclose(block_intervals(np.ones((3,2)),[0,0,1],1000),np.ones((2,2)))


if __name__=='__main__': unittest.main()
