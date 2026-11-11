"""Execute the real worker body without its server-only module setup."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from nilearn.signal import standardize_signal


class VoxelCorrelationTest(unittest.TestCase):
    def test_worker_returns_pearson_after_sample_standardization(self):
        path = Path(__file__).resolve().parents[1]/'CentOS7/FC_lesion_voxel.py'
        worker = next(n for n in ast.parse(path.read_text()).body
                      if isinstance(n,ast.FunctionDef) and n.name=='_worker')
        rng = np.random.default_rng(42)
        seed = rng.normal(size=(120,1))
        brain = np.column_stack([seed[:,0]+rng.normal(size=120),
                                 -seed[:,0]+rng.normal(size=120),rng.normal(size=120)])
        seed_z = standardize_signal(seed,standardize='zscore_sample',detrend=False)
        brain_z = standardize_signal(brain,standardize='zscore_sample',detrend=False)
        scope = dict(np=np,nib=SimpleNamespace(load=lambda p:None),
                     _MASKER_SEED=SimpleNamespace(transform=lambda img:seed_z),
                     _MASKER_BRAIN=SimpleNamespace(transform=lambda img:brain_z))
        exec(compile(ast.Module(body=[worker],type_ignores=[]),str(path),'exec'),scope)
        _, actual = scope['_worker']((0,'server_input_not_opened'))
        expected = np.corrcoef(seed[:,0],brain.T)[0,1:]
        np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-14)


if __name__=='__main__': unittest.main()
