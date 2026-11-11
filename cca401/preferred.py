"""Author-selected formal candidate; independent of earlier 121-grid defaults."""
from pathlib import Path
import json
import numpy as np

CODE = Path(__file__).resolve().parents[1]
CURRENT = CODE/'results/cca401_160x23_seed1_retuned_mean_tminus1_20261031'
FEATURE_INPUT = CURRENT/'inputs/features.npz'
MOLECULAR_PRECISION = CURRENT/'molecular'
VOLUME_POLICY = 'both'
SEED = 1
OUTER_FOLDS = 10
INNER_FOLDS = 5
BRANCH = 'B'
INNER_REFERENCE = 'candidate-specific outer-training fit'
SCORE_SCALING = 'training score mean and SD'


def load_data():
    """Load the approved mean/T-1 input, with no reuse of the old median pack."""
    from . import nested as run
    manifest = json.loads(FEATURE_INPUT.with_suffix('.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['data_version'] == '160x23_direct_gsp1000_mean_tminus1'
    assert tuple(manifest['scales']) == run.SCALES_23
    with np.load(FEATURE_INPUT) as data:
        ids, y, volume = data['ids'], data['y'], data['volume']
        pack = {key: data[key] for key in ('sc_nc', 'fc')}
    assert len(ids) == 160 and y.shape == (160, 23)
    return ids, pack, y, volume


def selected_fit(model):
    folder = CURRENT/'selection'/VOLUME_POLICY/model.replace('+','_')/f'seed_{SEED}_{INNER_FOLDS}fold'
    selection = json.loads((folder/'selection.json').read_text())
    choice = next(c for c in selection['choices'] if c['branch'] == BRANCH)
    assert choice['reference'] == 'candidate' and choice['score_scaling'] == 'training'
    return CURRENT/choice['fit_directory'], choice


def oof_directory(model):
    return CURRENT/'oof'/VOLUME_POLICY/model.replace('+','_')/f'{OUTER_FOLDS}x{INNER_FOLDS}'/f'seed_{SEED}'
