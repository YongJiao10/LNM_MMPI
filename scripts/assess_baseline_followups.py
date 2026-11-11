#!/usr/bin/env python3
"""Summarize the adopted 160-person baseline sensitivity and permutations."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
BASELINE = CODE / 'results/cca401_160x23_seed1_retuned_mean_tminus1_20261031'


def top_sides(values, names):
    positive = np.array([i for i in np.argsort(values)[::-1] if values[i] > 0])[:5]
    negative = np.array([i for i in np.argsort(values) if values[i] < 0])[:5]
    return {label: [dict(scale=str(names[i]), loading=float(values[i])) for i in indices]
            for label, indices in [('positive', positive), ('negative', negative)]}


def collect_demographic():
    root = CODE / 'results/baseline_followups_20261031/age_sex_sensitivity'
    adjusted = pd.read_csv(root / 'primary_clinical.csv')
    original = pd.read_csv(BASELINE / 'primary_clinical.csv').set_index('model')
    result = []
    for row in adjusted.itertuples(index=False):
        old_fit = BASELINE / original.loc[row.model, 'fit_directory']
        fit = root / row.fit_directory
        symptoms = pd.read_csv(fit / 'symptom_loadings.csv', index_col=0)
        with np.load(old_fit / 'model.npz') as a, np.load(fit / 'model.npz') as b:
            full_r = np.corrcoef(a['behavior'].T, b['behavior'].T)[:10, 10:]
            adjusted_full = b['behavior'][:, :2]
        folder = f'oof/both/{row.model.replace("+", "_")}/10x5/seed_1/B_aligned.npz'
        with np.load(BASELINE / folder) as a, np.load(root / folder) as b:
            old_mean, new_mean = a['behavior'].mean(axis=0), b['behavior'].mean(axis=0)
            oof_r = np.corrcoef(old_mean.T, new_mean.T)[:10, 10:]
        result.append(dict(model=row.model, full_behavior_first2_cross_r=full_r[:2, :2].tolist(),
            oof_mean_behavior_first2_cross_r=oof_r[:2, :2].tolist(),
            dimensions=[dict(cd=j+1,
                full_vs_oof_behavior_r=float(np.corrcoef(adjusted_full[:, j], new_mean[:, j])[0, 1]),
                full_symptoms=top_sides(symptoms.iloc[:, j].to_numpy(), symptoms.index),
                oof_mean_symptoms=top_sides(new_mean[:, j], symptoms.index)) for j in range(2)]))
    molecular = root / 'molecular' / Path(adjusted.loc[adjusted.model.eq('nc+fc'), 'fit_directory'].item()).relative_to('fits')
    summary = dict(models=result, pls=pd.read_csv(molecular / 'pls.csv').to_dict('records'),
        pet=pd.read_csv(molecular / 'pet.csv').to_dict('records'),
        genes=pd.read_csv(molecular / 'genes.csv').to_dict('records'))
    (root / 'assessment.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('Collected demographic sensitivity evidence.', flush=True)


def fixed_score_age_sex():
    """Describe age/sex-adjusted associations of the fixed baseline OOF scores."""
    demographics = pd.read_csv(CODE / 'data/MMPI_Demographics.csv')
    covariates = pd.read_csv(CODE / 'results/baseline_followups_20261031/inputs/covariates.csv')
    original = pd.read_csv(BASELINE / 'primary_clinical.csv').set_index('model')
    assert demographics.study_id.is_unique and covariates.study_id.is_unique
    expected_ids = set(covariates.study_id)
    assert len(expected_ids) == 160
    snapshot = covariates.merge(demographics[['study_id', 'scan_age', 'gender']],
                                on='study_id', validate='one_to_one', suffixes=('_saved', '_current'))
    assert len(snapshot) == 160
    np.testing.assert_allclose(snapshot.scan_age_saved, snapshot.scan_age_current)
    np.testing.assert_array_equal(snapshot.registry_gender, snapshot.gender)
    rows = []
    for model in ('nc', 'fc', 'nc+fc'):
        folder = BASELINE / f'oof/both/{model.replace("+", "_")}/10x5/seed_1'
        scores = pd.read_csv(folder / 'outer_scores.csv')
        scores = scores.loc[scores.branch.eq('B')].copy()
        assert len(scores) == 160 and scores.study_id.is_unique
        assert set(scores.study_id) == expected_ids
        joined = scores.merge(demographics[['study_id', 'scan_age', 'gender']],
                              on='study_id', validate='one_to_one')
        assert len(joined) == 160
        assert np.isfinite(joined[['scan_age', 'gender']].to_numpy()).all()
        assert set(joined.gender) == {0, 1}
        design = np.column_stack([np.ones(len(joined)), joined.scan_age, joined.gender])
        for dimension in (1, 2):
            values = joined[[f'x_score_{dimension}', f'y_score_{dimension}']].to_numpy()
            assert np.isfinite(values).all()
            raw = np.corrcoef(values.T)[0, 1]
            np.testing.assert_allclose(raw, original.loc[model, f'r_{dimension}'],
                                       rtol=1e-7, atol=1e-9)
            residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
            adjusted = np.corrcoef(residual.T)[0, 1]
            rows.append(dict(model=model, dimension=dimension, n=len(joined),
                             primary_held_out_r=raw, age_sex_partial_r=adjusted,
                             change_in_r=adjusted-raw))
    output = CODE / 'results/baseline_followups_20261031/age_sex_sensitivity/fixed_score_age_sex.csv'
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    print(result.to_string(index=False), flush=True)
    print(f'Saved {output}', flush=True)


def verify_permutations():
    from scipy.stats import false_discovery_control

    root = CODE / 'results/baseline_followups_20261031/permutations'
    nulls = pd.read_csv(root / 'permutation_nulls.csv')
    inference = pd.read_csv(root / 'inference.csv')
    observed = pd.read_csv(BASELINE / 'primary_clinical.csv').set_index('model')
    status = json.loads((root / 'run.json').read_text())
    assert status['status'] == 'complete' and status['model_permutations'] == 3000
    assert len(nulls) == 3000 and len(inference) == 6
    assert set(nulls.model) == set(observed.index) == {'nc', 'fc', 'nc+fc'}
    assert not nulls.duplicated(['model', 'permutation_index']).any()
    assert np.isfinite(nulls[['r_1', 'r_2']]).all().all()
    summaries = []
    for row in inference.itertuples(index=False):
        group = nulls.loc[nulls.model.eq(row.model)]
        assert set(group.permutation_index) == set(range(1000))
        values = group[f'r_{row.dimension}'].to_numpy()
        r = observed.loc[row.model, f'r_{row.dimension}']
        np.testing.assert_allclose(row.observed_r, r, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(row.cohens_d, 2*r/np.sqrt(1-r*r), atol=1e-12)
        exceedances = int(np.sum(values >= row.observed_r))
        np.testing.assert_allclose(row.p_permutation, (exceedances+1)/1001, atol=1e-12)
        summaries.append(dict(model=row.model, dimension=row.dimension, n_perm=1000,
            exceedances=exceedances, null_min=float(values.min()),
            null_median=float(np.median(values)), null_q95=float(np.quantile(values, .95)),
            null_q99=float(np.quantile(values, .99)), null_max=float(values.max())))
    np.testing.assert_allclose(inference.q_fdr_6,
        false_discovery_control(inference.p_permutation.to_numpy()), atol=1e-12)
    pd.DataFrame(summaries).to_csv(root / 'null_summary.csv', index=False)
    (root / 'validation.json').write_text(json.dumps(dict(status='passed',
        model_permutations=3000, each_model_indices='0..999 exactly once',
        observed_comparison='saved baseline summaries only; no refitting',
        checks=['finite null correlations', 'baseline r and d', 'right-tail exceedance P', 'six-test BH'],
        new_fits=0, new_null_tests=0), indent=2)+'\n')
    print(inference.to_string(index=False), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('demographic', 'fixed-score-age-sex', 'permutations'))
    args = parser.parse_args()
    {'demographic': collect_demographic, 'fixed-score-age-sex': fixed_score_age_sex,
     'permutations': verify_permutations}[args.action]()
