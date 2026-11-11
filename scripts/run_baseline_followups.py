#!/usr/bin/env python3
"""Run baseline age/sex sensitivity and matched permutation inference."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import false_discovery_control, pearsonr

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE), str(CODE / 'scripts/molecular_figures/analysis')]
from cca401 import nested as run, preferred, seed_screen as shared
from cca401_union_grid import GRID, PAIRS
import cca401_pipeline as pipeline

ROOT = CODE / 'results/baseline_followups_20261031'
BASELINE = CODE / 'results/cca401_160x23_seed1_retuned_mean_tminus1_20261031'
DEMOGRAPHIC = ROOT / 'age_sex_sensitivity'
PERMUTATIONS = ROOT / 'permutations'
ASSETS = CODE / 'results/baseline_shared_assets_20261031'
MODELS = ('nc', 'fc', 'nc+fc')
COVARIATES = ['lesion_volume', 'scan_age', 'registry_gender']
ORIGINAL_PREPROCESSING = pipeline.preprocessing
PERM_DATA = None
N_PERM = 1000
SEARCH = CODE / 'results/cca401_160_age_sex_multiseed_20261103'
MATCHED_MODELS = CODE / 'results/cca401_160_candidate_matched_models_20261103'
ALL_CONTROLLED_MODELS = CODE / 'results/cca401_160_age_sex_all_models_20261103'
NEAR_CANDIDATES = CODE / 'results/cca401_160_age_sex_near_candidates_20261104'
SEARCH_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42, 100, 111, 123, 555, 666, 888,
                1000, 1111, 1234, 1492, 1776, 1968, 1969, 1977, 1982, 1984, 1991,
                2001, 2019, 2025, 2026, 2049, 2061, 2077, 3001, 11111, 11235, 12321,
                12345, 13579, 16180, 22222, 24680, 27182, 31415, 33333, 44444,
                54321, 99999, 123456)
SEARCH_DESIGNS = ((5, 5), (10, 5), (10, 10), (5, 10))
ORIGINAL_SPLIT = run.prepare_split


def write_json(path, value):
    pipeline.write_json(path, value)


def prepare():
    assert preferred.CURRENT == BASELINE
    ids, pack, y, volume = preferred.load_data()
    table = pd.read_csv(CODE / 'data/MMPI_Demographics.csv').set_index('study_id')
    assert table.index.is_unique
    demographics = table.loc[ids, ['scan_age', 'gender']].to_numpy(float)
    assert np.isfinite(demographics).all() and set(demographics[:, 1]) == {0., 1.}
    confounds = np.column_stack([volume, demographics])
    inputs = ROOT / 'inputs'
    inputs.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(inputs / 'baseline.npz', ids=ids, y=y, volume=volume,
                        confounds=confounds, **pack)
    pd.DataFrame(confounds, index=pd.Index(ids, name='study_id'), columns=COVARIATES).to_csv(
        inputs / 'covariates.csv')
    with np.load(BASELINE / 'inputs/split_plan.npz') as z:
        np.savez_compressed(inputs / 'split_plan.npz', **{name: z[name] for name in z.files})
    protocol = dict(
        status='fixed_before_followups', baseline=str(BASELINE), n=160, n_scales=23,
        seed=1, outer_folds=10, inner_folds=5, grid=GRID.tolist(), grid_pairs=225,
        inner_reference=preferred.INNER_REFERENCE, score_scaling=preferred.SCORE_SCALING,
        baseline_volume_policy='both', baseline_inputs='correctedT-1/Fisher-z arithmetic mean',
        demographic=dict(covariates=COVARIATES, age_definition='age at imaging, scan_age',
            gender_definition='registry gender,0=female,1=male', missing_values=0,
            input_adjustment='linear residualization of both views within every training fold',
            splits='exact baseline outer,inner and independent full-data5fold partitions',
            fresh_tuning=True, education_included=False, rcca_permutations=0),
        permutation=dict(n_per_model=N_PERM, models=list(MODELS), indices=list(range(N_PERM)),
            outer_splits='observed baseline partitions fixed',
            null='shuffle Y rows, then repeat training-only volume adjustment,normalization,inner tuning,alignment,scoring',
            inner_partitions='regenerated from permuted training Y with master seed plus outer-fold index',
            tail='right tail for positive pooled OOF r', multiplicity='BH across6 model-by-dimension tests',
            scope='fixed baseline configuration,not full seed/design/rule selection', demographic_adjustment=False,
            observed_acceptance='first2CD scores,loadings and selected taus must reproduce stored baseline'),
        molecular=dict(n_spin=100000, seed=42, n_boot=1000, pet_fdr='BH10 per CD'))
    write_json(ROOT / 'protocol.json', protocol)
    write_json(ROOT / 'workflow.json', dict(status='prepared', phase='ready'))
    print('Prepared baseline160/23 with complete imaging age and registry gender; no patients dropped.', flush=True)


def load_inputs(adjusted):
    with np.load(ROOT / 'inputs/baseline.npz') as z:
        ids, y = z['ids'], z['y']
        pack = {name: z[name] for name in ('sc_nc', 'fc')}
        covariates = z['confounds'] if adjusted else z['volume']
    assert len(ids) == 160 and y.shape == (160, 23)
    return ids, pack, y, covariates


def demographic_data():
    return load_inputs(True)


def baseline_fit(model):
    path = BASELINE / f'selection/both/{model.replace("+", "_")}/seed_1_5fold/selection.json'
    record = json.loads(path.read_text())
    choice = next(c for c in record['choices'] if c['branch'] == 'B')
    assert choice['reference'] == 'candidate' and choice['score_scaling'] == 'training'
    return BASELINE / choice['fit_directory']


def covariate_preprocessing(x, y, covariates, policy):
    values = ORIGINAL_PREPROCESSING(x, y, covariates, policy)
    return {**{name.replace('_volume_', '_covariate_'): value for name, value in values.items()},
            'covariate_names': np.array(COVARIATES)}


def demographic_full_grid(x, y, covariates, model, policy, seed, k):
    assert (policy, seed, k) == ('both', 1, 5)
    with np.load(ROOT / 'inputs/split_plan.npz') as z:
        splits = [(z[f'full_train_{j}'], z[f'full_test_{j}']) for j in range(1, 6)]
    return pipeline.select_grid_from_splits(x, y, covariates, model, policy, seed, k, splits)


def configure_demographic():
    pipeline.CURRENT, pipeline.SCALES = DEMOGRAPHIC, run.SCALES_23
    pipeline.PAIRS = shared.PAIRS = PAIRS
    pipeline.load_data = demographic_data
    pipeline.select_grid = demographic_full_grid
    pipeline.preprocessing = covariate_preprocessing
    shared.VOLUME_POLICY = 'both'


def demographic_model(model):
    ids, pack, y, covariates = demographic_data()
    x = run.select_features(pack, model)
    cfg = shared.config_for(model, 1, 10, 5)
    folder = DEMOGRAPHIC / f'oof/both/{model.replace("+", "_")}/10x5/seed_1'
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / 'inputs/split_plan.npz') as z:
        outer = [(z[f'outer_train_{k}'], z[f'outer_test_{k}']) for k in range(1, 11)]
        for k in range(1, 11):
            path = folder / f'fold_{k:02d}.npz'
            if path.exists():
                with np.load(path) as saved:
                    np.testing.assert_array_equal(saved['train_index'], outer[k-1][0])
                    np.testing.assert_array_equal(saved['test_index'], outer[k-1][1])
                    np.testing.assert_array_equal(saved['pairs'], PAIRS)
                continue
            inner = [(z[f'inner_train_{k}_{j}'], z[f'inner_test_{k}_{j}']) for j in range(1, 6)]
            result = shared.fold_from_splits(x, y, covariates, outer, k, cfg, inner)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **result)
            temporary.replace(path)
    shared.DATA = ids, pack, y, covariates
    rows = shared.aggregate(folder, model, cfg)
    selection = pipeline.full_task((model, 'both', 1, 5))
    selection.update(brain_covariates=COVARIATES, behavior_covariates=COVARIATES,
                     partition_source='exact baseline160/23/seed1 partitions')
    selection_path = DEMOGRAPHIC / f'selection/both/{model.replace("+", "_")}/seed_1_5fold/selection.json'
    write_json(selection_path, selection)
    for choice in selection['choices']:
        fit = DEMOGRAPHIC / choice['fit_directory']
        meta = json.loads((fit / 'manifest.json').read_text())
        meta.update(brain_covariates=COVARIATES, behavior_covariates=COVARIATES,
                    analysis='age/sex-adjusted baseline sensitivity', baseline_unchanged=True)
        write_json(fit / 'manifest.json', meta)
    choices = {c['branch']: c for c in selection['choices']}
    for row in rows:
        choice = choices[row['branch']]
        row.update(fit_directory=choice['fit_directory'],
                   final_tau_x=choice['tau'][0], final_tau_y=choice['tau'][1],
                   brain_covariates=','.join(COVARIATES), behavior_covariates=','.join(COVARIATES))
    write_json(folder / 'manifest.json', dict(status='complete', n=160, n_scales=23,
        seed=1, design='10x5', model=model, grid_pairs=225,
        brain_covariates=COVARIATES, behavior_covariates=COVARIATES,
        source_kind='fresh_nested_oof', rcca_permutations=0))
    print(f'Demographic sensitivity complete: {model}', flush=True)
    return rows


def demographics():
    configure_demographic()
    DEMOGRAPHIC.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(DEMOGRAPHIC / 'run.json', dict(status='running', covariates=COVARIATES))
    rows = []
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn'),
                             initializer=configure_demographic) as pool:
        for result in pool.map(demographic_model, MODELS):
            rows.extend(result)
    clinical = pd.DataFrame(rows)
    assert len(clinical) == 12 and clinical.n.eq(160).all()
    clinical.to_csv(DEMOGRAPHIC / 'clinical_results.csv', index=False)
    primary = clinical.loc[clinical.branch.eq('B')]
    primary.to_csv(DEMOGRAPHIC / 'primary_clinical.csv', index=False)
    diagnostics = []
    for record in primary.to_dict('records'):
        original = baseline_fit(record['model'])
        with np.load(original / 'model.npz') as a, np.load(DEMOGRAPHIC / record['fit_directory'] / 'model.npz') as b:
            correlations = np.corrcoef(a['behavior'].T, b['behavior'].T)[:10, 10:]
            old, new = linear_sum_assignment(-abs(correlations))
            for i, j in zip(old, new):
                diagnostics.append(dict(model=record['model'], baseline_dimension=int(i+1),
                    adjusted_dimension=int(j+1), behavior_loading_r=float(correlations[i, j])))
    pd.DataFrame(diagnostics).to_csv(DEMOGRAPHIC / 'baseline_loading_correspondence.csv', index=False)
    assets = DEMOGRAPHIC / 'assets'
    if not assets.exists():
        assets.symlink_to(ASSETS, target_is_directory=True)
    import current_molecular as molecular
    molecular.CURRENT, molecular.N_SPIN, molecular.N_BOOT, molecular.SEED = DEMOGRAPHIC, 100000, 1000, 42
    fit = primary.loc[primary.model.eq('nc+fc'), 'fit_directory'].item()
    result = molecular.molecular_task(fit)
    write_json(DEMOGRAPHIC / 'run.json', dict(status='complete', n=160, n_scales=23,
        covariates=COVARIATES, clinical_rows=12, molecular=result,
        elapsed_seconds=time.monotonic()-started, rcca_permutations=0))


def initialize_permutations():
    global PERM_DATA
    ids, pack, y, volume = load_inputs(False)
    with np.load(ROOT / 'inputs/split_plan.npz') as z:
        outer = [(z[f'outer_train_{k}'], z[f'outer_test_{k}']) for k in range(1, 11)]
    PERM_DATA = ids, pack, y, volume, outer


def permutation_task(task):
    model, index = task
    ids, pack, original_y, volume, outer = PERM_DATA
    cfg = replace(shared.config_for(model, 1, 10, 5), tau_grid=tuple(GRID))
    order = np.random.RandomState(index).permutation(160) if index >= 0 else np.arange(160)
    y = original_y[order]
    x = run.select_features(pack, model)
    started = time.monotonic()
    pieces = [run.calculate_fold(ids, x, y, volume, outer, k, 'both', cfg, 'training') for k in range(1, 11)]
    xs, ys = [r['x_score'] for r in pieces], [r['y_score'] for r in pieces]
    wx, wy = [r['x_weight'] for r in pieces], [r['y_weight'] for r in pieces]
    run.harmonize_outer_component_signs(xs, ys, wx, wy, 2)
    scores_x, scores_y = np.concatenate(xs), np.concatenate(ys)
    rs = shared.paired_r(scores_x[:, :2], scores_y[:, :2])
    selected = [[float(r['tau_x']), float(r['tau_y'])] for r in pieces]
    if index < 0:
        folder = BASELINE / f'oof/both/{model.replace("+", "_")}/10x5/seed_1'
        with np.load(folder / 'B_aligned.npz') as saved:
            for key, value in [('x_score', scores_x), ('y_score', scores_y),
                               ('brain', np.stack(wx)), ('behavior', np.stack(wy))]:
                np.testing.assert_allclose(value[..., :2], saved[key][..., :2], rtol=1e-7, atol=1e-9)
        parameters = pd.read_csv(folder / 'selected_parameters.csv').query("branch=='B'").sort_values('outer_fold')
        np.testing.assert_array_equal(selected, parameters[['tau_x', 'tau_y']].to_numpy())
    record = dict(model=model, permutation_index=index, r_1=float(rs[0]), r_2=float(rs[1]),
                  selected_pairs=selected, elapsed_seconds=time.monotonic()-started)
    target = PERMUTATIONS / ('observed_acceptance' if index < 0 else 'nulls') / model.replace('+', '_') / f'{index:04d}.json'
    write_json(target, record)
    return record


def permutations():
    PERMUTATIONS.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(PERMUTATIONS / 'run.json', dict(status='observed_acceptance_running', workers=6))
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn'),
                             initializer=initialize_permutations) as pool:
        observed = list(pool.map(permutation_task, [(m, -1) for m in MODELS]))
        write_json(PERMUTATIONS / 'acceptance.json', dict(status='passed', models=observed,
            check='all30 observed-fold selected taus and evaluatedCD1/CD2 scores/loadings reproduce baseline'))
        jobs = [(model, i) for i in range(N_PERM) for model in MODELS
                if not (PERMUTATIONS / 'nulls' / model.replace('+', '_') / f'{i:04d}.json').exists()]
        completed = 3000-len(jobs)
        write_json(PERMUTATIONS / 'run.json', dict(status='permutations_running', n_perm=1000,
            total_model_permutations=3000, complete=completed, workers=6))
        for future in as_completed([pool.submit(permutation_task, job) for job in jobs]):
            result = future.result()
            completed += 1
            write_json(PERMUTATIONS / 'progress.json', dict(total=3000, complete=completed,
                elapsed_seconds=time.monotonic()-started))
            print(f'Permutation {completed}/3000: {result["model"]}, index={result["permutation_index"]}', flush=True)
    nulls = pd.DataFrame(json.loads(p.read_text()) for p in (PERMUTATIONS / 'nulls').glob('*/*.json'))
    assert len(nulls) == 3000 and not nulls.duplicated(['model', 'permutation_index']).any()
    nulls.to_csv(PERMUTATIONS / 'permutation_nulls.csv', index=False)
    rows = []
    for record in observed:
        for j in (1, 2):
            values = nulls.loc[nulls.model.eq(record['model']), f'r_{j}'].to_numpy()
            assert len(values) == 1000
            r = record[f'r_{j}']
            rows.append(dict(model=record['model'], dimension=j, observed_r=r,
                cohens_d=float(2*r/np.sqrt(1-r*r)), p_permutation=float((sum(values >= r)+1)/1001)))
    inference = pd.DataFrame(rows)
    inference['q_fdr_6'] = false_discovery_control(inference.p_permutation.to_numpy())
    inference.to_csv(PERMUTATIONS / 'inference.csv', index=False)
    write_json(PERMUTATIONS / 'run.json', dict(status='complete', n_perm=1000,
        tests=6, model_permutations=3000, elapsed_seconds=time.monotonic()-started,
        scope='fixed baseline,not adjustment for candidate exploration'))


def controlled_split(x, y, covariates, train, test, policy, config):
    """Age and sex on both sides; volume on the side(s) specified by policy."""
    if policy == 'both':
        return ORIGINAL_SPLIT(x, y, covariates, train, test, policy, config)
    if policy != 'brain_only':
        raise ValueError(policy)
    x_test = None if test is None else x[test]
    y_test = None if test is None else y[test]
    x_train, x_test = run.residualize_train_test(
        x[train], covariates[train], x_test, None if test is None else covariates[test])
    y_train, y_test = run.residualize_train_test(
        y[train], covariates[train, 1:], y_test, None if test is None else covariates[test, 1:])
    x_train, x_mean, x_sd = run.preprocess(x_train, config.xnorm)
    y_train, y_mean, y_sd = run.preprocess(y_train, config.ynorm)
    if test is not None:
        x_test = (x_test - x_mean) / x_sd
        y_test = (y_test - y_mean) / y_sd
    return run.PreparedSplit(x_train, y_train, x_test, y_test)


def controlled_preprocessing(x, y, covariates, policy):
    from sklearn.linear_model import LinearRegression
    values = {}
    for side, data in (('x', x), ('y', y)):
        cols = slice(None) if side == 'x' or policy == 'both' else slice(1, None)
        design = covariates[:, cols]
        reg = LinearRegression(fit_intercept=True).fit(design, data)
        _, mean, sd = run.preprocess(data - reg.predict(design), 'norm')
        values.update({f'{side}_covariate_coef': reg.coef_,
                       f'{side}_covariate_intercept': reg.intercept_,
                       f'{side}_mean': mean, f'{side}_sd': sd,
                       f'{side}_covariate_names': np.array(COVARIATES if side == 'x' or policy == 'both' else COVARIATES[1:])})
    return values


def configure_search(output=SEARCH):
    pipeline.CURRENT, pipeline.SCALES = output, run.SCALES_23
    pipeline.PAIRS = shared.PAIRS = PAIRS
    pipeline.load_data = demographic_data
    pipeline.preprocessing = controlled_preprocessing
    run.prepare_split = controlled_split
    shared.DATA = demographic_data()


def search_oof(task, model='nc+fc'):
    policy, seed, no, ni = task
    shared.VOLUME_POLICY = policy
    ids, pack, y, covariates = shared.DATA
    x = run.select_features(pack, model)
    cfg = shared.config_for(model, seed, no, ni)
    folder = pipeline.CURRENT / f'oof/{policy}/{model.replace("+", "_")}/{no}x{ni}/seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        return pd.read_csv(folder / 'summary.csv').to_dict('records')
    outer = run.make_splits(y, no, seed, cfg.y_bins, cfg.y_pc)
    for k in range(1, no + 1):
        path = folder / f'fold_{k:02d}.npz'
        if not path.exists():
            result = shared.fold_shared(x, y, covariates, outer, k, cfg)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **result)
            temporary.replace(path)
    rows = shared.aggregate(folder, model, cfg)
    registry_sex = pd.Series(load_inputs(True)[3][:, 2], index=ids)
    for row in rows:
        with np.load(folder / f"{row['branch']}_aligned.npz") as aligned:
            sex = registry_sex.loc[aligned['ids']].to_numpy()
            for label, value in (('F', 0), ('M', 1)):
                for j in (0, 1):
                    row[f'sex_{label}_r_{j+1}'] = float(pearsonr(
                        aligned['x_score'][sex == value, j], aligned['y_score'][sex == value, j]).statistic)
    pd.DataFrame(rows).to_csv(folder / 'summary.csv', index=False)
    brain_covariates = COVARIATES if covariates.shape[1] == 3 else ['lesion_volume']
    behavior_covariates = brain_covariates if policy == 'both' else brain_covariates[1:]
    write_json(folder / 'manifest.json', dict(status='complete', source_kind='fresh_nested_oof',
        policy=policy, seed=seed, outer_folds=no, inner_folds=ni, n=160, grid_pairs=225,
        brain_covariates=brain_covariates, behavior_covariates=behavior_covariates))
    return rows


def search_independent_oof(task):
    policy, seed, no, ni, model = task
    return search_oof((policy, seed, no, ni), model)


def all_controlled_models():
    configure_search(ALL_CONTROLLED_MODELS)
    ALL_CONTROLLED_MODELS.mkdir(parents=True, exist_ok=True)
    jobs = [(p, s, no, ni, model) for p in ('both', 'brain_only') for s in SEARCH_SEEDS
            for no, ni in SEARCH_DESIGNS for model in ('nc', 'fc')]
    write_json(ALL_CONTROLLED_MODELS / 'workflow.json', dict(status='running', phase='clinical',
        jobs_total=len(jobs), models=['nc', 'fc'], n=160, n_scales=23, grid_pairs=225,
        age_sex_adjusted=True, combined_source=str(SEARCH / 'clinical_molecular_summary.csv')))
    rows = []
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn'),
                             initializer=configure_search, initargs=(ALL_CONTROLLED_MODELS,)) as pool:
        for future in as_completed([pool.submit(search_independent_oof, job) for job in jobs]):
            result = future.result()
            rows.extend(result)
            write_json(ALL_CONTROLLED_MODELS / 'progress.json', dict(completed=len(rows)//4, total=len(jobs)))
            print(f"Independent models: {len(rows)//4}/{len(jobs)}; {result[0]['model']}, "
                  f"seed {result[0]['seed']}, {result[0]['design']}, {result[0]['volume_policy']}", flush=True)
    assert len(rows) == 3264
    clinical = pd.DataFrame(rows).sort_values(['model', 'volume_policy', 'seed', 'outer_folds', 'inner_folds', 'branch'])
    clinical.to_csv(ALL_CONTROLLED_MODELS / 'independent_clinical.csv', index=False)
    combined = pd.read_csv(SEARCH / 'clinical_molecular_summary.csv')
    keys = ['seed', 'outer_folds', 'inner_folds', 'volume_policy', 'reference', 'score_scaling']
    assert not combined.duplicated(keys).any() and len(combined) == 1632
    for model in ('nc', 'fc'):
        part = clinical.loc[clinical.model.eq(model)].copy()
        assert not part.duplicated(keys).any() and len(part) == 1632
        columns = keys + ['r_1', 'r_2', 'd_1', 'd_2', 'brain_icc_1', 'brain_icc_2',
                          'behavior_icc_1', 'behavior_icc_2', 'semantic_swap_flag', 'min_behavior_similarity']
        part = part[columns].rename(columns={name: f'{model}_{name}' for name in columns if name not in keys})
        combined = combined.merge(part, on=keys, validate='one_to_one')
    combined.to_csv(ALL_CONTROLLED_MODELS / 'all_models_clinical_molecular.csv', index=False)
    write_json(ALL_CONTROLLED_MODELS / 'workflow.json', dict(status='complete', phase='ready_for_candidate_review',
        jobs_total=len(jobs), independent_rows=len(clinical), matched_settings=len(combined)))


def configure_near(adjusted):
    out = NEAR_CANDIDATES / ('age_sex_adjusted' if adjusted else 'volume_only')
    pipeline.CURRENT, pipeline.SCALES = out, run.SCALES_23
    pipeline.PAIRS = shared.PAIRS = PAIRS
    pipeline.load_data = demographic_data if adjusted else volume_data
    pipeline.preprocessing = controlled_preprocessing if adjusted else ORIGINAL_PREPROCESSING
    run.prepare_split = controlled_split if adjusted else ORIGINAL_SPLIT
    shared.DATA = pipeline.load_data()


def volume_data():
    return load_inputs(False)


def near_oof(task):
    policy, model = task
    return search_oof((policy, 33333, 10, 10), model)


def near_full(task):
    policy, model = task
    return pipeline.full_task((model, policy, 33333, 10))


def near_candidates():
    NEAR_CANDIDATES.mkdir(parents=True, exist_ok=True)
    write_json(NEAR_CANDIDATES / 'workflow.json', dict(status='running', phase='refits'))
    for adjusted in (True, False):
        configure_near(adjusted)
        jobs = [(p, m) for p in ('brain_only', 'both') for m in MODELS]
        with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn'),
                                 initializer=configure_near, initargs=(adjusted,)) as pool:
            pending = [pool.submit(near_full, task) for task in jobs]
            if not adjusted:
                pending += [pool.submit(near_oof, task) for task in jobs]
            for i, future in enumerate(as_completed(pending), 1):
                future.result()
                print(f'Adjusted={adjusted}: {i}/{len(pending)} tasks complete', flush=True)
    import current_molecular as molecular
    molecular.CURRENT = NEAR_CANDIDATES / 'volume_only'
    molecular.N_SPIN, molecular.N_BOOT, molecular.SEED = 100000, 1000, 42
    assets = molecular.CURRENT / 'assets'
    if not assets.exists():
        assets.symlink_to(ASSETS, target_is_directory=True)
    for policy, branch in (('brain_only', 'C'), ('both', 'B')):
        selection = json.loads((molecular.CURRENT / f'selection/{policy}/nc_fc/seed_33333_10fold/selection.json').read_text())
        fit = next(c['fit_directory'] for c in selection['choices'] if c['branch'] == branch)
        molecular.molecular_task(fit)
        print(f'Unadjusted molecular evaluation complete: {policy}', flush=True)
    write_json(NEAR_CANDIDATES / 'workflow.json', dict(status='complete', phase='ready_for_candidate_review'))


def search_full(task):
    policy, seed, k = task
    return pipeline.full_task(('nc+fc', policy, seed, k))


def init_search_molecular():
    configure_search()
    import current_molecular as molecular
    molecular.CURRENT, molecular.N_SPIN, molecular.N_BOOT, molecular.SEED = SEARCH, 100000, 1000, 42


def search_molecular(fit):
    import current_molecular as molecular
    return molecular.molecular_task(fit)


def matched_model_oof(task):
    adjusted, policy, seed, model = task
    run.prepare_split = controlled_split if adjusted else ORIGINAL_SPLIT
    shared.VOLUME_POLICY = policy
    shared.PAIRS = PAIRS
    ids, pack, y, covariates = load_inputs(adjusted)
    shared.DATA = ids, pack, y, covariates
    x = run.select_features(pack, model)
    cfg = shared.config_for(model, seed, 10, 5)
    label = 'age_sex_adjusted' if adjusted else 'volume_only'
    folder = MATCHED_MODELS / f'{label}/{policy}/{model.replace("+", "_")}/10x5/seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        return pd.read_csv(folder / 'summary.csv').to_dict('records')
    outer = run.make_splits(y, 10, seed, cfg.y_bins, cfg.y_pc)
    for k in range(1, 11):
        path = folder / f'fold_{k:02d}.npz'
        if not path.exists():
            result = shared.fold_shared(x, y, covariates, outer, k, cfg)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **result)
            temporary.replace(path)
    rows = shared.aggregate(folder, model, cfg)
    write_json(folder / 'manifest.json', dict(status='complete', model=model, seed=seed,
        age_sex_adjusted=adjusted, volume_policy=policy, design='10x5', grid_pairs=225))
    return rows


def matched_models():
    MATCHED_MODELS.mkdir(parents=True, exist_ok=True)
    settings = ((3, 'brain_only'), (13579, 'both'))
    jobs = [(adjusted, policy, seed, model) for seed, policy in settings
            for adjusted in (False, True) for model in MODELS
            if not (adjusted and model == 'nc+fc')]
    write_json(MATCHED_MODELS / 'workflow.json', dict(status='running', jobs_total=len(jobs)))
    rows = []
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(matched_model_oof, job): job for job in jobs}
        for future in as_completed(futures):
            adjusted, policy, seed, model = futures[future]
            for row in future.result():
                row['age_sex_adjusted'] = adjusted
                rows.append(row)
            write_json(MATCHED_MODELS / 'progress.json', dict(completed=len(rows)//4, total=len(jobs)))
            print(f'Completed {len(rows)//4}/{len(jobs)}: {model}, seed {seed}, adjusted={adjusted}', flush=True)
    prior = pd.read_csv(SEARCH / 'clinical_results.csv')
    for seed, policy in settings:
        part = prior[(prior.seed == seed) & (prior.volume_policy == policy) & (prior.design == '10x5')].copy()
        part['age_sex_adjusted'] = True
        rows.extend(part.to_dict('records'))
    result = pd.DataFrame(rows)
    assert len(result) == 48
    result.to_csv(MATCHED_MODELS / 'clinical_results.csv', index=False)
    write_json(MATCHED_MODELS / 'workflow.json', dict(status='complete', jobs_total=len(jobs),
        rows=len(result), adjusted_combined_source=str(SEARCH / 'clinical_results.csv')))


def multiseed_search():
    configure_search()
    SEARCH.mkdir(parents=True, exist_ok=True)
    write_json(SEARCH / 'protocol.json', dict(n=160, n_scales=23, model='nc+fc',
        seeds=list(SEARCH_SEEDS), designs=[list(x) for x in SEARCH_DESIGNS],
        volume_policies=['both', 'brain_only'], brain_covariates=COVARIATES,
        behavior_covariates={'both': COVARIATES, 'brain_only': COVARIATES[1:]},
        references=['candidate', 'fixed_primary'], score_scaling=['raw', 'training'],
        grid_pairs=225, n_spin=100000, n_boot=1000, spin_seed=42, rcca_permutations=0))
    oof_jobs = [(p, s, no, ni) for p in ('both', 'brain_only') for s in SEARCH_SEEDS for no, ni in SEARCH_DESIGNS]
    full_jobs = [(p, s, k) for p in ('both', 'brain_only') for s in SEARCH_SEEDS for k in (5, 10)]
    write_json(SEARCH / 'workflow.json', dict(status='running', phase='clinical', oof_total=len(oof_jobs), full_total=len(full_jobs)))
    rows, selections = [], []
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn'), initializer=configure_search) as pool:
        jobs = {pool.submit(search_oof, job): 'oof' for job in oof_jobs}
        jobs.update({pool.submit(search_full, job): 'full' for job in full_jobs})
        for future in as_completed(jobs):
            if jobs[future] == 'oof':
                rows.extend(future.result())
            else:
                selections.append(future.result())
            write_json(SEARCH / 'clinical_progress.json', dict(oof_complete=len(rows)//4,
                oof_total=len(oof_jobs), full_complete=len(selections), full_total=len(full_jobs)))
            print(f'Clinical: {len(rows)//4}/{len(oof_jobs)} OOF, {len(selections)}/{len(full_jobs)} full', flush=True)
    assert len(rows) == 1632 and len(selections) == 204
    choices = {(s['volume_policy'], s['cv_seed'], s['cv_folds'], c['branch']): c
               for s in selections for c in s['choices']}
    for row in rows:
        choice = choices[(row['volume_policy'], row['seed'], row['inner_folds'], row['branch'])]
        row.update(fit_directory=choice['fit_directory'], final_tau_x=choice['tau'][0], final_tau_y=choice['tau'][1])
    clinical = pd.DataFrame(rows).sort_values(['volume_policy', 'seed', 'outer_folds', 'inner_folds', 'branch'])
    clinical.to_csv(SEARCH / 'clinical_results.csv', index=False)
    assets = SEARCH / 'assets'
    if not assets.exists():
        assets.symlink_to(ASSETS, target_is_directory=True)
    fits = sorted(clinical.fit_directory.unique())
    write_json(SEARCH / 'workflow.json', dict(status='running', phase='molecular', fits_total=len(fits)))
    completed = []
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn'), initializer=init_search_molecular) as pool:
        for future in as_completed([pool.submit(search_molecular, fit) for fit in fits]):
            completed.append(future.result())
            write_json(SEARCH / 'molecular_progress.json', dict(complete=len(completed), total=len(fits)))
            print(f'Molecular: {len(completed)}/{len(fits)}', flush=True)
    write_json(SEARCH / 'workflow.json', dict(status='complete', phase='ready_for_candidate_review',
        clinical_rows=len(clinical), unique_molecular_fits=len(fits), rcca_permutations=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'demographics', 'permutations', 'multiseed', 'matched-models', 'all-controlled-models', 'near-candidates'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
        return
    try:
        {'demographics': demographics, 'permutations': permutations, 'multiseed': multiseed_search,
         'matched-models': matched_models, 'all-controlled-models': all_controlled_models,
         'near-candidates': near_candidates}[args.action]()
    except Exception:
        target = (NEAR_CANDIDATES / 'workflow.json' if args.action == 'near-candidates' else
                  ALL_CONTROLLED_MODELS / 'workflow.json' if args.action == 'all-controlled-models' else
                  MATCHED_MODELS / 'workflow.json' if args.action == 'matched-models' else
                  SEARCH / 'workflow.json' if args.action == 'multiseed' else
                  (DEMOGRAPHIC if args.action == 'demographics' else PERMUTATIONS) / 'run.json')
        write_json(target, dict(status='failed', phase=args.action, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
