"""Extend the validated 121-pair screen to the author's 225-pair union grid.

Old grid metrics and matches are deterministic reusable calculations. New pairs
are fitted with the same native solver, preprocessing, splits and four rules.
Every selected outer model is projected against its outer-training anchor.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import cca401_pipeline as p

OLD_CURRENT = p.CODE/'results/cca401_current'
CURRENT = p.CODE/'results/cca401_union225'
OLD_PAIRS = p.PAIRS.copy()
GRID = np.array(sorted(set([round(i/10,1) for i in range(11)] +
    [0.,.001,.01,.1,.3,.5,.7,.9,.99,.999,1.])))
PAIRS = np.array([(a,b) for a in GRID for b in GRID])
OLD_POSITIONS = np.array([np.flatnonzero(np.all(PAIRS==pair,axis=1))[0] for pair in OLD_PAIRS])
NEW_POSITIONS = np.setdiff1d(np.arange(len(PAIRS)),OLD_POSITIONS)


def configure():
    """Set the explicit analysis context in the parent and each spawned worker."""
    import current_molecular as molecular
    import run_cca401_current as runner
    p.CURRENT=CURRENT
    p.PAIRS=PAIRS
    p.shared.PAIRS=PAIRS
    molecular.CURRENT=CURRENT
    runner.OUT=CURRENT


def extend_fold(x,y,volume,outer,k,cfg,policy,original):
    started=time.monotonic()
    np.testing.assert_array_equal(original['pairs'],OLD_PAIRS)
    tr,te=outer[k-1]
    np.testing.assert_array_equal(original['train_index'],tr)
    np.testing.assert_array_equal(original['test_index'],te)
    inner=p.run.make_splits(y[tr],cfg.inner_folds,cfg.seed+k,cfg.y_bins,cfg.y_pc)
    np.testing.assert_array_equal(original['inner_valid_flat'],np.concatenate([b for a,b in inner]))
    prepared=p.run.prepare_split(x,y,volume,tr,te,policy,cfg)
    inner_data=[p.run.prepare_split(x[tr],y[tr],volume[tr],a,b,policy,cfg) for a,b in inner]
    anchor=p.shared.fit_bundle(prepared,*cfg.resolved_primary_params)
    full={tuple(cfg.resolved_primary_params):anchor}
    grid=np.empty((4,len(PAIRS),2))
    indices=np.empty((2,len(PAIRS),cfg.inner_folds,10),dtype=np.int16)
    signs=np.empty_like(indices,dtype=np.int8)
    matches=np.empty_like(indices,dtype=float)
    grid[:,OLD_POSITIONS]=original['grid_r']
    indices[:,OLD_POSITIONS]=original['inner_indices']
    signs[:,OLD_POSITIONS]=original['inner_signs']
    matches[:,OLD_POSITIONS]=original['inner_matches']
    for ip in NEW_POSITIONS:
        pair=PAIRS[ip]
        candidate=p.shared.fit_bundle(prepared,*pair)
        full[tuple(pair)]=candidate
        pieces=[[] for _ in p.BRANCHES]
        for ik,data in enumerate(inner_data):
            fitted=p.shared.fit_bundle(data,*pair)
            for ir,reference in enumerate((candidate['wy'],anchor['wy'])):
                match=p.shared.mapping(reference,fitted['wy'])
                indices[ir,ip,ik],signs[ir,ip,ik],matches[ir,ip,ik]=match
                for iz,scale in enumerate(('raw','training')):
                    result=p.shared.apply_mapping(fitted,match,scale)
                    pieces[2*ir+iz].append((result['x'][:,:2],result['y'][:,:2]))
        for ib,parts in enumerate(pieces):
            grid[ib,ip]=p.shared.paired_r(np.concatenate([a for a,b in parts]),
                                         np.concatenate([b for a,b in parts]))
    winners=np.argmax(grid.mean(axis=2),axis=1)
    assert np.all(grid.mean(axis=2).max(axis=1)>=original['grid_r'].mean(axis=2).max(axis=1)-1e-12)
    selected=[]
    for ib,ip in enumerate(winners):
        pair=tuple(PAIRS[ip])
        if pair not in full:full[pair]=p.shared.fit_bundle(prepared,*pair)
        selected.append(p.shared.project(full[pair],anchor['wy'],p.BRANCHES[ib][2]))
    output=dict(test_index=te,train_index=tr,pairs=PAIRS,grid_r=grid,winners=winners,
        inner_indices=indices,inner_signs=signs,inner_matches=matches,
        inner_train_indices=original['inner_train_indices'],
        inner_valid_flat=original['inner_valid_flat'],inner_valid_lengths=original['inner_valid_lengths'],
        elapsed=np.array(time.monotonic()-started))
    for key in selected[0]:output[key]=np.stack([v[key] for v in selected])
    return output


def oof_task(task):
    model,policy,seed,design=task
    folder=CURRENT/'oof'/policy/model.replace('+','_')/design/f'seed_{seed}'
    manifest=folder/'manifest.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        assert record['status']=='complete' and record['grid_pairs']==225
        return pd.read_csv(folder/'summary.csv').to_dict('records')
    no,ni=map(int,design.split('x'))
    cfg=p.shared.config_for(model,seed,no,ni)
    ids,pack,y,volume=p.load_data()
    x=p.run.select_features(pack,model)
    source=p.original_oof_path(*task)
    old_meta=json.loads((source/'metadata.json').read_text())
    for key,value in dict(model=model,seed=seed,outer_folds=no,inner_folds=ni,n=160,volume_policy=policy).items():
        assert old_meta[key]==value
    np.testing.assert_array_equal(old_meta['grid'],OLD_PAIRS)
    outer=p.run.make_splits(y,no,seed,cfg.y_bins,cfg.y_pc)
    folder.mkdir(parents=True,exist_ok=True)
    p.write_json(folder/'metadata.json',{**old_meta,'grid':PAIRS.tolist(),
        'old_metrics_source':str(source),'reused_pairs':121,'new_pairs':104})
    started=time.monotonic()
    for k in range(1,no+1):
        target=folder/f'fold_{k:02d}.npz'
        if target.exists():continue
        with np.load(source/f'fold_{k:02d}.npz') as z:original={n:z[n] for n in z.files}
        result=extend_fold(x,y,volume,outer,k,cfg,policy,original)
        temporary=target.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**result);temporary.replace(target)
    p.shared.DATA=(ids,pack,y,volume)
    p.shared.VOLUME_POLICY=policy
    rows=p.shared.aggregate(folder,model,cfg)
    # Aggregate verifies that the 160 test positions occur exactly once.
    p.write_json(manifest,dict(status='complete',source_kind='nested_oof_union225',
        model=model,volume_policy=policy,seed=seed,design=design,n=160,grid_pairs=225,
        reused_pairs=121,new_pairs=104,source_directory=str(source),
        elapsed_seconds=time.monotonic()-started,rcca_permutations=0))
    return rows
