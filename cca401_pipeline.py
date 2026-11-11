"""Current rCCA: nested evaluation plus independently tuned full-data refits.

Alignment anchors label components. Only full-data-CV-selected fits supply
anatomical and molecular maps. No coefficient thresholding or resampling tests.
"""
from __future__ import annotations

import json
import fcntl
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
BASE = ROOT/'review_prep/collaborator_reviews_JCG_ADB_0810/supplementary_analysis_0810'
from cca401 import seed_screen as shared
from cca401 import nested as run

CURRENT = CODE/'results/cca401_current'
BRANCHES = shared.BRANCHES
PAIRS = shared.PAIRS.copy()
SCALES = run.SCALES_23
ANCHORS = run.make_config
SOURCE_ROOTS = {'brain_only':BASE/'cca401_seed_screen_20261028',
                'both':BASE/'cca401_seed_screen_both_20261028'}


def write_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def load_data():
    ids,pack,y = run.load_data(run.SCALES_23,CODE/'data/Iowa_Lesion_MMPI_LNM_features.npz',
                             CODE/'mmpi/Iowa_MMPI2RF_byNandi.csv')
    volume = run.load_total_volumes(CODE/'data/Iowa_Lesion_MMPI_LesionSize.csv',ids)
    assert len(ids)==len(np.unique(ids))==160 and y.shape==(160,23)
    return ids,pack,y,volume


def cfg_for(model,seed,k):
    return shared.config_for(model,int(seed),10,int(k),scales=SCALES)


def select_grid(x,y,v,model,policy,seed,k):
    """Run the same four inner selection rules, now with all data as training.

    This does not reuse any outer-test statistic or historical final pair.
    The explicit seed is the final selection CV seed, without an outer offset.
    """
    cfg=cfg_for(model,seed,k)
    splits=run.make_splits(y,k,seed,cfg.y_bins,cfg.y_pc)
    return select_grid_from_splits(x,y,v,model,policy,seed,k,splits)


def select_grid_from_splits(x,y,v,model,policy,seed,k,splits):
    """Tune on an explicitly specified full-data CV partition."""
    cfg=cfg_for(model,seed,k)
    prepared=[run.prepare_split(x,y,v,tr,te,policy,cfg) for tr,te in splits]
    all_indices=np.arange(len(y))
    full_data=run.prepare_split(x,y,v,all_indices,all_indices,policy,cfg)
    full=[shared.fit_bundle(full_data,*pair) for pair in PAIRS]
    fixed_i=int(np.flatnonzero(np.all(PAIRS==cfg.resolved_primary_params,axis=1))[0])
    fixed=full[fixed_i]['wy']
    grid=np.empty((4,len(PAIRS),2))
    matches=np.empty((2,len(PAIRS),k,10),dtype=np.int16)
    signs=np.empty_like(matches,dtype=np.int8)
    correlations=np.empty_like(matches,dtype=float)
    for ip,pair in enumerate(PAIRS):
        pieces=[[] for _ in BRANCHES]
        references=(full[ip]['wy'],fixed)
        for ik,p in enumerate(prepared):
            fitted=shared.fit_bundle(p,*pair)
            for ir,reference in enumerate(references):
                match=shared.mapping(reference,fitted['wy'])
                matches[ir,ip,ik],signs[ir,ip,ik],correlations[ir,ip,ik]=match
                for iz,scale in enumerate(('raw','training')):
                    result=shared.apply_mapping(fitted,match,scale)
                    pieces[2*ir+iz].append((result['x'][:,:2],result['y'][:,:2]))
        for ib,parts in enumerate(pieces):
            grid[ib,ip]=shared.paired_r(np.concatenate([p[0] for p in parts]),
                                       np.concatenate([p[1] for p in parts]))
    winners=np.argmax(grid.mean(axis=2),axis=1)
    return dict(cfg=cfg,splits=splits,full=full,fixed=fixed,grid_r=grid,winners=winners,
                matching_indices=matches,matching_signs=signs,matching_r=correlations,
                prepared=full_data)


def pair_name(pair):
    return '__'.join(str(float(c)).replace('.','p') for c in pair)


def preprocessing(x,y,v,policy):
    arrays={}
    for side,data in [('x',x),('y',y)]:
        if side=='x' or policy=='both':
            reg=LinearRegression(fit_intercept=True).fit(v,data)
            residual=data-reg.predict(v)
            arrays[f'{side}_volume_coef']=reg.coef_
            arrays[f'{side}_volume_intercept']=reg.intercept_
        else:
            residual=data.copy()
            arrays[f'{side}_volume_coef']=np.zeros((data.shape[1],1))
            arrays[f'{side}_volume_intercept']=np.zeros(data.shape[1])
        _,mean,sd=run.preprocess(residual,'norm')
        arrays[f'{side}_mean']=mean;arrays[f'{side}_sd']=sd
    return arrays


def save_final_fit(model,policy,pair,bundle,reference,ids,x,y,v):
    folder=CURRENT/'fits'/policy/model.replace('+','_')/pair_name(pair)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.write.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _save_final_fit(folder,model,policy,pair,bundle,reference,ids,x,y,v)


def _save_final_fit(folder,model,policy,pair,bundle,reference,ids,x,y,v):
    # One cache per deterministic full-data model, not per CV branch or seed.
    projected=shared.project(bundle,reference,'raw')
    if (folder/'manifest.json').exists():
        previous=np.load(folder/'model.npz')
        np.testing.assert_allclose(previous['brain'],projected['wx'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(previous['behavior'],projected['wy'],rtol=1e-12,atol=1e-12)
        return folder
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'model.npz',brain=projected['wx'],behavior=projected['wy'],
        raw_brain=bundle['wx'],raw_behavior=bundle['wy'],ids=ids,
        x_score=projected['x'],y_score=projected['y'],
        component_index=projected['index'],component_sign=projected['sign'],
        component_reference_r=projected['match'],tau=np.array(pair),
        **preprocessing(x,y,v,policy))
    meta=pd.read_csv(CODE/'data/reference/ahba/roi_metadata_116.csv').sort_values('roi_id')
    rows=[]
    for j in (0,1):
        modalities=[('sdc',0),('fdc',116)] if model=='nc+fc' else [('sdc' if model=='nc' else 'fdc',0)]
        for modality,start in modalities:
            for roi,value in zip(meta.roi_id,projected['wx'][start:start+116,j]):
                rows.append(dict(map_id=f'{"combined" if model=="nc+fc" else "independent"}_cd{j+1}_{modality}',
                                 dimension=j+1,modality=modality,roi_id=int(roi),weight=float(value)))
    maps=pd.DataFrame(rows).merge(meta,on='roi_id',validate='many_to_one')
    maps.to_csv(folder/'maps116.csv',index=False)
    regions=[]
    for j in (0,1):
        for rank,i in enumerate(np.argsort(abs(projected['wx'][:,j]))[::-1][:20],1):
            roi=meta.iloc[int(i)%116]
            regions.append(dict(dimension=j+1,rank=rank,modality='sdc' if model=='nc' or (model=='nc+fc' and i<116) else 'fdc',
                roi_id=int(roi.roi_id),label=roi.label,network=roi.network,weight=float(projected['wx'][i,j])))
    pd.DataFrame(regions).to_csv(folder/'top20_regions.csv',index=False)
    pd.DataFrame(projected['wy'][:,:2],index=SCALES,columns=['CD1','CD2']).to_csv(folder/'symptom_loadings.csv')
    write_json(folder/'manifest.json',dict(status='complete',source_kind='full_data_cv_selected_fit',
        solver='cca_zoo v4.0.1 RidgeCCA unmodified',model=model,volume_policy=policy,
        tau=[float(c) for c in pair],n=len(ids),scales=list(SCALES),n_components=10,coefficient_filter='none',
        alignment_anchor=list(cfg_for(model,1111,5).resolved_primary_params),
        anchor_role='component labels only; map values from selected fit',rcca_permutations=0))
    return folder


def full_task(task):
    model,policy,seed,k=task
    folder=CURRENT/'selection'/policy/model.replace('+','_')/f'seed_{seed}_{k}fold'
    if (folder/'selection.json').exists():
        value=json.loads((folder/'selection.json').read_text())
        assert value['source_kind']=='full_data_cv_refit' and value['cv_seed']==seed and value['cv_folds']==k
        return value
    started=time.monotonic();ids,pack,y,v=load_data();x=run.select_features(pack,model)
    result=select_grid(x,y,v,model,policy,seed,k)
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'grid.npz',pairs=PAIRS,grid_r=result['grid_r'],winners=result['winners'],
        matching_indices=result['matching_indices'],matching_signs=result['matching_signs'],
        matching_r=result['matching_r'],train_indices=np.concatenate([a for a,b in result['splits']]),
        train_lengths=np.array([len(a) for a,b in result['splits']]),
        valid_indices=np.concatenate([b for a,b in result['splits']]),
        valid_lengths=np.array([len(b) for a,b in result['splits']]),ids=ids)
    choices=[];rows=[]
    for ib,(branch,reference,scale) in enumerate(BRANCHES):
        ip=int(result['winners'][ib]);pair=PAIRS[ip]
        fit_folder=save_final_fit(model,policy,pair,result['full'][ip],result['fixed'],ids,x,y,v)
        choices.append(dict(branch=branch,reference=reference,score_scaling=scale,
            pair_index=ip,tau=[float(c) for c in pair],fit_directory=str(fit_folder.relative_to(CURRENT)),
            tuning_r=result['grid_r'][ib,ip].tolist(),tuning_mean_r=float(result['grid_r'][ib,ip].mean())))
        for index,p in enumerate(PAIRS):
            r=result['grid_r'][ib,index]
            rows.append(dict(branch=branch,tau_x=float(p[0]),tau_y=float(p[1]),r_1=r[0],r_2=r[1],mean_r=r.mean()))
    pd.DataFrame(rows).to_csv(folder/'grid.csv',index=False)
    record=dict(status='complete',source_kind='full_data_cv_refit',model=model,volume_policy=policy,
        cv_seed=seed,cv_folds=k,inner_selection_seed_rule='master seed',n=len(ids),scales=list(SCALES),
        objective='mean of first two pooled validation correlations',choices=choices,
        elapsed_seconds=time.monotonic()-started,rcca_permutations=0)
    write_json(folder/'selection.json',record)
    return record


def original_oof_path(model,policy,seed,design):
    return SOURCE_ROOTS[policy]/'tasks'/model.replace('+','_')/design/f'seed_{seed}'


def oof_task(task):
    """Reuse verified original fits after validating their exact task provenance."""
    model,policy,seed,design=task
    folder=CURRENT/'oof'/policy/model.replace('+','_')/design/f'seed_{seed}'
    if (folder/'manifest.json').exists():
        completed=json.loads((folder/'manifest.json').read_text())
        assert completed['status']=='complete' and completed['source_kind']=='validated_nested_oof'
        for key,value in dict(model=model,volume_policy=policy,seed=seed,design=design,n=160).items():
            assert completed[key]==value
        return pd.read_csv(folder/'summary.csv').to_dict('records')
    source=original_oof_path(*task)
    metadata=json.loads((source/'metadata.json').read_text())
    no,ni=map(int,design.split('x'))
    for name,value in dict(model=model,seed=seed,outer_folds=no,inner_folds=ni,volume_policy=policy,n=160).items():
        assert metadata[name]==value,(name,metadata[name],value)
    ids,_,y,_=load_data();splits=run.make_splits(y,no,seed,5,2)
    for k,(_,test) in enumerate(splits,1):
        f=np.load(source/f'fold_{k:02d}.npz')
        np.testing.assert_array_equal(f['test_index'],test)
    summary=pd.read_csv(source/'summary.csv')
    assert len(summary)==4 and set(summary.branch)==set('ABCD')
    rows=[]
    for branch,_,_ in BRANCHES:
        z=np.load(source/f'{branch}_aligned.npz')
        np.testing.assert_array_equal(np.sort(z['ids']),np.sort(ids))
        assert len(np.unique(z['ids']))==160
        row=summary[summary.branch==branch].iloc[0].to_dict()
        for j in (0,1):
            np.testing.assert_allclose(pearsonr(z['x_score'][:,j],z['y_score'][:,j]).statistic,row[f'r_{j+1}'],atol=1e-12)
            assert run.compute_icc(z['brain'][:,:,j])==row[f'brain_icc_{j+1}']
            assert run.compute_icc(z['behavior'][:,:,j])==row[f'behavior_icc_{j+1}']
        row.update(volume_policy=policy,source_directory=str(source),oof_reused=True)
        rows.append(row)
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/'summary.csv',index=False)
    write_json(folder/'manifest.json',dict(status='complete',n=160,model=model,volume_policy=policy,
        seed=seed,design=design,source_directory=str(source),source_kind='validated_nested_oof',
        rcca_permutations=0))
    return rows


def load_final(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['source_kind']!='full_data_cv_selected_fit' or manifest['coefficient_filter']!='none':
        raise ValueError('Molecular maps must come from a current full-data-CV-selected fit without filtering')
    return manifest,np.load(directory/'model.npz')
