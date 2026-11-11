"""Shared candidate fits, paired alignment and observed-screen aggregation."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from cca_zoo.linear import RidgeCCA
from . import nested as run
VOLUME_POLICY = 'brain_only'
DATA = None
BRANCHES = (('A','candidate','raw'), ('B','candidate','training'),
            ('C','fixed_primary','raw'), ('D','fixed_primary','training'))
PAIRS = np.array([(a,b) for a in run.DEFAULT_TAU_GRID for b in run.DEFAULT_TAU_GRID])

def config_for(model,seed,outer_folds,inner_folds=5,scales=run.SCALES_23):
    return replace(run.make_config(model,scales),seed=seed,
                   outer_folds=outer_folds,inner_folds=inner_folds)

def paired_r(x,y):
    x=x-x.mean(axis=0);y=y-y.mean(axis=0)
    result=np.sum(x*y,axis=0)/np.sqrt(np.sum(x*x,axis=0)*np.sum(y*y,axis=0))
    if not np.all(np.isfinite(result)):
        raise ValueError('Nonfinite correlation in a supported data fit')
    return result

def fit_bundle(prepared,cx,cy):
    X,Y=prepared.x_train.astype(np.float64),prepared.y_train.astype(np.float64)
    model=RidgeCCA(n_components=10,shrinkage=[float(cx),float(cy)]).fit([X,Y])
    wx,wy=model.weights_
    tx=(X-model.means_[0])@wx;ty=(Y-model.means_[1])@wy
    sx,sy=model.transform([prepared.x_test.astype(np.float64),prepared.y_test.astype(np.float64)])
    result=dict(wx=wx,wy=wy,sx=sx,sy=sy,mx=tx.mean(axis=0),my=ty.mean(axis=0),
                dx=tx.std(axis=0),dy=ty.std(axis=0))
    if any(not np.all(np.isfinite(v)) for v in result.values()):
        raise ValueError('Nonfinite fitted projections or loadings')
    if np.any(result['dx']<=0) or np.any(result['dy']<=0):
        raise ValueError('Zero training score SD')
    return result

def mapping(reference,weights):
    c=np.corrcoef(reference.T,weights.T)[:10,10:]
    rows,index=linear_sum_assignment(-np.abs(c))
    np.testing.assert_array_equal(rows,np.arange(10))
    sign=np.sign(c[rows,index]);sign[sign==0]=1
    polarity=np.where(-reference[:,:2].min(axis=0)>reference[:,:2].max(axis=0),-1,1)
    sign[:2]*=polarity
    return index,sign,c[rows,index]

def apply_mapping(bundle,match,scaling):
    index,sign,corr=match
    x=bundle['sx'][:,index]*sign;y=bundle['sy'][:,index]*sign
    mx=bundle['mx'][index]*sign;my=bundle['my'][index]*sign
    dx=bundle['dx'][index];dy=bundle['dy'][index]
    if scaling=='training':
        x=(x-mx)/dx;y=(y-my)/dy
    return dict(x=x,y=y,wx=bundle['wx'][:,index]*sign,wy=bundle['wy'][:,index]*sign,
                mx=mx,my=my,dx=dx,dy=dy,index=index,sign=sign,match=corr)

def project(bundle,reference,scaling):
    return apply_mapping(bundle,mapping(reference,bundle['wy']),scaling)

def fold_shared(x,y,volume,outer,k,cfg):
    tr,_=outer[k-1]
    inner=run.make_splits(y[tr],cfg.inner_folds,cfg.seed+k,cfg.y_bins,cfg.y_pc)
    return fold_from_splits(x,y,volume,outer,k,cfg,inner)


def fold_from_splits(x,y,volume,outer,k,cfg,inner):
    """Evaluate the unchanged parameter-grid algorithm using explicit inner splits."""
    start=time.monotonic();tr,te=outer[k-1]
    xt,yt,vt=x[tr],y[tr],volume[tr]
    inner_data=[run.prepare_split(xt,yt,vt,a,b,VOLUME_POLICY,cfg) for a,b in inner]
    prepared=run.prepare_split(x,y,volume,tr,te,VOLUME_POLICY,cfg)
    # These full-outer-training fits also provide all candidate references.
    full=[fit_bundle(prepared,cx,cy) for cx,cy in PAIRS]
    fixed_index=int(np.flatnonzero(np.all(PAIRS==cfg.resolved_primary_params,axis=1))[0])
    fixed_ref=full[fixed_index]['wy']
    grid_r=np.empty((4,len(PAIRS),2))
    indices=np.empty((2,len(PAIRS),cfg.inner_folds,10),dtype=np.int16)
    signs=np.empty_like(indices,dtype=np.int8)
    matches=np.empty_like(indices,dtype=float)
    for ip,(cx,cy) in enumerate(PAIRS):
        pieces=[[] for _ in BRANCHES]
        refs=(full[ip]['wy'],fixed_ref)
        for ik,p in enumerate(inner_data):
            bundle=fit_bundle(p,cx,cy)
            for ir,reference in enumerate(refs):
                match=mapping(reference,bundle['wy'])
                for iz,scale in enumerate(('raw','training')):
                    result=apply_mapping(bundle,match,scale)
                    pieces[2*ir+iz].append((result['x'][:,:2],result['y'][:,:2]))
                    if iz==0:
                        indices[ir,ip,ik]=result['index']
                        signs[ir,ip,ik]=result['sign']
                        matches[ir,ip,ik]=result['match']
        for ib,parts in enumerate(pieces):
            grid_r[ib,ip]=paired_r(np.concatenate([p[0] for p in parts]),
                                 np.concatenate([p[1] for p in parts]))
    winners=np.argmax(grid_r.mean(axis=2),axis=1)  # first grid pair wins exact ties
    selected=[project(full[int(ip)],fixed_ref,BRANCHES[ib][2]) for ib,ip in enumerate(winners)]
    output=dict(test_index=te,train_index=tr,pairs=PAIRS,grid_r=grid_r,winners=winners,
                inner_indices=indices,inner_signs=signs,inner_matches=matches,
                inner_train_indices=np.concatenate([a for a,b in inner]),
                inner_train_lengths=np.array([len(a) for a,b in inner]),
                inner_valid_flat=np.concatenate([b for a,b in inner]),
                inner_valid_lengths=np.array([len(b) for a,b in inner]),
                elapsed=np.array(time.monotonic()-start))
    for key in selected[0]:output[key]=np.stack([v[key] for v in selected])
    return output

def aggregate(taskdir,model,cfg,scales=run.SCALES_23):
    ids,_,_,_=DATA
    folds=[]
    for k in range(1,cfg.outer_folds+1):
        with np.load(taskdir/f'fold_{k:02d}.npz') as z:folds.append({n:z[n].copy() for n in z.files})
    with np.load(run.CODE_DIR/'data/reference/axis_labels/cca260_mean_symptom_loadings.npz') as old:
        baseline=old[model][[run.SCALES_23.index(scale) for scale in scales]]
    demographics=pd.read_csv(run.CODE_DIR/'mmpi/Iowa_MMPI2RF_byNandi.csv').set_index('ID')['Sex']
    summaries=[];score_rows=[];parameter_rows=[];grid_rows=[]
    for ib,(branch,ref,scale) in enumerate(BRANCHES):
        xs=[f['x'][ib].copy() for f in folds];ys=[f['y'][ib].copy() for f in folds]
        wx=[f['wx'][ib].copy() for f in folds];wy=[f['wy'][ib].copy() for f in folds]
        run.harmonize_outer_component_signs(xs,ys,wx,wy,2)
        test_order=np.concatenate([f['test_index'] for f in folds])
        np.testing.assert_array_equal(np.sort(test_order),np.arange(len(ids)))
        X,Y=np.concatenate(xs),np.concatenate(ys)
        ybar=np.mean(wy,axis=0)[:,:2]
        corr=np.corrcoef(baseline.T,ybar.T)[:2,2:]
        mean_semantics=np.abs(np.diag(corr))
        alternate=(abs(corr[0,1])+abs(corr[1,0])>abs(corr[0,0])+abs(corr[1,1]))
        item=dict(icc_method='foldwise_unit_l2_ICC1',model=model,seed=cfg.seed,outer_folds=cfg.outer_folds,inner_folds=cfg.inner_folds,volume_policy=VOLUME_POLICY,
                  design=f'{cfg.outer_folds}x{cfg.inner_folds}',branch=branch,reference=ref,score_scaling=scale,n=len(ids),
                  semantic_swap_flag=bool(alternate),min_behavior_similarity=float(mean_semantics.min()))
        for j in (0,1):
            r=float(pearsonr(X[:,j],Y[:,j]).statistic)
            loo=[pearsonr(np.delete(X[:,j],i),np.delete(Y[:,j],i)).statistic for i in range(len(X))]
            item[f'r_{j+1}']=r;item[f'd_{j+1}']=float(2*r/np.sqrt(1-r*r))
            item[f'brain_icc_{j+1}']=run.compute_icc(np.stack(wx)[:,:,j])
            item[f'behavior_icc_{j+1}']=run.compute_icc(np.stack(wy)[:,:,j])
            item[f'old_behavior_similarity_{j+1}']=float(mean_semantics[j])
            item[f'loo_min_{j+1}']=float(min(loo));item[f'loo_max_{j+1}']=float(max(loo))
            for sex in ('F','M'):
                keep=demographics.loc[ids[test_order]].to_numpy()==sex
                item[f'sex_{sex}_r_{j+1}']=float(pearsonr(X[keep,j],Y[keep,j]).statistic)
        item['min_r']=min(item['r_1'],item['r_2']);item['mean_r']=(item['r_1']+item['r_2'])/2
        summaries.append(item)
        np.savez_compressed(taskdir/f'{branch}_aligned.npz',brain=np.stack(wx),behavior=np.stack(wy),
                            ids=ids[test_order],x_score=X,y_score=Y)
        for k,f in enumerate(folds,1):
            for ip,pair in enumerate(PAIRS):
                rr=f['grid_r'][ib,ip]
                grid_rows.append(dict(branch=branch,outer_fold=k,tau_x=pair[0],tau_y=pair[1],
                                      r_1=rr[0],r_2=rr[1],mean_r=rr.mean()))
            winner=int(f['winners'][ib]);pair=PAIRS[winner]
            row=dict(branch=branch,outer_fold=k,tau_x=pair[0],tau_y=pair[1],
                     inner_r_1=f['grid_r'][ib,winner,0],inner_r_2=f['grid_r'][ib,winner,1])
            for j in (0,1):row[f'test_r_{j+1}']=pearsonr(xs[k-1][:,j],ys[k-1][:,j]).statistic
            parameter_rows.append(row)
            for i,sid in enumerate(ids[f['test_index']]):
                score_rows.append(dict(branch=branch,outer_fold=k,study_id=int(sid),
                    x_score_1=xs[k-1][i,0],y_score_1=ys[k-1][i,0],
                    x_score_2=xs[k-1][i,1],y_score_2=ys[k-1][i,1]))
    pd.DataFrame(parameter_rows).to_csv(taskdir/'selected_parameters.csv',index=False)
    pd.DataFrame(grid_rows).to_csv(taskdir/'inner_grid.csv',index=False)
    pd.DataFrame(score_rows).to_csv(taskdir/'outer_scores.csv',index=False)
    pd.DataFrame(summaries).to_csv(taskdir/'summary.csv',index=False)
    return summaries
