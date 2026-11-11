"""Molecular decoding of current CV-selected, unthresholded Combined fits.

Exact PLS1 algebra shares fixed predictors, rotations and bootstrap draws.
PET uses BH-FDR separately within each dimension; this module never chooses rCCA parameters.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm,pearsonr,hypergeom

CODE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(CODE))
import cca401_pipeline as pipeline
from cca401 import molecular_math as original

CURRENT=pipeline.CURRENT
AHBA=CODE/'data/reference/ahba'
SEED=42
N_SPIN=5000
N_BOOT=1000
DATA=None


def initialize():
    global DATA
    expression=pd.read_csv(AHBA/'ahba_expression_schaefer100.csv').set_index('cortical_id')
    meta=pd.read_csv(AHBA/'roi_metadata_116.csv').sort_values('roi_id')
    cortex=meta.loc[meta.structure=='cortex','roi_id'].to_numpy(int)-1
    X=expression.to_numpy(float)
    assert X.shape==(99,15633) and len(cortex)==100
    Xz=original.standardize_predictors(X);Xz-=Xz.mean(axis=0)
    receptors=np.load(CURRENT/'assets/receptor_maps.npz')
    names=list(receptors.files)
    indices=np.load(CURRENT/'assets/spin_indices.npy')
    libraries={}
    for name in original.ENRICHR_LIBRARIES:
        libraries[name]={}
        for line in (CURRENT/'assets'/f'{name}.gmt').read_text().splitlines():
            a=line.split('\t');libraries[name][a[0]]=set(a[2:])
    DATA=dict(X=X,Xz=Xz,gram=Xz@Xz.T,genes=expression.columns,
        keep=expression.index.to_numpy(int)-1,cortex=cortex,spin=indices,
        receptors=np.column_stack([receptors[n] for n in names]),receptor_names=names,libraries=libraries)


def prepare_spin_indices():
    path=CURRENT/'assets/spin_indices.npy'
    if path.exists():
        assert np.load(path).shape==(N_SPIN,100)
        return
    spin=original.build_spin_model(N_SPIN,SEED)
    lh,rh=spin.randomize(np.arange(50),np.arange(50,100))
    indices=np.concatenate([lh,rh],axis=1).astype(np.int16)
    assert indices.shape==(N_SPIN,100)
    np.save(path,indices)


def correlation_matrix(a,b):
    a=a-a.mean(axis=0);b=b-b.mean(axis=0)
    return a.T@b/np.outer(np.linalg.norm(a,axis=0),np.linalg.norm(b,axis=0))


def spatial(y_full):
    d=DATA;y=y_full[d['keep']]
    spun=y_full[d['spin']][:,d['keep']].T
    yc=spun-spun.mean(axis=0)
    score=d['gram']@yc;score-=score.mean(axis=0)
    null=np.abs(np.sum(score*yc,axis=0)/(np.linalg.norm(score,axis=0)*np.linalg.norm(yc,axis=0)))
    observed=abs(float(original.fit_single_component_pls_standardized(d['Xz'],y)['observed_r']))
    for index in np.flatnonzero(abs(null-observed)<=2e-12):
        null[index]=abs(float(original.fit_single_component_pls_standardized(d['Xz'],spun[:,index])['observed_r']))
    p=float((sum(null>=observed)+1)/(N_SPIN+1))
    return y,spun,observed,p,null


def normalize_columns(x):
    sd=x.std(axis=0,ddof=1)
    return np.divide(x-x.mean(axis=0),sd,out=np.zeros_like(x),where=sd>0)


def gene_weights(X,Y):
    Xz=normalize_columns(X);Xz-=Xz.mean(axis=0)
    Yz=normalize_columns(Y);Yz-=Yz.mean(axis=0)
    w=Xz.T@Yz
    return w/np.linalg.norm(w,axis=0)


def bootstrap_tables(Y):
    """All four response maps share predictor normalization and resample rows."""
    X=DATA['X'];base=gene_weights(X,Y)
    rng=np.random.default_rng(SEED)
    mean=np.zeros_like(base);M2=np.zeros_like(base)
    for i in range(N_BOOT):
        index=rng.choice(len(X),size=len(X),replace=True)
        w=gene_weights(X[index],Y[index])
        a=w-w.mean(axis=0);b=base-base.mean(axis=0)
        sign=np.where(np.sum(a*b,axis=0)<0,-1.,1.);w*=sign
        delta=w-mean;mean+=delta/(i+1);M2+=delta*(w-mean)
    sd=np.sqrt(M2/(N_BOOT-1))
    ratio=np.divide(base,sd,out=np.zeros_like(base),where=sd>0)
    tables=[]
    for j in range(Y.shape[1]):
        p=2*norm.sf(abs(ratio[:,j]));q=original.benjamini_hochberg(p)
        tables.append(pd.DataFrame(dict(gene=DATA['genes'],pls_weight=base[:,j],boot_mean=mean[:,j],
            boot_sd=sd[:,j],bootstrap_ratio=ratio[:,j],p_boot=p,q_boot_fdr=q)))
    return tables


def enrich(genes):
    rows=[];query={g.upper() for g in genes}
    for library,terms in DATA['libraries'].items():
        part=[]
        for term,members in terms.items():
            shared=members&query;x,m,k=len(shared),len(members),len(query)
            if not x:continue
            p=float(hypergeom.sf(x-1,20000,m,k))
            part.append(dict(library=library,term=term,overlap=x,term_size=m,p=p,genes=';'.join(sorted(shared))))
        q=original.benjamini_hochberg(np.array([r['p'] for r in part]))
        for row,value in zip(part,q):row['q']=float(value)
        rows.extend(part)
    return pd.DataFrame(rows,columns=['library','term','overlap','term_size','p','genes','q'])


def molecular_task(fit_relative):
    if DATA is None:initialize()
    started=time.monotonic()
    fit_directory=CURRENT/fit_relative
    manifest,model=pipeline.load_final(fit_directory)
    assert manifest['model']=='nc+fc'
    out=CURRENT/'molecular'/Path(fit_relative).relative_to('fits')
    if (out/'completion.json').exists():return json.loads((out/'completion.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    brain=model['brain'];d=DATA
    map_ids=[];responses=[];plsrows=[];petrows=[]
    for j in (0,1):
        for modality,start in [('sdc',0),('fdc',116)]:
            map_id=f'combined_cd{j+1}_{modality}';map_ids.append(map_id)
            y_full=brain[start:start+116,j][d['cortex']]
            y,spun,r,p,null=spatial(y_full);responses.append(y)
            folder=out/map_id;folder.mkdir(exist_ok=True)
            np.save(folder/'spin_null_abs_r.npy',null)
            plsrows.append(dict(map_id=map_id,r=r,p_spin=p))
            if modality=='fdc':
                receptors=d['receptors'][d['keep']]
                observed=correlation_matrix(receptors,y[:,None]).ravel()
                null_pet=correlation_matrix(receptors,spun)
                for a,name in enumerate(d['receptor_names']):
                    for index in np.flatnonzero(abs(abs(null_pet[a])-abs(observed[a]))<=2e-12):
                        null_pet[a,index]=pearsonr(receptors[:,a],spun[:,index]).statistic
                    petrows.append(dict(map_id=map_id,receptor=name,r=float(observed[a]),
                        p_spin=float((sum(abs(null_pet[a])>=abs(observed[a]))+1)/(N_SPIN+1))))
                np.save(folder/'pet_spin_null_r.npy',null_pet)
    spatial_seconds=time.monotonic()-started
    pet=pd.DataFrame(petrows);assert len(pet)==20
    pet['q_fdr_20']=original.benjamini_hochberg(pet.p_spin.to_numpy())
    pet['q_fdr_within_cd_10']=pet.groupby('map_id').p_spin.transform(
        lambda values:original.benjamini_hochberg(values.to_numpy()))
    pet['q_fdr']=pet.q_fdr_within_cd_10
    pd.DataFrame(plsrows).to_csv(out/'pls.csv',index=False);pet.to_csv(out/'pet.csv',index=False)
    tables=bootstrap_tables(np.column_stack(responses))
    summaries=[];enrichment=[]
    for map_id,table in zip(map_ids,tables):
        folder=out/map_id
        table.to_csv(folder/'gene_weights.csv',index=False)
        stable=table.q_boot_fdr<.05
        summaries.append(dict(map_id=map_id,n_stable=int(sum(stable)),
            n_positive=int(sum(stable&(table.bootstrap_ratio>0))),n_negative=int(sum(stable&(table.bootstrap_ratio<0)))))
        for label,sign in [('positive',1),('negative',-1)]:
            genes=table.loc[stable&(table.bootstrap_ratio*sign>0),'gene'].tolist()
            (folder/f'{label}_genes.txt').write_text('\n'.join(genes)+'\n')
            if len(genes)<5:
                enrichment.append(dict(map_id=map_id,sign=label,n_genes=len(genes),status='original_minimum_5_genes_not_met'))
                continue
            terms=enrich(genes);terms.to_csv(folder/f'{label}_enrichment.csv',index=False)
            enrichment.append(dict(map_id=map_id,sign=label,n_genes=len(genes),status='complete',significant_terms=int(sum(terms.q<.05))))
    pd.DataFrame(summaries).to_csv(out/'genes.csv',index=False)
    pd.DataFrame(enrichment).to_csv(out/'enrichment_summary.csv',index=False)
    a=tables[1].set_index('gene');b=tables[3].set_index('gene').loc[a.index]
    sa=set(a.index[a.q_boot_fdr<.05]);sb=set(b.index[b.q_boot_fdr<.05])
    overlap=dict(gene_weight_r=float(pearsonr(a.pls_weight,b.pls_weight).statistic),
        cd1_stable=len(sa),cd2_stable=len(sb),shared=len(sa&sb),union=len(sa|sb),
        jaccard=len(sa&sb)/len(sa|sb) if sa|sb else None)
    pipeline.write_json(out/'fdc_overlap.json',overlap)
    record=dict(status='complete',fit_directory=fit_relative,output_directory=str(out.relative_to(CURRENT)),
        source_kind='current_full_data_cv_fit',n_spin=N_SPIN,n_boot=N_BOOT,seed=SEED,
        n_maps=4,pet_tests=20,coefficient_filter='none',rcca_permutations=0,
        spatial_seconds=spatial_seconds,elapsed_seconds=time.monotonic()-started,
        gene_lists_uploaded=False,metascape_recomputed=False)
    pipeline.write_json(out/'completion.json',record)
    return record
