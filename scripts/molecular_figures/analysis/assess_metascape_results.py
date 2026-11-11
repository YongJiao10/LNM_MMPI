"""Inspect the fixed 160-person baseline's Metascape session tbvppka78.

Read the already authorized, extracted package and preserve full per-list
tables plus network/evidence summaries. No enrichment reruns or plotting.
"""
from pathlib import Path
import json
import pandas as pd
import xml.etree.ElementTree as ET

r=Path('/Users/jy/Documents/LNM_CCA_project/LNM_MMPI_code/results/cca401_160x23_seed1_retuned_mean_tminus1_20261031/metascape/outputs')
p=r/'unpacked'
a=pd.read_csv(p/'Enrichment_GO/GO_AllLists.csv')
f=pd.read_csv(p/'Enrichment_GO/_FINAL_GO.csv')
e=pd.read_csv(p/'Evidence/Evidence.csv')
sel=set(e.loc[e.TotalEvidenceCounts>=3,'Gene'].astype(str))
ns={'g':'http://www.cs.rpi.edu/XGMML'}
g=ET.parse(p/'Enrichment_GO/GONetwork.xgmml').getroot()
nodes=[{**n.attrib,**{x.get('name'):x.get('value') for x in n.findall('g:att',ns)}} for n in g.findall('g:node',ns)]
edges=[{**n.attrib,**{x.get('name'):x.get('value') for x in n.findall('g:att',ns)}} for n in g.findall('g:edge',ns)]
scores=[float(x['SCORE']) for x in edges]
node_terms={x['label'] for x in nodes}
summary={'session_id':'tbvppka78','individual_rows':len(a),'individual_by_list':{},'combined_rows':len(f),'combined_cluster_count':int(f.GROUP_ID.nunique()),'combined_ungrouped_rows':int(f.GROUP_ID.isna().sum()),'evidence_counts':{str(k):int(v) for k,v in e.TotalEvidenceCounts.value_counts().sort_index().items()},'selected_evidence_ge3_count':len(sel),'combined_winner_hitlist_sizes':{str(k):int(v) for k,v in f['#GeneInHitList'].value_counts().items()},'pooled_rows_outside_selected_genes':sum(not set(s.split('|')).issubset(sel) for s in f.loc[f['#GeneInHitList']==1889,'GeneID']),'network':{'nodes':len(nodes),'edges':len(edges),'clusters':sorted({int(x['GROUP_ID']) for x in nodes}),'similarity_min':min(scores),'similarity_max':max(scores),'unknown_terms':sorted(node_terms-set(f.GO)),'unmatched_node_cluster_count':sum(int(x['GROUP_ID'])!=int(f.loc[f.GO.eq(x['label']),'GROUP_ID'].iloc[0]) for x in nodes)}}
fields=['GO','Description','Category','LogP','Log(q-value)','Enrichment','#GeneInGOAndHitList']
for name,x in a.groupby('GeneList'):
    summary['individual_by_list'][name]={'rows':len(x),'hitlist_annotated_size':int(x['#GeneInHitList'].iloc[0]),'q_below_05_rows':int((x['Log(q-value)'] < -1.3010299956639813).sum()),'top10':x.nsmallest(10,'LogP')[fields].to_dict('records')}
    x.to_csv(r/(name+'_enrichment.csv'),index=False)

term='GO:0120035'
all4=a.loc[a.GO==term]
full_hits=set('|'.join(all4.GeneID).split('|'))
actual=set(f.loc[f.GO.eq(term),'GeneID'].iloc[0].split('|'))
summary['pooled_term_check']={'GO':term,'raw_lists':all4.GeneList.tolist(),'full_four_list_hit_union':len(full_hits),'evidence_ge3_intersection':len(full_hits&sel),'actual_pooled_hits':len(actual),'exact_match':actual==(full_hits&sel)}
heat=pd.read_csv(p/'Enrichment_heatmap/HeatmapSelectedGO.csv')
summary['top20_heatmap']=heat.to_dict('records')
rep=f.loc[f.FirstInGroupByLogP.eq(1)&f.GROUP_ID.le(20)]
summary['top20_representatives']=rep[['GROUP_ID',*fields,'#GeneInHitList']].to_dict('records')
rep[['GROUP_ID',*fields,'#GeneInHitList']].to_csv(r/'top20_cluster_representatives.csv',index=False)
pd.DataFrame(nodes).to_csv(r/'network_nodes.csv',index=False)
pd.DataFrame(edges).to_csv(r/'network_edges.csv',index=False)
(r/'enrichment_review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['individual_by_list','top20_heatmap','top20_representatives']},indent=2))
print('each_list',{k:{a:b for a,b in v.items() if a!='top10'} for k,v in summary['individual_by_list'].items()})
print('top20_per_list_log10p')
print(heat.to_string(index=False))
