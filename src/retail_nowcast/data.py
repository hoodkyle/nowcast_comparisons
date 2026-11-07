from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Sequence, Mapping
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .msrs import load_msrs as _load_msrs
from .targets import category_delta_targets
from .panels import build_feature_store
from .timing import build_vintage_feature_store


CATEGORIES=("TOTAL","441","442","443","444","445","446","447","448","451","452","453")

STATES=tuple("AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY".split())

PRIVATE=("fiserv","gamma")

def _path(root,relative):
 p=Path(root)/relative
 if not p.is_file(): raise FileNotFoundError(p)
 return p

def load_msrs(root):
 p=_path(root,'data/raw/msrs/state_retail_yy.csv'); frame=_load_msrs(p)
 if set(frame.loc[~frame.state.eq('USA'),'state'])!=set(STATES): raise ValueError('MSRS must contain all 50 states plus DC')
 if set(frame.category)!=set(CATEGORIES): raise ValueError('MSRS category universe differs from frozen 12-category contract')
 return frame,p

def load_factor_panels(root,vintage='v1'):
 if vintage not in {'v1','v2','v3'}: raise ValueError('unknown information vintage')
 sp=_path(root,f'data/processed/information_vintages/state_factor_panel_{vintage}.csv.gz'); ip=_path(root,f'data/processed/information_vintages/industry_factor_panel_{vintage}.csv.gz')
 state=pd.read_csv(sp); industry=pd.read_csv(ip)
 for d in (state,industry):
  d['date']=pd.to_datetime(d['date']).dt.to_period('M').dt.to_timestamp(); d['reference_month']=pd.to_datetime(d['reference_month']).dt.to_period('M').dt.to_timestamp()
 return state,industry,sp,ip

def target_panel(msrs,categories=CATEGORIES): return category_delta_targets(msrs,categories)

def _filter(panel,source_configuration,stage):
 source_configuration={'F':'F_ONLY','G':'G_ONLY','O':'NO_PRIVATE'}.get(source_configuration,source_configuration)
 if source_configuration not in {'F_ONLY','G_ONLY','NO_PRIVATE'}: raise ValueError('unknown source configuration')
 allowed={'F_ONLY':{'fiserv'},'G_ONLY':{'gamma'},'NO_PRIVATE':set()}[source_configuration]
 source=panel.source.astype(str); return panel.loc[(~source.isin(PRIVATE))|source.isin(allowed)].copy()

def build_store(msrs,state,industry,source_configuration,stage,*,origin=None):
 source_configuration={'F':'F_ONLY','G':'G_ONLY','O':'NO_PRIVATE'}.get(source_configuration,source_configuration)
 state=_filter(state,source_configuration,stage); industry=_filter(industry,source_configuration,stage)
 for name, panel, keys in [('state',state,['source','state','concept','date']),('industry',industry,['source','concept','date'])]:
  if 'timing_variant' in panel: keys.append('timing_variant')
  if panel.duplicated(keys).any(): raise ValueError(f'duplicate {name} panel key before pivot')
 if origin is not None:
  cutoff=pd.Timestamp(origin).to_period('M').to_timestamp(); state=state.loc[state.date.le(cutoff)].copy(); industry=industry.loc[industry.date.le(cutoff)].copy()
 raw_msrs=msrs.loc[pd.to_datetime(msrs.date).le(pd.Timestamp(origin))].copy() if origin is not None else msrs
 targets=target_panel(raw_msrs); stores=[]
 dates=pd.date_range(targets.date.min(),targets.date.max(),freq='MS'); grid=pd.MultiIndex.from_product([STATES,dates],names=['state','date']).to_frame(index=False); skeleton=grid.merge(targets.loc[targets.category.eq('TOTAL'),['state','date','value','lag_yoy','z','lag_delta']],on=['state','date'],how='left')
 reference=build_paper_feature_store(skeleton[['state','date','value','lag_yoy','z','lag_delta']],state,industry,source_configuration,stage)
 feature_cols=[c for c in reference.columns if c not in {'z','lag_yoy','lag_delta'}]
 feature_frame=reference[feature_cols]
 for category in CATEGORIES:
  target=targets.loc[targets.category.eq(category),['state','date','value','lag_yoy','z','lag_delta']].copy()
  target=grid.merge(target,on=['state','date'],how='left',validate='one_to_one')
  built=target.merge(feature_frame,on=['state','date'],how='left',validate='one_to_one')
  built.insert(1,'category',category); built['target']=built['z']; built['origin']=pd.Timestamp(origin) if origin is not None else pd.NaT; built['source_configuration']=source_configuration; built['information_stage']=stage; stores.append(built)
 result=pd.concat(stores,ignore_index=True).sort_values(['date','state','category']).reset_index(drop=True)
 if result.duplicated(['state','category','date']).any(): raise ValueError('duplicate canonical category/state/date key')
 return result


def build_paper_feature_store(target, state, industry, source, stage):
    allowed={'F_ONLY':{'fiserv'},'G_ONLY':{'gamma'},'NO_PRIVATE':set()}[source]
    sources={'msrs','marts','mrts'} | allowed
    state=state.loc[state.source.isin(sources)].copy()
    industry=industry.loc[industry.source.isin(sources)].copy()
    for panel in (state,industry):
        if set(panel.loc[panel.source.isin({'fiserv','gamma'}),'source']) != allowed:
            raise ValueError('Transaction source unavailable in the supplied panel')
    if stage in {'early_information','mrts_lag1'}:
        return build_feature_store(target,state,industry,mrts_lag=1 if stage=='mrts_lag1' else 2)
    vintage={'marts_current':'v2_advance','mrts_current':'v3_full_national'}[stage]
    return build_vintage_feature_store(target,state,industry,vintage)
