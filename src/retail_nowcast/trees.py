from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Sequence, Mapping
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd


def _native_columns(train,features):
 keep=[]
 for c in features:
  x=pd.to_numeric(train[c],errors='coerce').to_numpy(float); finite=x[np.isfinite(x)]
  if finite.size and np.nanstd(finite)>0: keep.append(c)
 return tuple(keep)

def _matrix(frame,cols): return frame.loc[:,list(cols)].apply(pd.to_numeric,errors='coerce').to_numpy(float)

def _tree(train,test,features,spec,seed):
 train=train.loc[np.isfinite(pd.to_numeric(train.target,errors='coerce'))].copy(); test=test.copy(); features=list(features)
 states=sorted(train.state.astype(str).unique())
 for state in states[1:]:
  col=f'state_effect__{state}'; train[col]=train.state.astype(str).eq(state).astype('int8'); test[col]=test.state.astype(str).eq(state).astype('int8'); features.append(col)
 cols=_native_columns(train,features); x=_matrix(train,cols); q=_matrix(test,cols); y=pd.to_numeric(train.target,errors='coerce').to_numpy(float); ok=np.isfinite(y)
 if not ok.any(): return np.full(len(test),np.nan),{'rows':0,'features':list(cols)}
 fam=spec['family']; params=dict(spec['hyperparameters'])
 if fam=='catboost_gpu':
  from catboost import CatBoostRegressor
  model=CatBoostRegressor(random_seed=seed,**params)
 elif fam=='xgboost_gpu':
  from xgboost import XGBRegressor
  model=XGBRegressor(random_state=seed,**params)
 else: raise ValueError(f'unsupported registry family {fam}')
 model.fit(x[ok],y[ok])
 if fam == 'xgboost_gpu':
  config = json.loads(model.get_booster().save_config())
  device = str(config.get('learner', {}).get('generic_param', {}).get('device', ''))
  if not device.startswith('cuda'): raise RuntimeError(f'XGBoost did not train on CUDA: {device!r}')
 return np.asarray(model.predict(q),float),{'rows':int(ok.sum()),'features':list(cols),'model':model,'native_missing_variant':True,'device':params.get('devices',params.get('device'))}
