"""Estimate only the specifications and information sets reported in the paper."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import numpy as np
import pandas as pd
from .data import load_msrs,load_factor_panels,build_store,CATEGORIES,STATES
from .features import build_origin_dataset
from .models import fit_predict
from .panels import scope_columns
from .trees import _tree
from .factors import _ridge_panel_predict

ROOT=Path(__file__).resolve().parents[2]
PUBLIC={'RW0':'persistence','local_lag_ridge':'local_lag_ridge','OLS':'ols','ridge':'ridge','lasso':'lasso','elastic_net':'elastic_net','RF':'random_forest','HGB':'histogram_gradient_boosting','CatBoost':'two_month_catboost','XGBoost':'two_month_xgboost','PCA2_ridge':'two_month_pca_ridge','tiny_MLP':'mlp','tiny_temporal_transformer':'temporal_transformer','scalar_AR1_state_space':'scalar_ar1_state_space'}
STAGES=('early_information','marts_current','mrts_current')
REPLACEMENT_CATEGORIES=tuple(c for c in CATEGORIES if c!='443')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def write(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(content);temp.replace(path)


def json_write(path,obj):write(path,json.dumps(obj,sort_keys=True,allow_nan=False)+'\n')


def errors(frame,mask):
    e=frame.loc[mask,'actual'].to_numpy(float)-frame.loc[mask,'prediction'].to_numpy(float)
    if not len(e) or not np.isfinite(e).all():raise ValueError('Missing forecast on the required evaluation sample')
    return dict(n=len(e),sse=float(e@e),sae=float(np.abs(e).sum()))


def summarize(frames,key_fields):
    rows=[]
    for key,parts in frames.items():
        f=pd.concat(parts,ignore_index=True)
        for cat in sorted(f.category.unique()):
            sub=f[f.category.eq(cat)];rows.append({**dict(zip(key_fields,key)),'category':cat,**errors(sub,sub.eligible)})
    return pd.DataFrame(rows)


class Estimation:
    def __init__(self,data_root,output,exercise):
        self.data_root=Path(data_root);self.output=Path(output);self.exercise=exercise
        inputs=json.loads((ROOT/'config/inputs.json').read_text())['files']
        required={p:h for p,h in inputs.items() if exercise in {'all','timing'} or ('_v2.' not in p and '_v3.' not in p)}
        for name,digest in required.items():
            path=self.data_root/name
            if not path.is_file():raise FileNotFoundError(f'Required source file is unavailable: {name}')
            if sha(path)!=digest:raise ValueError(f'Source file differs from the study snapshot: {name}')
        self.spec=json.loads((ROOT/'config/models.json').read_text())
        self.members={s['name']:s for s in self.spec['ensemble_members']}
        if len(self.members)!=54:raise ValueError('The tree median requires 54 distinct specifications')
        code={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/retail_nowcast').glob('*.py'))}
        self.identity=hashlib.sha256(json.dumps({'inputs':required,'code':code,'spec':self.spec},sort_keys=True).encode()).hexdigest()
        self.msrs,_=load_msrs(self.data_root);self.panels={}
        for v in (('v1','v2','v3') if exercise in {'all','timing'} else ('v1',)):
            self.panels[v]=load_factor_panels(self.data_root,v)[:2]
        self.results={};self.cached=0;self.fitted=0

    def store(self,month,source,stage='early_information',exclude_electronics=False):
        v={'early_information':'v1','marts_current':'v2','mrts_current':'v3'}[stage]
        state,industry=self.panels[v]
        if exclude_electronics:
            state=state.loc[~state.concept.astype(str).str.match(r'^443\d*(?:_|$)')]
            industry=industry.loc[~industry.concept.astype(str).str.match(r'^443\d*(?:_|$)')]
        return build_store(self.msrs,state,industry,source,stage,origin=pd.Timestamp(month))

    def forecast(self,month,category,source,stage,model,seed,test,fit,exclude=False):
        unit=dict(month=month,category=category,source=source,stage=stage,model=model,seed=int(seed),exclude_electronics=exclude)
        identity=hashlib.sha256(json.dumps({'run':self.identity,'unit':unit},sort_keys=True).encode()).hexdigest()
        directory=self.output/'fits'/identity
        seal=directory/'completion.json'
        if seal.exists():
            doc=json.loads(seal.read_text())
            if doc['identity']!=identity or any(sha(directory/name)!=digest for name,digest in doc['files'].items()):raise ValueError('Saved forecast is incomplete or changed')
            self.cached+=1
            f=pd.read_csv(directory/'predictions.csv',dtype={'category':str},parse_dates=['date'],float_precision='round_trip')
        else:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):prediction=fit()
            f=test[['state','category','date']].copy().reset_index(drop=True)
            f['actual']=test.target.to_numpy(float);f['prediction']=np.asarray(prediction,float)
            if len(f)!=51 or f.duplicated(['state','category','date']).any() or set(f.state)!=set(STATES):raise ValueError('Invalid forecast key universe')
            write(directory/'predictions.csv',f.to_csv(index=False))
            json_write(seal,{'identity':identity,'unit':unit,'files':{'predictions.csv':sha(directory/'predictions.csv')}})
            self.fitted+=1
        np.testing.assert_allclose(f.actual,test.target,rtol=0,atol=0,equal_nan=True)
        if list(f.state)!=list(test.state):raise ValueError('Saved forecast states differ')
        return f

    def split(self,store,month,category):
        train=store.loc[store.category.eq(category)&store.date.lt(pd.Timestamp(month))].copy()
        test=store.loc[store.category.eq(category)&store.date.eq(pd.Timestamp(month))].sort_values('state').copy()
        if train.empty or train.date.max()>=pd.Timestamp(month):raise ValueError('Invalid estimation chronology')
        return train,test

    def tree(self,train,test,spec):
        query=test.copy();query['target']=np.nan;query['z']=np.nan
        p,meta=_tree(train,query,scope_columns(train,spec['feature_scope']),spec,spec['seed'])
        return p

    def factor(self,train,test):
        candidates=[c for c in train if c.startswith('pca_industry_')]+[c for c in train if c.startswith('pca_state_')]
        features=[c for c in candidates if test[c].notna().all() and train.loc[train[c].notna(),'date'].nunique()>=24]
        usable=train.dropna(subset=['z',*features])
        if not features or usable.date.nunique()<24:return np.full(len(test),np.nan)
        return _ridge_panel_predict(usable,test,features,ridge=1.)

    def own_lag(self,train,test):
        usable=train.dropna(subset=['target','lag_delta']).copy();usable['z']=usable.target
        return _ridge_panel_predict(usable,test,['lag_delta'],ridge=1.)

    def main(self,calibration=False):
        outputs=defaultdict(list);seeds=defaultdict(list);own=defaultdict(list);coverage=[]
        months=['2023-03'] if calibration else [str(p) for p in pd.period_range('2023-04','2026-05',freq='M')]
        for month in months:
            source='F_ONLY' if month<'2024-04' else 'G_ONLY';store=self.store(month,source)
            for category in (('TOTAL',) if calibration else CATEGORIES):
                train,test=self.split(store,month,category);ds=build_origin_dataset(store,month,category)
                if list(ds.score['keys'].state)!=list(test.state):raise ValueError('Common-sample forecast keys differ')
                mask=np.asarray(ds.score['eligible'],bool)&np.isfinite(test.target.to_numpy(float))
                if month=='2024-03':mask[:]=False
                coverage.append(dict(origin=month,category=category,expected=51,common_count=int(mask.sum()),evaluable=bool(mask.any())))
                for spec in self.spec['representatives']:
                    members=[]
                    for seed in spec['seeds']:
                        f=self.forecast(month,category,source,STAGES[0],PUBLIC[spec['name']],seed,test,lambda:fit_predict(ds,spec,seed=seed)[0])
                        if not np.isfinite(f.loc[mask,'prediction']).all():raise ValueError('Unavailable main forecast')
                        f['eligible']=mask;members.append(f.prediction.to_numpy())
                        if len(spec['seeds'])>1:seeds[(PUBLIC[spec['name']],seed)].append(f.copy())
                    f=f.copy();f['prediction']=np.mean(members,axis=0);outputs[(PUBLIC[spec['name']],)].append(f)
                trees=[]
                for name,spec in self.members.items():
                    f=self.forecast(month,category,source,STAGES[0],name,spec['seed'],test,lambda:self.tree(train,test,spec))
                    if not np.isfinite(f.prediction).all():raise ValueError('Unavailable tree-median member')
                    trees.append(f.prediction.to_numpy())
                    if name==self.spec['single_month']:
                        selected=f.copy();selected['eligible']=mask;outputs[('single_month_catboost',)].append(selected)
                f=f.copy();f['prediction']=np.median(trees,axis=0);f['eligible']=mask;outputs[('boosted_tree_median_54',)].append(f)
                for name,fun,collection in [('state_industry_factor_ridge',self.factor,outputs),('local_lag_ridge_own_history',self.own_lag,own)]:
                    f=self.forecast(month,category,source,STAGES[0],name,0,test,lambda:fun(train,test))
                    if not np.isfinite(f.loc[mask,'prediction']).all():raise ValueError('Unavailable factor or own-lag forecast')
                    f['eligible']=mask;collection[(name,)].append(f)
            print(f'Main comparison: completed {month}',flush=True)
        if not calibration:
            self.results.update({'main.csv':summarize(outputs,['model']),'neural_seeds.csv':summarize(seeds,['model','seed']),'local_ridge.csv':summarize(own,['model']),'coverage.csv':pd.DataFrame(coverage)})

    def timing(self,calibration=False):
        outputs=defaultdict(list)
        months=['2023-03'] if calibration else [str(p) for p in pd.period_range('2023-04','2026-05',freq='M')]
        spec=self.members[self.spec['single_month']]
        for month in months:
            source='F_ONLY' if month<'2024-04' else 'G_ONLY';early=self.store(month,source)
            categories=('TOTAL',) if calibration else CATEGORIES
            masks={c:np.asarray(build_origin_dataset(early,month,c).score['eligible'],bool) for c in categories}
            for stage in STAGES:
                store=early if stage==STAGES[0] else self.store(month,source,stage)
                for c in categories:
                    train,test=self.split(store,month,c);mask=masks[c]&np.isfinite(test.target.to_numpy(float))
                    if month=='2024-03':mask[:]=False
                    f=self.forecast(month,c,source,stage,spec['name'],spec['seed'],test,lambda:self.tree(train,test,spec))
                    if not np.isfinite(f.loc[mask,'prediction']).all():raise ValueError('Unavailable forecast in the fixed timing sample')
                    f['eligible']=mask
                    for period in ('full','pre_transition' if month<'2024-04' else 'post_transition'):
                        outputs[(stage,period)].append(f.copy())
            print(f'Information timing: completed {month}',flush=True)
        if not calibration:self.results['timing.csv']=summarize(outputs,['stage','period'])

    def replacement(self,calibration=False):
        outputs=defaultdict(list);coverage=[];monthly=[]
        months=['2023-03'] if calibration else [str(p) for p in pd.period_range('2023-04','2024-03',freq='M')]
        spec=self.members[self.spec['single_month']]
        for month in months:
            common=None
            for source in ('F_ONLY','G_ONLY','NO_PRIVATE'):
                store=self.store(month,source,exclude_electronics=True)
                cols=[c for c in scope_columns(store,'all_industry') if '__fiserv__' not in c and '__gamma__' not in c]
                official=store[['state','category','date','target',*cols]]
                if common is None:common=official
                else:pd.testing.assert_frame_equal(common,official,check_exact=True)
                for c in (('TOTAL',) if calibration else REPLACEMENT_CATEGORIES):
                    train,test=self.split(store,month,c);mask=np.isfinite(test.target.to_numpy(float))
                    if any(re.search(r'(?:^|__)443\d*(?:_|$)',col) for col in scope_columns(store,'all_industry')):raise ValueError('Electronics predictor remains')
                    f=self.forecast(month,c,source,STAGES[0],spec['name'],spec['seed'],test,lambda:self.tree(train,test,spec),exclude=True)
                    if not np.isfinite(f.prediction).all():raise ValueError('Unavailable replacement forecast')
                    f['eligible']=mask;outputs[(source,)].append(f)
                    if not calibration:monthly.append(dict(month=month,category=c,alternative=source,**errors(f,mask)))
                    if source=='F_ONLY':coverage.append(dict(month=month,category=c,expected=51,evaluated=int(mask.sum()),excluded=int((~mask).sum())))
            print(f'Source replacement: completed {month}',flush=True)
        if not calibration:self.results.update({'replacement_categories.csv':summarize(outputs,['alternative']),'replacement_monthly.csv':pd.DataFrame(monthly),'replacement_coverage.csv':pd.DataFrame(coverage)})

    def run(self):
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Estimation requires a CUDA GPU; table rebuilding does not')
        selected=('main','replacement','timing') if self.exercise=='all' else (self.exercise,)
        # Verify the fixed specifications at the closed March 2023 origin first.
        for name in selected:getattr(self,name)(calibration=True)
        for name in selected:getattr(self,name)()
        destination=self.output/'summaries';destination.mkdir(parents=True,exist_ok=True)
        for name,frame in self.results.items():write(destination/name,frame.to_csv(index=False,float_format='%.17g'))
        manifest=destination/'manifest.json'
        files=json.loads(manifest.read_text())['files'] if manifest.exists() else {}
        files.update({name:sha(destination/name) for name in self.results})
        json_write(manifest,{'description':'Aggregate errors from the fixed paper specifications','files':files})
        json_write(self.output/'estimation.json',{'identity':self.identity,'new_fits':self.fitted,'reused_fits':self.cached,'exercise':self.exercise})
        return destination
