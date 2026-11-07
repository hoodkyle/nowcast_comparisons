"""Appendix comparison changing only the MRTS reference-month lag."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import build_store, CATEGORIES
from .estimation import Estimation, ROOT, sha, write, json_write, summarize
from .features import build_origin_dataset
from .tables import metrics, render, TABLES


def lag_one_panel(panel):
    """Assign each observed MRTS reference month to the following target month."""
    selected=panel.source.eq('mrts')
    if not selected.any():return panel.copy()
    out=panel.copy()
    before=out.loc[selected].copy()
    expected=pd.to_datetime(before.reference_month)+pd.DateOffset(months=2)
    if not pd.to_datetime(before.date).eq(expected).all():
        raise ValueError('Expected the original two-month MRTS panel')
    out.loc[selected,'date']=pd.to_datetime(before.reference_month)+pd.DateOffset(months=1)
    out.loc[selected,'timing_variant']='lag1_official'
    # Values and reference months are unchanged; nothing is interpolated.
    pd.testing.assert_frame_equal(before.drop(columns=['date','timing_variant']),out.loc[selected].drop(columns=['date','timing_variant']))
    keys=['source','concept','date','timing_variant']+(['state'] if 'state' in out else [])
    if out.duplicated(keys).any():raise ValueError('Duplicate keys in one-month MRTS panel')
    return out


class MRTSLagComparison(Estimation):
    def __init__(self,data_root,output):
        super().__init__(data_root,output,'main')
        self.lag_one=tuple(lag_one_panel(p) for p in self.panels['v1'])
        self.settings=json.loads((ROOT/'config/mrts_lag.json').read_text())
        import hashlib
        self.identity=hashlib.sha256((self.identity+json.dumps(self.settings,sort_keys=True)).encode()).hexdigest()

    def run_appendix(self):
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the fixed tree specification')
        # Calibrate this run at the closed origin. Input checksums and fit
        # identities apply without requiring a push or remote access.
        self.estimate(calibration=True)
        return self.estimate()

    def alternative_store(self,month,source):
        return build_store(self.msrs,*self.lag_one,source,'mrts_lag1',origin=pd.Timestamp(month))

    def estimate(self,calibration=False):
        frames=defaultdict(list);coverage=[]
        spec=self.members[self.spec['single_month']]
        months=[self.settings['calibration_origin']] if calibration else self.settings['origins']
        for month in months:
            source='F_ONLY' if month<'2024-04' else 'G_ONLY'
            baseline=self.store(month,source)
            alternative=self.alternative_store(month,source)
            # The only altered raw predictors are MRTS and its availability indicators.
            same=[c for c in baseline if c not in {'information_stage'} and '__mrts__' not in c and not c.startswith('pca_industry_mrts_')]
            pd.testing.assert_frame_equal(baseline[same],alternative[same],check_exact=True)
            for category in (['TOTAL'] if calibration else CATEGORIES):
                mask=np.asarray(build_origin_dataset(baseline,month,category).score['eligible'],bool)
                for lag,store in [(2,baseline),(1,alternative)]:
                    train,test=self.split(store,month,category)
                    eligible=mask&np.isfinite(test.target.to_numpy(float))
                    if month=='2024-03':eligible[:]=False
                    f=self.forecast(month,category,source,'early_information' if lag==2 else 'mrts_lag1',spec['name'],spec['seed'],test,lambda:self.tree(train,test,spec))
                    if not np.isfinite(f.loc[eligible,'prediction']).all():raise ValueError('Missing appendix forecast')
                    f['eligible']=eligible;frames[(lag,)].append(f)
                coverage.append(dict(origin=month,category=category,evaluated=int(eligible.sum())))
            print(f'MRTS lag comparison: completed {month}',flush=True)
        if calibration:
            json_write(self.output/'calibration.json',{'origin':self.settings['calibration_origin'],'target':'TOTAL','success':True,'identity':self.identity})
            return
        result=summarize(frames,['mrts_lag'])
        baseline=result[result.mrts_lag.eq(2)].set_index('category')
        saved=pd.read_csv(ROOT/'summaries/main.csv',dtype={'category':str},float_precision='round_trip')
        saved=saved[saved.model.eq('single_month_catboost')].set_index('category')
        if not baseline.n.equals(saved.loc[baseline.index,'n']):raise ValueError('Baseline counts differ from the paper')
        # GPU reductions can differ in their last digits; retain and report the re-estimated baseline.
        comparison=pd.DataFrame({'category':baseline.index,'rmse_difference':np.sqrt(baseline.sse/baseline.n).to_numpy()-np.sqrt(saved.loc[baseline.index,'sse']/saved.loc[baseline.index,'n']).to_numpy()})
        destination=self.output/'summaries';destination.mkdir(parents=True,exist_ok=True)
        write(destination/'mrts_lag.csv',result.to_csv(index=False,float_format='%.17g'))
        write(destination/'coverage.csv',pd.DataFrame(coverage).to_csv(index=False))
        write(self.output/'baseline_comparison.csv',comparison.to_csv(index=False))
        json_write(destination/'manifest.json',{'files':{n:sha(destination/n) for n in ['mrts_lag.csv','coverage.csv']},'identity':self.identity,'settings':self.settings})
        return destination


def build_tables(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    if sha(source/'mrts_lag.csv')!=manifest['files']['mrts_lag.csv']:raise ValueError('Changed appendix summaries')
    frame=pd.read_csv(source/'mrts_lag.csv',dtype={'category':str})
    scored=metrics(frame,['mrts_lag']).set_index(['mrts_lag','category'])
    rows=[]
    for c in ['TOTAL','MACRO','POOLED',*CATEGORIES[1:]]:
        a=scored.loc[(2,c)];b=scored.loc[(1,c)]
        rows.append([{'TOTAL':'Total retail','MACRO':'Mean category','POOLED':'Pooled categories'}.get(c,c),a.rmse,b.rmse,100*(1-b.rmse/a.rmse),a.mae,b.mae,100*(1-b.mae/a.mae)])
    name='mrts_lag_comparison';TABLES[name]=('Sensitivity to the assumed MRTS information lag','tab:mrts-lag-comparison')
    headers=['Target','RMSE: t-2','RMSE: t-1','RMSE reduction (%)','MAE: t-2','MAE: t-1','MAE reduction (%)']
    notes='Single-month CatBoost, with fixed settings and seed 17, re-estimated under both MRTS lags. All other predictors, source histories, and estimation rules are unchanged. April 2023--May 2026, excluding March 2024; identical main-comparison evaluation observations. RMSE and MAE are in percentage points. Mean category measures average eleven category-specific losses; pooled measures combine their errors first. Positive reductions indicate lower errors with MRTS through $t-1$. Current revised snapshots are used, not historical release vintages.'
    write(output/(name+'.tex'),render(name,headers,rows,notes,[3,3,1,3,3,1]))
    pd.DataFrame(rows,columns=headers).to_csv(output/(name+'.csv'),index=False)
    return output/(name+'.tex')
