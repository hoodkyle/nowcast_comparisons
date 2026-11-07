"""Build the paper's tables from unrounded aggregate forecast errors."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
CATEGORIES=['TOTAL','441','442','443','444','445','446','447','448','451','452','453']
MODELS=[('persistence','Persistence'),('local_lag_ridge','Local lag ridge'),('ols','OLS'),('ridge','Ridge'),('lasso','Lasso'),('elastic_net','Elastic net'),('two_month_pca_ridge','Two-month PCA ridge'),('state_industry_factor_ridge','State-and-industry factor ridge'),('scalar_ar1_state_space','Scalar AR(1) state space'),('random_forest','Random forest'),('histogram_gradient_boosting','Histogram gradient boosting'),('single_month_catboost','Single-month CatBoost'),('two_month_catboost','Two-month CatBoost'),('two_month_xgboost','Two-month XGBoost'),('boosted_tree_median_54','Boosted-tree median (54 members)'),('mlp','MLP'),('temporal_transformer','Temporal transformer')]
CATEGORY_NAMES=dict(zip(CATEGORIES,['Total retail','441: Motor vehicles and parts','442: Furniture and home furnishings','443: Electronics and appliances','444: Building and garden supplies','445: Food and beverage','446: Health and personal care','447: Gasoline stations','448: Clothing and accessories','451: Sporting goods, hobbies, books, and music','452: General merchandise','453: Miscellaneous retailers']))
STAGES=['early_information','marts_current','mrts_current']
STAGE_NAMES=['Early information','Current MARTS added','Current MRTS also added']
PERIODS=['full','pre_transition','post_transition']
PERIOD_NAMES=['April 2023--May 2026','April 2023--March 2024','April 2024--May 2026']
ARMS=['F_ONLY','G_ONLY','NO_PRIVATE']
ARM_NAMES=['Fiserv retained','Gamma replaces Fiserv','Official information alone']
TABLES={
'data_coverage':('Historical coverage of the target and private transaction sources','tab:data-coverage'),
'information_timing':('Assumed information availability at consecutive nowcast dates','tab:information-timing'),
 'timing_information':('Information sets in the timing comparison','tab:timing-information'),
'main_headline':('Selected nowcast specifications: state total-retail accuracy','tab:main-headline'),
'main_categories':('Nowcast RMSE by retail subsector','tab:main-categories'),
'neural_seed_summary':('Neural forecast averaging: state total-retail accuracy','tab:neural-seeds'),
'source_replacement_accuracy':('Forecast accuracy under an assumed April 2023 source replacement','tab:source-replacement-accuracy'),
'source_replacement_gains':('Forecast error reductions under alternative transaction-source choices','tab:source-replacement-gains'),
 'timing_accuracy':('Single-month CatBoost accuracy as national information becomes available','tab:timing-accuracy'),
 'timing_gains':('Incremental accuracy gains from additional national information','tab:timing-gains'),
'main_full_comparison':('Full nowcast comparison: total retail and mean category accuracy','tab:main-full'),
'main_coverage':('Common evaluation-sample coverage','tab:main-coverage'),
'main_pooled':('Full nowcast comparison: pooled category accuracy','tab:main-pooled'),
'main_category_rmse_detail':('Full nowcast comparison: category RMSE','tab:main-category-rmse'),
'main_category_mae_detail':('Full nowcast comparison: category MAE','tab:main-category-mae'),
'neural_seed_detail':('Neural estimation variability and forecast averaging by target','tab:neural-seed-detail'),
'local_ridge_training_sample':('Local lag ridge under alternative estimation-sample restrictions','tab:local-ridge-sample'),
'source_replacement_categories':('Source-replacement forecast accuracy by retail target','tab:source-replacement-categories'),
'source_replacement_monthly':('Monthly total-retail forecast accuracy under alternative source choices','tab:source-replacement-monthly'),
'computing_environment':('Computing environment','tab:computing-environment')}


def metrics(frame,groups):
    """Score individual categories, pooled errors, and equal-weight category losses."""
    if frame.duplicated(groups+['category']).any(): raise ValueError('Duplicate summary keys')
    result=[]
    for key,block in frame.groupby(groups,sort=False,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        prefix=dict(zip(groups,key));block=block.copy()
        if (block.n<=0).any() or not np.isfinite(block[['sse','sae']]).all().all():raise ValueError('Invalid error sums')
        block['rmse']=np.sqrt(block.sse/block.n);block['mae']=block.sae/block.n
        result.extend(block.to_dict('records'))
        cats=block[block.category.ne('TOTAL')]
        n=int(cats.n.sum());sse=float(cats.sse.sum());sae=float(cats.sae.sum())
        result.extend([{**prefix,'category':'POOLED','n':n,'sse':sse,'sae':sae,'rmse':np.sqrt(sse/n),'mae':sae/n},
                       {**prefix,'category':'MACRO','n':n,'sse':np.nan,'sae':np.nan,'rmse':cats.rmse.mean(),'mae':cats.mae.mean()}])
    return pd.DataFrame(result)


def read(source):
    source=Path(source);manifest=json.loads((source/'manifest.json').read_text())
    result={}
    for name,digest in manifest['files'].items():
        path=source/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Summary content changed: '+name)
        result[name]=pd.read_csv(path,dtype={'category':str},float_precision='round_trip')
    return result


def contents(source, requested=None):
    raw=read(ROOT/'summaries')
    if Path(source).resolve()!=(ROOT/'summaries').resolve():
        supplied=read(source)
        for name in (requested or TABLES):
            needed=required_summaries(name)
            if set(needed)-set(supplied):raise ValueError('Estimate the required comparison first: '+name)
        raw.update(supplied)
    main=metrics(raw['main.csv'],['model']).set_index(['model','category'])
    local=metrics(raw['local_ridge.csv'],['model']).set_index(['model','category'])
    seeds=metrics(raw['neural_seeds.csv'],['model','seed'])
    timing=metrics(raw['timing.csv'],['stage','period']).set_index(['stage','period','category'])
    replacement=metrics(raw['replacement_categories.csv'],['alternative']).set_index(['alternative','category'])
    monthly=raw['replacement_monthly.csv'].copy();monthly['mae']=monthly.sae/monthly.n;monthly['rmse']=np.sqrt(monthly.sse/monthly.n);monthly=monthly.set_index(['month','category','alternative'])
    result={};main_notes='April 2023--May 2026, excluding March 2024. Common observations: 1,887 total retail and 20,307 across eleven subsectors. RMSE and MAE are in percentage points. Mean category losses average category-specific measures equally. Neural forecasts average seeds 0, 1, and 2 before scoring. Percentage reductions use unrounded losses relative to persistence.'
    rep_notes='April 2023--March 2024; 50 states and the District of Columbia. Common samples: 612 total retail and 6,002 subsector observations. Assumed Fiserv loss in April 2023 and immediate Gamma acquisition with its observed history. Electronics (443) is excluded as a target and category-specific predictor; published aggregates are unchanged. All alternatives use MSRS/MARTS through $t-1$, MRTS through $t-2$, and own-target histories. Fixed single-month CatBoost uses 400 iterations, depth 6, learning rate 0.05, leaf regularization 3, seed 17, and expanding estimation histories strictly before $t$. RMSE and MAE are in percentage points; mean category losses average ten category-specific measures.'
    timing_notes='Fixed single-month CatBoost and common main evaluation observations at all three stages. Fiserv before April 2024; Gamma with its history afterward. Early information uses private measures for $t$, MSRS/MARTS through $t-1$, and MRTS through $t-2$. Later stages add current MARTS and then current MRTS while retaining lagged predictors. MSRS for $t$ is excluded. Current revised snapshots and assumed availability are used. Mean category losses average eleven category-specific measures. Percentage reductions use unrounded losses; positive values indicate lower errors. March 2024 is excluded.'
    def put(name,headers,rows,notes=main_notes,decimals=None):result[name]=(headers,rows,notes,decimals or [3]*(len(headers)-1))
    def vals(model,cat='TOTAL'):return [main.loc[(model,cat),m] for m in ('rmse','mae')]
    def gain(model,cat='TOTAL'):return 100*(1-main.loc[(model,cat),'rmse']/main.loc[('persistence',cat),'rmse'])
    headline={'persistence','local_lag_ridge','state_industry_factor_ridge','random_forest','single_month_catboost','two_month_catboost','boosted_tree_median_54','mlp','temporal_transformer'}
    put('main_headline',['Specification','RMSE','MAE','RMSE reduction (%)'],[[label,*vals(m),gain(m)] for m,label in MODELS if m in headline],decimals=[3,3,1])
    chosen=['persistence','single_month_catboost','two_month_catboost','boosted_tree_median_54','mlp','temporal_transformer']
    put('main_categories',['Category',*map(dict(MODELS).get,chosen)],[[CATEGORY_NAMES[c],*[main.loc[(m,c),'rmse'] for m in chosen]] for c in CATEGORIES[1:]])
    put('main_full_comparison',['Specification','Total RMSE','Total MAE','Mean category RMSE','Mean category MAE','Total reduction (%)','Mean category reduction (%)'],[[label,*vals(m),*vals(m,'MACRO'),gain(m),gain(m,'MACRO')] for m,label in MODELS],decimals=[3,3,3,3,1,1])
    put('main_pooled',['Specification','RMSE','MAE','RMSE reduction (%)'],[[label,*vals(m,'POOLED'),gain(m,'POOLED')] for m,label in MODELS],decimals=[3,3,1])
    for metric in ('rmse','mae'):
        put('main_category_'+metric+'_detail',['Specification',*CATEGORIES],[[label,*[main.loc[(m,c),metric] for c in CATEGORIES]] for m,label in MODELS])
    cov=raw['coverage.csv'].groupby('category').agg(expected=('expected','sum'),evaluated=('common_count','sum'),months=('evaluable','sum'))
    put('main_coverage',['Category','Potential observations','Evaluated observations','Evaluated months'],[[CATEGORY_NAMES[c],*cov.loc[c].tolist()] for c in sorted(CATEGORIES)],decimals=[0,0,0])
    stats=seeds.groupby(['model','category']).agg(rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),mae_mean=('mae','mean'),mae_std=('mae','std'))
    put('neural_seed_summary',['Method','Mean seed RMSE','Seed SD','Averaged forecast RMSE'],[[label,stats.loc[(m,'TOTAL'),'rmse_mean'],stats.loc[(m,'TOTAL'),'rmse_std'],main.loc[(m,'TOTAL'),'rmse']] for m,label in [('mlp','MLP'),('temporal_transformer','Transformer')]])
    put('neural_seed_detail',['Method / target','RMSE mean','RMSE SD','RMSE average','MAE mean','MAE SD','MAE average'],[[label+': '+{'TOTAL':'Total','MACRO':'Mean category','POOLED':'Pooled'}.get(c,c),stats.loc[(m,c),'rmse_mean'],stats.loc[(m,c),'rmse_std'],main.loc[(m,c),'rmse'],stats.loc[(m,c),'mae_mean'],stats.loc[(m,c),'mae_std'],main.loc[(m,c),'mae']] for m,label in [('mlp','MLP'),('temporal_transformer','Transformer')] for c in sorted(CATEGORIES+['MACRO','POOLED'])])
    put('local_ridge_training_sample',['Target','Common sample RMSE','Common sample MAE','Own-lag sample RMSE','Own-lag sample MAE'],[[{'TOTAL':'Total retail','POOLED':'Pooled subsectors','MACRO':'Mean category'}.get(c,c),*vals('local_lag_ridge',c),*[local.loc[('local_lag_ridge_own_history',c),m] for m in ('rmse','mae')]] for c in CATEGORIES+['POOLED','MACRO']],notes=main_notes+' The own-lag sample requires only the observed estimation target and its first lag. The equation, penalty, and evaluation observations are unchanged.')
    columns=[('TOTAL','mae'),('TOTAL','rmse'),('MACRO','mae'),('MACRO','rmse')]
    headers=['Alternative','Total MAE','Total RMSE','Mean category MAE','Mean category RMSE']
    put('source_replacement_accuracy',headers,[[label,*[replacement.loc[(a,c),m] for c,m in columns]] for a,label in zip(ARMS,ARM_NAMES)],rep_notes)
    pairs=[('G_ONLY','F_ONLY','Gamma relative to Fiserv'),('G_ONLY','NO_PRIVATE','Gamma relative to official information alone'),('F_ONLY','NO_PRIVATE','Fiserv relative to official information alone')]
    put('source_replacement_gains',['Comparison',*headers[1:]],[[label,*[100*(1-replacement.loc[(a,c),m]/replacement.loc[(b,c),m]) for c,m in columns]] for a,b,label in pairs],rep_notes+' Entries are percentage reductions from unrounded losses; positive values indicate lower errors for the first alternative.',decimals=[1]*4)
    put('source_replacement_categories',['Retail target','Fiserv MAE','Fiserv RMSE','Gamma MAE','Gamma RMSE','Official-only MAE','Official-only RMSE'],[[CATEGORY_NAMES[c],*[replacement.loc[(a,c),m] for a in ARMS for m in ('mae','rmse')]] for c in CATEGORIES if c!='443'],rep_notes)
    put('source_replacement_monthly',['Month','Fiserv MAE','Fiserv RMSE','Gamma MAE','Gamma RMSE','Official-only MAE','Official-only RMSE'],[[pd.Timestamp(d).strftime('%b. %Y'),*[monthly.loc[(d,'TOTAL',a),m] for a in ARMS for m in ('mae','rmse')]] for d in sorted(raw['replacement_monthly.csv'].month.unique())],rep_notes)
    columns=[('TOTAL','rmse'),('TOTAL','mae'),('MACRO','rmse'),('MACRO','mae')]
    put('timing_accuracy',['Period / stage','Total RMSE','Total MAE','Mean category RMSE','Mean category MAE'],[[pname+': '+sname,*[timing.loc[(stage,p,c),m] for c,m in columns]] for p,pname in zip(PERIODS,PERIOD_NAMES) for stage,sname in zip(STAGES,STAGE_NAMES)],timing_notes)
    comparisons=[(STAGES[0],STAGES[1],'Early to current MARTS'),(STAGES[1],STAGES[2],'Current MARTS to current MRTS'),(STAGES[0],STAGES[2],'Early to current MRTS')]
    put('timing_gains',['Period / change','Total RMSE (%)','Total MAE (%)','Mean category RMSE (%)','Mean category MAE (%)'],[[pname+': '+label,*[100*(1-timing.loc[(b,p,c),m]/timing.loc[(a,p,c),m]) for c,m in columns]] for p,pname in zip(PERIODS,PERIOD_NAMES) for a,b,label in comparisons],timing_notes,decimals=[1]*4)
    descriptive=json.loads((ROOT/'config/descriptive_tables.json').read_text())
    for name,spec in descriptive.items():put(name,spec['headers'],spec['rows'],spec['notes'])
    if set(result)!=set(TABLES):raise ValueError('Incomplete table definitions')
    return result


def escape(value):
    return str(value).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')


def render(name,headers,rows,notes,decimals):
    caption,label=TABLES[name]
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{@{}l'+'r'*(len(headers)-1)+'@{}}',r'\toprule',' & '.join(map(escape,headers))+r' \\',r'\midrule']
    for row in rows:
        cells=[escape(row[0])]+[f'{v:.{d}f}' if isinstance(v,(int,float,np.number)) and np.isfinite(v) else ('--' if isinstance(v,(int,float,np.number)) else escape(v)) for v,d in zip(row[1:],decimals)]
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\par\smallskip',r'\begin{minipage}{\textwidth}\footnotesize',r'\textit{Notes:} '+notes,r'\end{minipage}',r'\end{table}','']
    return '\n'.join(lines)


def build(source,output,table='all'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    requested=list(TABLES) if table=='all' else [table]
    if set(requested)-set(TABLES):raise ValueError('Unknown table')
    values=contents(source,requested)
    for name in requested:
        headers,rows,notes,decimals=values[name]
        (output/(name+'.tex')).write_text(render(name,headers,rows,notes,decimals))
        pd.DataFrame(rows,columns=headers).to_csv(output/(name+'.csv'),index=False,float_format='%.17g')
    return requested


def required_summaries(table):
    if table in {'data_coverage','information_timing','timing_information','computing_environment'}:return []
    if table.startswith('source_replacement'):return ['replacement_categories.csv','replacement_monthly.csv']
    if table.startswith('timing'):return ['timing.csv']
    if table=='main_coverage':return ['coverage.csv']
    if table.startswith('neural'):return ['main.csv','neural_seeds.csv']
    if table=='local_ridge_training_sample':return ['main.csv','local_ridge.csv']
    return ['main.csv']
