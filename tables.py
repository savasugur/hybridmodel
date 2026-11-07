import sys, json; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from pipeline import NAMES
J=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'experiments.json'))
order=['HEC-HMS','Persistence','ARX (one-step)','ANFIS (one-step)','Hybrid (one-step)','ANFIS (recursive)','Hybrid (recursive)','Hybrid (no Q feedback)']
def f(x,n=3): return f'{x:.{n}f}'.replace('-','$-$')
# Table: test event 2005, primary protocol, physical units
L=[]
m=J['hms|4']['met']
for k in order:
    v=m[k]; L.append(f"{k} & {f(v['NSE'])} & {f(v['KGE'])} & {f(v['RMSE']*1000,2)} & {f(v['MAE']*1000,2)} & {f(v['PBIAS'],1)} & {f(v['PeakErr'],1)} & {f(v["PeakTime"],0)} & {f(v['VolErr'],1)} \\\\")
    if k in('Persistence','Hybrid (one-step)'): L.append('\\midrule')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'tab_test.tex','w').write('\n'.join(L))
# Table: LOEO NSE / PBIAS all events
L=[]
for k in order:
    L.append(k+' & '+' & '.join(f(J[f'hms|{t}']['met'][k]['NSE']) for t in range(5))+' \\\\')
    if k in('Persistence','Hybrid (one-step)'): L.append('\\midrule')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'tab_loeo.tex','w').write('\n'.join(L))
# Table: normalisation sensitivity, 2005 test NSE
L=[]
for k in order:
    L.append(k+' & '+' & '.join(f(J[f'{md}|4']['met'][k]['NSE']) for md in ['own','train','hms'])+' \\\\')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'tab_norm.tex','w').write('\n'.join(L))
# Full LOEO physical metrics (supplementary)
L=[]
for t in range(5):
    for k in order:
        v=J[f'hms|{t}']['met'][k]
        L.append(f"{NAMES[t] if k=='HEC-HMS' else ''} & {k} & {f(v['NSE'])} & {f(v['KGE'])} & {f(v['RMSE']*1000,3)} & {f(v['MAE']*1000,3)} & {f(v['PBIAS'],1)} & {f(v['PeakErr'],1)} & {f(v["PeakTime"],0)} & {f(v['VolErr'],1)} \\\\")
    L.append('\\midrule')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'tab_full.tex','w').write('\n'.join(L[:-1]))
print(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'tab_test.tex').read()); print(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'tab_loeo.tex').read()); print(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'tab_norm.tex').read())
