import sys, json; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from experiments import *
# 1) optimizer sensitivity of the original-protocol hybrid on 2005
evs=load(); train=[0,1,2,3]; S=scale(evs,train,'own'); d=S[4]
X=np.vstack([X_of(S[i],'hyb') for i in train]); y=np.concatenate([y_of(S[i],'hyb') for i in train]); nf=len(X)-int(len(X)*0.2)
vals=[]
for ss in [0.002,0.005,0.01,0.02,0.05]:
    for ep in [10,25,50,100]:
        m=ANFIS(X[:nf],y[:nf]); m.train(X[:nf],y[:nf],X[nf:],y[nf:],epochs=ep,ss=ss)
        s=m.predict(X_of(d,'hyb'))+d['H'][LAG:]; vals.append(metrics(d['O'][LAG:],s)['NSE'])
print('hybrid 2005 own-norm NSE over 20 optimizer settings: min %.4f max %.4f median %.4f'%(min(vals),max(vals),np.median(vals)))
# 2) LOEO summary (hms normalisation = primary leakage-free)
J=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'experiments.json'))
for mode in ['hms','train','own']:
    rows={}
    for t in range(5):
        for k,v in J[f'{mode}|{t}']['met'].items(): rows.setdefault(k,[]).append(v['NSE'])
    print(mode, {k:(round(np.median(v),3),round(min(v),3),sum(1 for a,b in zip(v,rows['HEC-HMS']) if a>b)) for k,v in rows.items()})
# 3) sensitivity: drop 1998 (possible duplicate of 1995), LOEO over 4 events, hms norm
import experiments as E
keep=[0,1,2,4]
evs_all=load()
for test in keep:
    tr=[i for i in keep if i!=test]
    S=scale(evs_all,tr,'hms'); d=S[test]; o=d['Qo'][LAG:]; t=d['t'][LAG:]
    mods={m:fit(S,tr,m)[0] for m in ['anfis','hyb']}
    r={'HMS':d['Qh'][LAG:],'Pers':d['Qo'][LAG-1:-1],'ANFIS1':back(d,simulate(d,'anfis',mods['anfis'],False)),'Hyb1':back(d,simulate(d,'hyb',mods['hyb'],False)),
       'ANFISrec':back(d,simulate(d,'anfis',mods['anfis'],True)),'Hybrec':back(d,simulate(d,'hyb',mods['hyb'],True))}
    print('no1998 test',NAMES[test],{k:round(metrics(o,v)['NSE'],3) for k,v in r.items()})
