import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from pipeline import *
ev=load(); LAG=2; test=4; train=[0,1,2,3]
for e in ev:
    e['Pn']=nz(e.Rainfall,e.Rainfall.min(),e.Rainfall.max()); mn,mx=e.Qobs.min(),e.Qobs.max()
    e['On']=nz(e.Qobs,mn,mx); e['Hn']=nz(e.Qhms,mn,mx)
def feats(e,hyb):
    N=len(e); cols=[e.Pn.values,e.Hn.values,e.On.values] if hyb else [e.Pn.values,e.On.values]
    X=lagX(cols,LAG,N); y=(e.On.values-e.Hn.values if hyb else e.On.values)[LAG:]
    return X,y
for hyb in [False,True]:
    X=np.vstack([feats(ev[i],hyb)[0] for i in train]); y=np.concatenate([feats(ev[i],hyb)[1] for i in train])
    nfit=len(X)-int(np.floor(len(X)*0.2))
    print('hyb' if hyb else 'anfis','rules=',subclust(np.c_[X[:nfit],y[:nfit]]))
    w=ols(X[:nfit],y[:nfit])
    for i in range(5):
        Xi,yi=feats(ev[i],hyb); o=ev[i].On.values[LAG:]
        s=pred(w,Xi)+(ev[i].Hn.values[LAG:] if hyb else 0)
        m=metrics(o,s); print(NAMES[i], {k:round(v,4) for k,v in m.items()})
    if hyb:
        print('HMS'); 
        for i in range(5):
            m=metrics(ev[i].On.values[LAG:],ev[i].Hn.values[LAG:]); print(NAMES[i],{k:round(v,4) for k,v in m.items()})
