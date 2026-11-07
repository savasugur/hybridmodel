import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from pipeline import *
ev=load(); LAG=2; train=[0,1,2,3]
for e in ev:
    e['Pn']=nz(e.Rainfall,e.Rainfall.min(),e.Rainfall.max()); mn,mx=e.Qobs.min(),e.Qobs.max()
    e['On']=nz(e.Qobs,mn,mx); e['Hn']=nz(e.Qhms,mn,mx)
def feats(e):
    X=lagX([e.Pn.values,e.Hn.values,e.On.values],LAG,len(e)); return X,(e.On.values-e.Hn.values)[LAG:]
X=np.vstack([feats(ev[i])[0] for i in train]); y=np.concatenate([feats(ev[i])[1] for i in train])
nf=len(X)-int(len(X)*0.2)
m=ANFIS(X[:nf],y[:nf]); print('rules',m.R)
tr,va=m.train(X[:nf],y[:nf],X[nf:],y[nf:])
print('final tr %.4f va %.4f'%(tr[-1],va[-1]))
for i in range(5):
    Xi,_=feats(ev[i]); s=m.predict(Xi)+ev[i].Hn.values[LAG:]
    print(NAMES[i],{k:round(v,4) for k,v in metrics(ev[i].On.values[LAG:],s).items()})
