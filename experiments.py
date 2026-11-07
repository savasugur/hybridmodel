import sys, json; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from pipeline import *
LAG=2
def phys_metrics(t,o,s):
    m=metrics(o,s)
    ip,isim=np.argmax(o),np.argmax(s)
    m['PeakErr']=100*(s.max()-o.max())/o.max()
    m['PeakTime']=t[isim]-t[ip]
    vo=np.trapezoid(o,t*60); vs=np.trapezoid(s,t*60)
    m['VolErr']=100*(vs-vo)/vo
    return m
def scale(evs,train,mode):
    """returns per-event dict with Pn,On,Hn and Q scale (mn,mx) for back-transform"""
    out=[]
    if mode=='train':
        Pall=np.concatenate([evs[i].Rainfall for i in train]); Qall=np.concatenate([evs[i].Qobs for i in train])
        pm,pM,qm,qM=Pall.min(),Pall.max(),Qall.min(),Qall.max()
    for e in evs:
        if mode=='own':
            pm,pM,qm,qM=e.Rainfall.min(),e.Rainfall.max(),e.Qobs.min(),e.Qobs.max()
        if mode=='hms':  # leakage-free per-event scale: HEC-HMS simulated peak (known before observation)
            pm,pM,qm,qM=e.Rainfall.min(),e.Rainfall.max(),0.0,e.Qhms.max()
        out.append(dict(P=nz(e.Rainfall.values,pm,pM),O=nz(e.Qobs.values,qm,qM),H=nz(e.Qhms.values,qm,qM),qm=qm,qM=qM,t=e.Time_min.values.astype(float),Qo=e.Qobs.values,Qh=e.Qhms.values))
    return out
def X_of(d,model,Qlag=None):
    Q=d['O'] if Qlag is None else Qlag
    if model in('anfis','arx'): return lagX([d['P'],Q],LAG,len(Q))
    if model=='hyb': return lagX([d['P'],d['H'],Q],LAG,len(Q))
    if model=='hybnoq': return lagX([d['P'],d['H']],LAG,len(Q))
def y_of(d,model):
    return (d['O']-d['H'])[LAG:] if model in('hyb','hybnoq') else d['O'][LAG:]
def fit(S,train,model):
    X=np.vstack([X_of(S[i],model) for i in train]); y=np.concatenate([y_of(S[i],model) for i in train])
    if model=='arx': return ('lin',ols(X,y)),None
    nf=len(X)-int(len(X)*0.2); m=ANFIS(X[:nf],y[:nf]); tr,va=m.train(X[:nf],y[:nf],X[nf:],y[nf:]); return ('anfis',m),(tr,va,m.R)
def f(mod,X): return pred(mod[1],X) if mod[0]=='lin' else mod[1].predict(X)
def simulate(d,model,mod,recursive):
    N=len(d['O']); base=d['H'] if model in('hyb','hybnoq') else 0*d['O']
    if not recursive or model=='hybnoq':
        return f(mod,X_of(d,model))+base[LAG:]
    q=d['O'].copy(); q[LAG:]=np.nan   # only initial LAG observed values as initial condition
    for t in range(LAG-1,N-1):
        x=[]
        for k in range(LAG):
            x+= [d['P'][t-k],d['H'][t-k],q[t-k]] if model=='hyb' else [d['P'][t-k],q[t-k]]
        v=f(mod,np.array([x]))[0]+base[t+1]
        q[t+1]=max(v,(0-d['qm'])/(d['qM']-d['qm']+1e-12))  # no negative discharge
    return q[LAG:]
def back(d,s): return dn(s,d['qm'],d['qM'])
def run(test,mode):
    evs=load(); train=[i for i in range(5) if i!=test]; S=scale(evs,train,mode); d=S[test]
    t=d['t'][LAG:]; o=d['Qo'][LAG:]; res={}; info={}
    res['HEC-HMS']=d['Qh'][LAG:]
    res['Persistence']=d['Qo'][LAG-1:-1]
    mods={}
    for m in ['arx','anfis','hyb','hybnoq']:
        mods[m],info[m]=fit(S,train,m)
    res['ARX (one-step)']=back(d,simulate(d,'arx',mods['arx'],False))
    res['ANFIS (one-step)']=back(d,simulate(d,'anfis',mods['anfis'],False))
    res['Hybrid (one-step)']=back(d,simulate(d,'hyb',mods['hyb'],False))
    res['ANFIS (recursive)']=back(d,simulate(d,'anfis',mods['anfis'],True))
    res['Hybrid (recursive)']=back(d,simulate(d,'hyb',mods['hyb'],True))
    res['Hybrid (no Q feedback)']=back(d,simulate(d,'hybnoq',mods['hybnoq'],False))
    met={k:phys_metrics(t,o,np.asarray(v,float)) for k,v in res.items()}
    rules={k:(v[2] if v else 1) for k,v in info.items()}
    return t,o,res,met,rules,info
if __name__=='__main__':
    out={}
    for mode in ['own','train','hms']:
        for test in range(5):
            t,o,res,met,rules,_=run(test,mode)
            out[f'{mode}|{test}']=dict(met={k:{a:float(b) for a,b in v.items()} for k,v in met.items()},rules=rules)
            print(f'\n### norm={mode} test={NAMES[test]} rules={rules}')
            print(pd.DataFrame(met).T.round(4).to_string())
    json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'experiments.json','w'),indent=1)
