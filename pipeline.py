import numpy as np, pandas as pd, sys
import os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')+os.sep
FILES=['event_1988_30Jun.csv','event_1990_02May.csv','event_1995_03Jun.csv','event_1998_23Mar.csv','event_2005_01Jun.csv']
NAMES=['30 Jun 1988','02 May 1990','03 Jun 1995','23 Mar 1998','01 Jun 2005']
def load():
    return [pd.read_csv(D+f) for f in FILES]
def nz(x,mn,mx): return (x-mn)/(mx-mn+1e-12)
def dn(x,mn,mx): return x*(mx-mn+1e-12)+mn
def metrics(o,s):
    o=np.asarray(o,float); s=np.asarray(s,float)
    nse=1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
    r=np.corrcoef(o,s)[0,1]
    kge=1-np.sqrt((r-1)**2+(s.std(ddof=1)/o.std(ddof=1)-1)**2+(s.mean()/o.mean()-1)**2)
    return dict(NSE=nse,RMSE=np.sqrt(np.mean((o-s)**2)),MAE=np.mean(np.abs(o-s)),KGE=kge,PBIAS=100*np.sum(o-s)/np.sum(o))
# subtractive clustering (Chiu 1994 / MATLAB subclust defaults)
def subclust(X,ra=0.8,sq=1.25,acc=0.5,rej=0.15):
    mn=X.min(0); mx=X.max(0); Xn=(X-mn)/np.where(mx-mn==0,1,mx-mn)
    alpha=4/ra**2; rb=ra*sq; beta=4/rb**2
    d2=((Xn[:,None,:]-Xn[None,:,:])**2).sum(-1)
    P=np.exp(-alpha*d2).sum(1); P1=P.max(); centers=[]
    while True:
        k=P.argmax(); pk=P[k]
        if pk>acc*P1: ok=True
        elif pk<rej*P1: break
        else:
            dmin=np.sqrt(min(((Xn[k]-Xn[c])**2).sum() for c in centers)) if centers else 1e9
            if dmin/ra+pk/P1>=1: ok=True
            else: P[k]=0; continue
        centers.append(k); P=P-pk*np.exp(-beta*((Xn-Xn[k])**2).sum(1))
        P[P<0]=0
        if len(centers)>50: break
    return len(centers)
def lagX(cols,lag,N):
    rows=[]
    for t in range(lag-1,N-1):
        r=[]
        for k in range(lag):
            for c in cols: r.append(c[t-k])
        rows.append(r)
    return np.array(rows)
def ols(X,y):
    A=np.c_[X,np.ones(len(X))]; w,*_=np.linalg.lstsq(A,y,rcond=None); return w
def pred(w,X): return np.c_[X,np.ones(len(X))]@w

def subclust_centers(X,ra=0.8,sq=1.25,acc=0.5,rej=0.15):
    mn=X.min(0); mx=X.max(0); rng=np.where(mx-mn==0,1,mx-mn); Xn=(X-mn)/rng
    alpha=4/ra**2; beta=4/(ra*sq)**2
    d2=((Xn[:,None,:]-Xn[None,:,:])**2).sum(-1)
    P=np.exp(-alpha*d2).sum(1); P1=P.max(); C=[]
    while True:
        k=P.argmax(); pk=P[k]
        if pk>acc*P1: pass
        elif pk<rej*P1: break
        else:
            dmin=min(np.sqrt(((Xn[k]-Xn[c])**2).sum()) for c in C) if C else 1e9
            if not (dmin/ra+pk/P1>=1): P[k]=0; continue
        C.append(k); P=P-pk*np.exp(-beta*((Xn-Xn[k])**2).sum(1)); P[P<0]=0
    centers=X[C]; sig=ra*(mx-mn)/np.sqrt(8)
    return centers,sig
class ANFIS:
    """First-order Sugeno ANFIS, Gaussian MFs, genfis2 init, Jang hybrid learning (MATLAB-like step-size rules)."""
    def __init__(s,X,y,ra=0.8):
        C,sig=subclust_centers(np.c_[X,y],ra); d=X.shape[1]
        s.c=C[:,:d].copy(); s.sg=np.tile(np.where(sig[:d]==0,1e-6,sig[:d]),(len(C),1)); s.R=len(C)
        s.lse(X,y)
    def fire(s,X):
        w=np.exp(-((X[:,None,:]-s.c[None])**2/(2*s.sg[None]**2)).sum(-1)); return w/(w.sum(1,keepdims=True)+1e-300),w
    def design(s,X):
        wn,_=s.fire(X); Xa=np.c_[X,np.ones(len(X))]
        return (wn[:,:,None]*Xa[:,None,:]).reshape(len(X),-1)
    def lse(s,X,y):
        A=s.design(X); s.th,*_=np.linalg.lstsq(A,y,rcond=None)
    def predict(s,X): return s.design(X)@s.th
    def train(s,X,y,Xv,yv,epochs=50,ss=0.01):
        tr=[];va=[];best=(np.inf,None); hist=[]
        for ep in range(epochs):
            s.lse(X,y); e=s.predict(X)-y; rm=np.sqrt(np.mean(e**2)); tr.append(rm)
            v=np.sqrt(np.mean((s.predict(Xv)-yv)**2)); va.append(v)
            if v<best[0]: best=(v,(s.c.copy(),s.sg.copy(),s.th.copy()))
            # gradient wrt premise
            wn,w=s.fire(X); Xa=np.c_[X,np.ones(len(X))]
            f=Xa@s.th.reshape(s.R,-1).T            # N x R rule outputs
            out=(wn*f).sum(1)
            dO_dwn=f-out[:,None]                   # d out / d w_r * w_r... using normalized
            g=(e[:,None]*dO_dwn*wn)                # N x R  (dE/dlog w_r)
            dx=X[:,None,:]-s.c[None]
            gc=(g[:,:,None]*dx/s.sg[None]**2).sum(0)
            gs=(g[:,:,None]*dx**2/s.sg[None]**3).sum(0)
            G=np.concatenate([gc.ravel(),gs.ravel()]); n=np.linalg.norm(G)
            if n>0:
                step=ss*G/n
                s.c-=step[:gc.size].reshape(gc.shape); s.sg-=step[gc.size:].reshape(gs.shape); s.sg=np.maximum(s.sg,1e-6)
            hist.append(rm)
            if len(hist)>=5 and all(hist[-i]<hist[-i-1] for i in range(1,5)): ss*=1.1
            if len(hist)>=5 and (hist[-1]-hist[-2])*(hist[-2]-hist[-3])<0 and (hist[-3]-hist[-4])*(hist[-4]-hist[-5])<0 and (hist[-2]-hist[-3])*(hist[-3]-hist[-4])<0: ss*=0.9
        s.c,s.sg,s.th=best[1]; return np.array(tr),np.array(va)
