import sys, json; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from experiments import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.linewidth':0.6,'axes.grid':True,'grid.color':'#dddddd','grid.linewidth':0.4,
  'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'lines.linewidth':1.2,'savefig.dpi':600})
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'figures','')
import os; os.makedirs(OUT,exist_ok=True)
C=dict(obs='#000000',hms='#0072B2',pers='#999999',anfis='#D55E00',hyb='#009E73',hyb2='#CC79A7')
evs=load()
# Fig 3: observed + HEC-HMS + hyetograph, 5 panels
fig=plt.figure(figsize=(7.1,8.2)); 
for i,e in enumerate(evs):
    ax=fig.add_subplot(3,2,i+1); t=e.Time_min/60
    ax.plot(t,e.Qobs*1000,color=C['obs'],marker='o',ms=2,lw=1,label='Observed')
    ax.plot(t,e.Qhms*1000,color=C['hms'],ls='--',lw=1.2,label='HEC-HMS')
    ax.set_ylabel('Discharge (L/s)'); ax.set_xlabel('Time since event start (h)')
    ax2=ax.twinx(); ax2.bar(t,e.Rainfall,width=5/60*0.9,color='#7f7f7f',alpha=0.55); ax2.invert_yaxis(); ax2.set_ylim(e.Rainfall.max()*3.2+1e-9,0)
    ax2.set_ylabel('Rainfall (mm/5 min)',color='#555555'); ax2.grid(False); ax2.spines['right'].set_visible(True)
    ax.set_ylim(0,max(e.Qobs.max(),e.Qhms.max())*1000*1.6)
    ax.set_title(f'({chr(97+i)}) {NAMES[i]}',loc='left',fontsize=8.5)
    if i==0: ax.legend(loc='upper right',fontsize=7)
fig.tight_layout(); fig.savefig(OUT+'fig3_events_hms.pdf'); fig.savefig(OUT+'fig3_events_hms.png',dpi=300); plt.close()

# Fig 5: training curves (primary protocol: hms normalisation, 2005 test)
train=[0,1,2,3]; S=scale(evs,train,'hms')
_,ia=fit(S,train,'anfis'); _,ih=fit(S,train,'hyb')
fig,axs=plt.subplots(1,2,figsize=(7.1,2.4))
for ax,(tr,va,R),ttl in zip(axs,[ia,ih],['(a) Standalone ANFIS','(b) Hybrid residual ANFIS']):
    ax.plot(range(1,len(tr)+1),tr,color=C['hms'],label='Training'); ax.plot(range(1,len(va)+1),va,color=C['anfis'],ls='--',label='Internal validation')
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (scaled discharge)'); ax.set_title(f'{ttl} — {R} rule(s)',loc='left',fontsize=8.5)
axs[0].legend(); fig.tight_layout(); fig.savefig(OUT+'fig4_training.pdf'); fig.savefig(OUT+'fig4_training.png',dpi=300); plt.close()

# Fig 6: 2005 test hydrographs, physical units
t,o,res,met,rules,_=run(4,'hms'); th=(t-t[0])/60
fig,axs=plt.subplots(2,1,figsize=(7.1,5.2),sharex=True)
ax=axs[0]; ax.plot(th,o*1000,color=C['obs'],marker='o',ms=2,lw=1,label='Observed')
for k,c,ls in [('Persistence',C['pers'],':'),('ANFIS (one-step)',C['anfis'],'--'),('Hybrid (one-step)',C['hyb'],'-')]:
    ax.plot(th,res[k]*1000,color=c,ls=ls,label=f"{k}  NSE={met[k]['NSE']:.3f}")
ax.set_ylabel('Discharge (L/s)'); ax.set_title('(a) One-step-ahead updating mode (observed $Q(t)$, $Q(t-1)$ as inputs)',loc='left',fontsize=8.5); ax.legend(fontsize=7)
ax=axs[1]; ax.plot(th,o*1000,color=C['obs'],marker='o',ms=2,lw=1,label='Observed')
for k,c,ls in [('HEC-HMS',C['hms'],'--'),('ANFIS (recursive)',C['anfis'],'-.'),('Hybrid (recursive)',C['hyb'],'-'),('Hybrid (no Q feedback)',C['hyb2'],':')]:
    ax.plot(th,np.asarray(res[k])*1000,color=c,ls=ls,label=f"{k}  NSE={met[k]['NSE']:.3f}")
ax.set_ylabel('Discharge (L/s)'); ax.set_xlabel('Time since event start (h)')
ax.set_title('(b) Simulation mode (no observed discharge after initialisation)',loc='left',fontsize=8.5); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT+'fig5_test2005.pdf'); fig.savefig(OUT+'fig5_test2005.png',dpi=300); plt.close()

# Fig 7: LOEO NSE dot plot
J=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','')+'experiments.json'))
models=['HEC-HMS','Persistence','ARX (one-step)','ANFIS (one-step)','Hybrid (one-step)','Hybrid (recursive)','Hybrid (no Q feedback)','ANFIS (recursive)']
mk=['o','s','^','D','v','P','X','*']; cols=['#0072B2','#999999','#E69F00','#D55E00','#009E73','#009E73','#CC79A7','#D55E00']
fig,ax=plt.subplots(figsize=(7.1,3.0))
for j,m in enumerate(models):
    v=[J[f'hms|{t}']['met'][m]['NSE'] for t in range(5)]
    x=np.arange(5)+(j-3.5)*0.09
    ax.scatter(x,np.clip(v,-0.5,1),marker=mk[j],s=(40 if mk[j]=='*' else 22),color=cols[j],label=m,edgecolor='white',linewidth=0.4,zorder=3)
ax.axhline(0,color='#555555',lw=0.6); ax.set_ylim(-0.55,1.03)
ax.set_xticks(range(5)); ax.set_xticklabels([f'{n}\n(held out)' for n in NAMES]); ax.set_ylabel('NSE (m$^3$/s)')
ax.text(4.45,-0.5,'values < −0.5 plotted at −0.5',fontsize=6.5,ha='right',color='#555555')
ax.legend(ncol=4,fontsize=6.8,loc='lower center',bbox_to_anchor=(0.5,1.0))
fig.tight_layout(); fig.savefig(OUT+'fig6_loeo.pdf'); fig.savefig(OUT+'fig6_loeo.png',dpi=300); plt.close()
print('ok',rules)
