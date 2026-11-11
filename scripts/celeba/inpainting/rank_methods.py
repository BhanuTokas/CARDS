# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Rank every method against one inpainting/blur GT CSV (Spearman vs the random-shape-adjusted GT per task, pooled,
bootstrap CIs over concepts, and CARDS-minus-method CIs). argv[1] = GT CSV, argv[2] = moved-mask CARDS CSV
(random_cards.py). env EXTRA = "label|csv|column;..." adds methods from CSVs with task, concept_name columns;
env KEEP = comma-separated concept filter. Validity: n_valid==5, fill_frac<=0.8, coverage>=95% per concept.
Run from the repo root with post_hoc_cbm on PYTHONPATH (load_methods reads the PCBM checkpoints)."""
import collections
import csv
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0,"scripts/celeba/analysis")
from score_methods_against_strict_control_official_train import load_methods

rng=np.random.default_rng(0)
MAXF=.8;COV=.95;TASKS=["Attractive","Male"]
rows=list(csv.DictReader(open(sys.argv[1])));mvf=sys.argv[2]
print("rows",len(rows),f"max|dp-stored| {max(abs(float(r['delta_p'])-float(r['delta_p_stored'])) for r in rows):.2e}")
M=dict(load_methods())
import os

for spec in os.environ.get("EXTRA","").split(";"):
    if not spec: continue
    label,fn,col=spec.split("|")
    M[label]={}
    for r in csv.DictReader(open(fn)): M[label][(r["task"],r["concept_name"])]=float(r[col])
mv=collections.defaultdict(list)
for r in csv.DictReader(open(mvf)):
    if int(r["n_valid"])>=3: mv[r["concept"]].append([float(r[k]) for k in ("real_A","real_M","moved_A","moved_M")])
M["CARDS real-moved"]={}
for c,v in mv.items():
    a=np.array(v);M["CARDS real-moved"][("Attractive",c)]=(a[:,0]-a[:,2]).mean();M["CARDS real-moved"][("Male",c)]=(a[:,1]-a[:,3]).mean()
M["Moved mask only"]={}
for c,v in mv.items():
    a=np.array(v);M["Moved mask only"][("Attractive",c)]=a[:,2].mean();M["Moved mask only"][("Male",c)]=a[:,3].mean()
SCALES={"p":("delta_p","blob_p"),"logit":("delta_logit","blob_logit"),"logodds":("delta_logodds","blob_logodds")}
def valid(r): return int(r["n_valid"])==5 and float(r["fill_frac"])<=MAXF
T={}
for t in TASKS:
    by=collections.defaultdict(list)
    for r in rows:
        if r["target_task"]==t: by[r["concept_name"]].append(r)
    cov={c:np.mean([valid(r) for r in v]) for c,v in by.items()}
    kept=sorted(c for c,x in cov.items() if x>=COV)
    if os.environ.get('KEEP'): kept=[c for c in kept if c in os.environ['KEEP'].split(',')]
    print(f"\n{t}: scored {len(kept)}/26; partial/dropped:",{c:round(x,2) for c,x in sorted(cov.items(),key=lambda z:z[1]) if x<COV})
    T[t]={}
    for c in kept:
        v=[r for r in by[c] if valid(r)]
        T[t][c]={s:np.array([(float(r[a]),float(r[b])) for r in v]) for s,(a,b) in SCALES.items()}
        T[t][c]["area"]=np.mean([float(r["area"]) for r in v])
    cs=kept
    def sb(s, cs=cs, t=t):
        o=[]
        for _ in range(300):
            A,B=[],[]
            for c in cs:
                x=T[t][c][s];x=x[:,0]-x[:,1];p=rng.permutation(len(x));h=len(x)//2
                A.append(x[p[:h]].mean());B.append(x[p[h:]].mean())
            q=spearmanr(A,B)[0];o.append(2*q/(1+q))
        return np.mean(o)
    area=[T[t][c]["area"] for c in cs]
    for s in SCALES:
        adj=[(T[t][c][s][:,0]-T[t][c][s][:,1]).mean() for c in cs];dp=[T[t][c][s][:,0].mean() for c in cs];ct=[T[t][c][s][:,1].mean() for c in cs]
        print(f" {s:8s} reliability adj {sb(s):.2f}; rho(adj,area) {spearmanr(adj,area)[0]:+.2f}; concept-blind rho(ctl,raw) {spearmanr(ct,dp)[0]:+.2f}; rho(raw,adj) {spearmanr(dp,adj)[0]:+.2f}")
for s in SCALES:
    print(f"\n=== scale {s}: rho vs adjusted (raw in parens); pooled; boot CI on pooled adj")
    arr={}
    for m in M:
        v=[]
        for t in TASKS:
            cs=sorted(T[t]);x=np.array([M[m][(t,c)] for c in cs])
            adj=np.array([(T[t][c][s][:,0]-T[t][c][s][:,1]).mean() for c in cs]);dp=np.array([T[t][c][s][:,0].mean() for c in cs])
            arr[(m,t)]=(x,adj);v+=[spearmanr(x,adj)[0],spearmanr(x,dp)[0]]
        arr[m]=v
    boot=collections.defaultdict(list)
    for _ in range(4000):
        per=collections.defaultdict(list)
        for t in TASKS:
            n=len(arr[("CARDS",t)][0]);i=rng.integers(0,n,n)
            for m in M:
                x,a=arr[(m,t)];per[m].append(spearmanr(x[i],a[i])[0])
        for m in M: boot[m].append(np.mean(per[m]))
    for m in M:
        v=arr[m];b=np.array(boot[m]);d=np.array(boot["CARDS"])-b
        print(f" {m:32s} A {v[0]:+.2f} ({v[1]:+.2f}) M {v[2]:+.2f} ({v[3]:+.2f}) pool {(v[0]+v[2])/2:+.2f} [{np.nanpercentile(b,2.5):+.2f},{np.nanpercentile(b,97.5):+.2f}]  CARDS minus this [{np.nanpercentile(d,2.5):+.2f},{np.nanpercentile(d,97.5):+.2f}]")
