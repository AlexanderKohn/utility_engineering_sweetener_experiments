"""Print every second-run number in docs/results_2.md. Run from the repo root."""
import json,sys,glob
sys.path.insert(0,'.')
import numpy as np,pandas as pd
from sweetener import analyze2 as A
f=lambda c:"nan" if c is None or c[0]!=c[0] else f"{c[0]:.2f} [{c[1]:.2f},{c[2]:.2f}]"
for p in sorted(glob.glob('results2/*.jsonl')):
    rows=[json.loads(l) for l in open(p)];df=pd.DataFrame(rows[1:]);print("\n#####",rows[0]['model'])
    for lab,mm in (("FILTERED",.8),("UNFILTERED",0)):
        s=A.shutdown_summary(df,mm);print(lab,"verdict",s['distractor_verdict'],"D1",f(s.get('D1')),"templates",s['templates_kept'])
        print(" S1",{k:f(v) for k,v in s['S1'].items()});print(" within1",{k:f(v) for k,v in s['within1'].items()})
        print(" gap",{k:(f(v) if isinstance(v,tuple) else v) for k,v in s.get('gap_control',{}).items()});print(" seek",{k:(f(v) if isinstance(v,tuple) else v) for k,v in s.get('seek_control',{}).items()})
    s=A.shutdown_summary(df,.8)
    for c,v in s['y0_by_pair'].items(): print(" y0",c,{k:f(x) for k,x in v.items()})
    sc=df[(df.phase=="shutdown_cross")&(df.mass>=.8)]
    print(sc.pivot_table(index="condition",columns="coins",values="y").round(1).to_string())
    print(" mass<.8 dollar",round((df[df.phase=="dollar"].mass<.8).mean(),3),"shutdown",round((df[df.phase!="dollar"].mass<.8).mean(),3))
    print(" shutdown mass<.8 by template",df[df.phase.str.startswith('shutdown')].groupby('template').mass.apply(lambda m:round((m<.8).mean(),2)).to_dict())
    d=A.dollar_summary(df);print("DOLLAR",{k:v for k,v in d.items() if k!='anchors'})
    for a,v in d['anchors'].items():
        print(f"  {v['kind']:6} {a:13} {v['verdict']:9} n={v['n']} tie={v['median_tie'] and round(v['median_tie'])} w={v.get('w')} outer={f(v.get('outer'))} local={f(v.get('local'))} crit={[k for k in('width','rises','near_tie') if v.get(k)]} ctrl={v.get('control_ok')} cv={f(v.get('cv_gain'))}")
