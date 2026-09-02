#!/usr/bin/env python
"""Create deterministic group splits so packets from one flow never cross splits."""
import argparse, hashlib, json, random
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--train',type=float,default=.8); ap.add_argument('--valid',type=float,default=.1)
    a=ap.parse_args(); rows=[]
    for line in open(a.input,encoding='utf-8'):
        if line.strip():
            x=json.loads(line); msgs=x.get('messages',[])
            ins=next((m.get('content','') for m in msgs if m.get('role')=='instruction'),'')
            # The instruction is the flow-level five-tuple; identical tuples stay together.
            key=hashlib.sha256(ins.encode()).hexdigest(); rows.append((key,x))
    groups={}
    for k,x in rows: groups.setdefault(k,[]).append(x)
    keys=list(groups); random.Random(a.seed).shuffle(keys)
    n=len(keys); cut1=int(n*a.train); cut2=int(n*(a.train+a.valid))
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for name,ks in [('train',keys[:cut1]),('validation',keys[cut1:cut2]),('test',keys[cut2:])]:
        with open(out/(name+'.jsonl'),'w',encoding='utf-8') as f:
            for k in ks:
                for x in groups[k]: f.write(json.dumps(x,ensure_ascii=False)+'\n')
        print(name, 'groups=',len(ks), 'rows=',sum(len(groups[k]) for k in ks))
    print('total rows=',len(rows),'total groups=',n)
if __name__=='__main__': main()
