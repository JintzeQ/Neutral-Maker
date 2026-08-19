#!/usr/bin/env python3
import argparse,bisect,gzip,json,math
from collections import defaultdict,deque
NOTIONAL=5.0; HARD_CAP=15.0

def load_events(path):
 out=[]
 with gzip.open(path,'rt',encoding='utf-8') as f:
  for line in f:
   x=json.loads(line)
   if x.get('type')=='EVENT': out.append(x)
 return out

def bidx(x,c): return bisect.bisect_right(c,x)

def replay(events,model):
 cuts=model['cuts']; table=model['table']; thr=model['threshold_bps']
 book={}; hist=defaultdict(lambda:deque(maxlen=60)); flow=defaultdict(lambda:deque(maxlen=80)); inv=defaultdict(float); cash=defaultdict(float)
 fills=baseline=rejects=0; maxinv=0.0
 for e in events:
  d=e['data']; s=d.get('s')
  if not s: continue
  if all(k in d for k in ('b','a','B','A')):
   b=float(d['b']); a=float(d['a']); B=float(d['B']); A=float(d['A']); mid=(a+b)/2
   book[s]=(b,a,B,A,mid); hist[s].append((e['recv_ts_ms'],mid)); continue
  if d.get('e')!='trade' or s not in book: continue
  b,a,B,A,mid=book[s]; qty=float(d.get('q',0)); bm=bool(d.get('m')); flow[s].append(-qty if bm else qty)
  side='BUY' if bm else 'SELL'; direction=1 if side=='SELL' else -1
  h=hist[s]; mom=(mid/h[0][1]-1)*10000 if len(h)>1 and h[0][1]>0 else 0.0
  denom=B+A; obi=(B-A)/denom if denom>0 else 0.0
  fsum=sum(flow[s]); fabs=sum(abs(x) for x in flow[s]); tfi=fsum/fabs if fabs>0 else 0.0
  spread=(a/b-1)*10000 if b>0 else 0.0
  fpx=b if side=='BUY' else a
  eligible=(inv[s]*mid+NOTIONAL <= HARD_CAP+0.01) if side=='BUY' else (inv[s]*mid-NOTIONAL >= -HARD_CAP-0.01)
  if not eligible: continue
  baseline+=1
  feat=(direction*mom,direction*tfi,direction*(-obi),spread)
  key=','.join(map(str,(bidx(feat[0],cuts['mom']),bidx(feat[1],cuts['tfi']),bidx(feat[2],cuts['obi']),bidx(feat[3],cuts['spread']))))
  pred=table.get(key,[0.0,0])[0]
  if pred < thr:
   rejects+=1; continue
  q=NOTIONAL/fpx
  if side=='BUY': inv[s]+=q; cash[s]-=NOTIONAL
  else: inv[s]-=q; cash[s]+=NOTIONAL
  maxinv=max(maxinv,abs(inv[s]*mid)); fills+=1
 last={s:v[4] for s,v in book.items()}; mtm=sum(cash[s]+inv[s]*last.get(s,0) for s in set(cash)|set(inv)); notional=fills*NOTIONAL
 loss=max(0.0,-mtm); cpm=loss/notional*1e6 if notional else math.inf; retained=fills/baseline if baseline else 0.0
 return {'fills':fills,'baseline_fills':baseline,'volume_retained':retained,'maker_notional':notional,'mtm_zero_fee':mtm,'loss_per_1m_volume':cpm,'max_inventory_usdc':maxinv,'inventory_cap_pass':maxinv<=HARD_CAP+0.01,'filter_rejections':rejects,'threshold_bps':thr}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('--model',default='v10_frozen_model.json'); ap.add_argument('--out',default='v10_frozen_forward_summary.json'); a=ap.parse_args()
 with open(a.model) as f: model=json.load(f)
 out=replay(load_events(a.input),model)
 out['source_model_run']=model.get('source_run'); out['pass']=bool(out['inventory_cap_pass'] and out['volume_retained']>=0.70 and out['loss_per_1m_volume']<=10.0)
 with open(a.out,'w') as f: json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()
