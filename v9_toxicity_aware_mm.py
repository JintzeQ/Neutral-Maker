#!/usr/bin/env python3
import argparse,gzip,json,math,bisect
from collections import defaultdict,deque
NOTIONAL=5.0
PROFILES=[
 {"name":"V9_VOLUME","thr":1.8,"widen":0.5,"hard_cap":15.0},
 {"name":"V9_BALANCED","thr":1.1,"widen":1.0,"hard_cap":15.0},
 {"name":"V9_DEFENSIVE","thr":0.7,"widen":1.5,"hard_cap":15.0},
]

def load(path):
 out=[]
 with gzip.open(path,'rt',encoding='utf-8') as f:
  for line in f:
   x=json.loads(line)
   if x.get('type')=='EVENT': out.append(x)
 return out

def replay(events,p):
 book={}; mids=defaultdict(list); hist=defaultdict(lambda:deque(maxlen=60)); flow=defaultdict(lambda:deque(maxlen=80)); inv=defaultdict(float); cash=defaultdict(float); fills=[]; maxinv=0.; rejects=0
 for e in events:
  d=e['data']; s=d.get('s'); t=e['recv_ts_ms']
  if not s: continue
  if all(k in d for k in ('b','a','B','A')):
   b=float(d['b']); a=float(d['a']); B=float(d['B']); A=float(d['A']); mid=(a+b)/2
   book[s]=(b,a,B,A,mid); hist[s].append((t,mid)); mids[s].append((t,mid)); continue
  if d.get('e')!='trade' or s not in book: continue
  b,a,B,A,mid=book[s]; px=float(d['p']); qty=float(d.get('q',0)); buyer_maker=bool(d.get('m'))
  signed=(-qty if buyer_maker else qty); flow[s].append(signed)
  h=hist[s]; mom=0.
  if len(h)>1 and h[0][1]>0: mom=(mid/h[0][1]-1)*10000
  denom=B+A; obi=(B-A)/denom if denom>0 else 0.
  fsum=sum(flow[s]); fabs=sum(abs(x) for x in flow[s]); tfi=fsum/fabs if fabs>0 else 0.
  # Toxicity is side-specific: positive score means expected adverse move after fill.
  side='BUY' if buyer_maker else 'SELL'
  direction=1 if side=='SELL' else -1
  toxic=max(0., direction*mom*0.45 + direction*tfi*1.2 + direction*(-obi)*0.6)
  widen=p['widen'] if toxic>p['thr'] else 0.
  qbid=b*(1-widen/10000); qask=a*(1+widen/10000); fill=False; fpx=0.
  if side=='BUY' and px<=qbid:
   if inv[s]*mid+NOTIONAL <= p['hard_cap']+0.01: fill=True; fpx=qbid
   else: rejects+=1
  elif side=='SELL' and px>=qask:
   if inv[s]*mid-NOTIONAL >= -p['hard_cap']-0.01: fill=True; fpx=qask
   else: rejects+=1
  if not fill: continue
  q=NOTIONAL/fpx
  if side=='BUY': inv[s]+=q; cash[s]-=NOTIONAL
  else: inv[s]-=q; cash[s]+=NOTIONAL
  maxinv=max(maxinv,abs(inv[s]*mid)); fills.append({'s':s,'t':t,'side':side,'px':fpx,'toxic':toxic})
 last={s:v[4] for s,v in book.items()}; mtm=sum(cash[s]+inv[s]*last.get(s,0) for s in set(cash)|set(inv)); notional=len(fills)*NOTIONAL
 # post-fill markouts from captured mids
 marks={100:[],500:[],1000:[],5000:[]}
 for f in fills:
  arr=mids[f['s']]; ts=[x[0] for x in arr]
  for hz in marks:
   i=bisect.bisect_left(ts,f['t']+hz)
   if i<len(arr):
    future=arr[i][1]; raw=(future/f['px']-1)*10000; signed=raw if f['side']=='BUY' else -raw; marks[hz].append(signed)
 avg={str(k): (sum(v)/len(v) if v else None) for k,v in marks.items()}
 loss=max(0.,-mtm); cpm=loss/notional*1e6 if notional else math.inf
 return {'profile':p['name'],'fills':len(fills),'maker_notional':notional,'mtm_zero_fee':mtm,'loss_per_1m_volume':cpm,'max_inventory_usdc':maxinv,'hard_cap':p['hard_cap'],'inventory_cap_pass':maxinv<=p['hard_cap']+0.01,'cap_rejections':rejects,'avg_signed_markout_bps':avg}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('--out',default='v9_summary.json'); a=ap.parse_args(); ev=load(a.input); split=int(len(ev)*.6); val=ev[:split]; oos=ev[split:]
 vr=[replay(val,p) for p in PROFILES]
 # Require at least 80% of maximum validation volume, then minimize loss/$1m.
 vmax=max(x['maker_notional'] for x in vr); eligible=[x for x in vr if x['maker_notional']>=.8*vmax]; best=min(eligible,key=lambda x:x['loss_per_1m_volume']); pp=next(p for p in PROFILES if p['name']==best['profile']); oo=replay(oos,pp)
 out={'objective':'minimize adverse-selection cost while retaining >=80% validation volume','validation':vr,'selected':best['profile'],'oos':oo,'pass':bool(oo['inventory_cap_pass'] and oo['maker_notional']>0 and oo['loss_per_1m_volume']<=10.0)}
 with open(a.out,'w') as f: json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()
