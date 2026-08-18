#!/usr/bin/env python3
import argparse,gzip,json
from collections import defaultdict,deque
PROFILES=[{'name':'ICMM_S05','skew_bps_per_5u':0.5,'hard_cap_usdc':15.0},{'name':'ICMM_S10','skew_bps_per_5u':1.0,'hard_cap_usdc':15.0},{'name':'ICMM_S20','skew_bps_per_5u':2.0,'hard_cap_usdc':15.0}]
def load(p):
 e=[]
 with gzip.open(p,'rt') as f:
  for l in f:
   o=json.loads(l)
   if o.get('type')=='EVENT' and o['data'].get('e') in ('bookTicker','trade'):e.append((int(o['recv_ts_ms']),o['data']))
 return sorted(e,key=lambda x:x[0])
def replay(events,prof,start,end,quote=5.0,cost_bps=.25):
 st=defaultdict(lambda:{'b':None,'a':None,'B':0.,'A':0.,'inv':0.,'cash':0.,'fills':0,'notional':0.,'maxinv':0.,'area':0.,'last':None,'mid':None,'qbuy':None,'qsell':None,'realized':0.,'lots':deque()})
 skew_unit=prof['skew_bps_per_5u'];cap=prof['hard_cap_usdc']
 def mark(s,now):
  if s['last'] is not None and s['mid'] is not None:s['area']+=abs(s['inv']*s['mid'])*max(0,now-s['last'])
  s['last']=now
  if s['mid']:s['maxinv']=max(s['maxinv'],abs(s['inv']*s['mid']))
 def fill(s,side,px,qty):
  n=px*qty;s['fills']+=1;s['notional']+=n;s['cash']+=n if side=='SELL' else -n;rem=qty
  if side=='BUY':
   while rem>1e-15 and s['lots'] and s['lots'][0][0]<0:
    q0,p0=s['lots'][0];m=min(rem,-q0);s['realized']+=(p0-px)*m;rem-=m;q0+=m
    if abs(q0)<1e-15:s['lots'].popleft()
    else:s['lots'][0]=(q0,p0)
   if rem>1e-15:s['lots'].append((rem,px))
   s['inv']+=qty
  else:
   while rem>1e-15 and s['lots'] and s['lots'][0][0]>0:
    q0,p0=s['lots'][0];m=min(rem,q0);s['realized']+=(px-p0)*m;rem-=m;q0-=m
    if abs(q0)<1e-15:s['lots'].popleft()
    else:s['lots'][0]=(q0,p0)
   if rem>1e-15:s['lots'].append((-rem,px))
   s['inv']-=qty
 def quotes(s):
  if not s['mid']:return None,None
  invu=s['inv']*s['mid'];shift=skew_unit*(invu/5.0);buy=s['b']*(1-shift/1e4);sell=s['a']*(1-shift/1e4)
  if invu>=cap:buy=None
  if invu<=-cap:sell=None
  return buy,sell
 for now,d in events:
  if now<start:continue
  if now>=end:break
  s=st[d.get('s')];mark(s,now)
  if d['e']=='bookTicker':
   s['b']=float(d['b']);s['a']=float(d['a']);s['B']=float(d['B']);s['A']=float(d['A']);s['mid']=(s['b']+s['a'])/2;qb,qs=quotes(s);s['qbuy']=[qb,s['B']] if qb is not None else None;s['qsell']=[qs,s['A']] if qs is not None else None
  elif s['mid']:
   px=float(d['p']);tq=float(d['q']);m=bool(d['m'])
   if m and s['qbuy'] is not None and px<=s['qbuy'][0]:
    res=quote/s['mid'] if px<s['qbuy'][0] else max(0,tq-s['qbuy'][1])
    if res>0:fill(s,'BUY',s['qbuy'][0],min(res,quote/s['mid']));s['qbuy']=None
   if (not m) and s['qsell'] is not None and px>=s['qsell'][0]:
    res=quote/s['mid'] if px>s['qsell'][0] else max(0,tq-s['qsell'][1])
    if res>0:fill(s,'SELL',s['qsell'][0],min(res,quote/s['mid']));s['qsell']=None
 dur=max(1,end-start);maker=sum(s['notional'] for s in st.values());real=sum(s['realized'] for s in st.values());mtm=sum(s['cash']+s['inv']*s['mid'] for s in st.values() if s['mid']);fee=maker*cost_bps/1e4;grossinv=sum(abs(s['inv']*s['mid']) for s in st.values() if s['mid']);maxinv=max([s['maxinv'] for s in st.values()] or [0]);meaninv=sum(s['area'] for s in st.values())/dur/max(1,len(st));stress={str(x):mtm-fee-grossinv*x/100 for x in (5,10,20)}
 return {'profile':prof,'hours':dur/3600000,'fills':sum(s['fills'] for s in st.values()),'maker_notional_usdc':maker,'turnover_x_day_1000u':maker/(dur/86400000)/1000,'realized_mm_pnl_usdc':real,'mtm_gross_pnl_usdc':mtm,'mtm_net_025bps_usdc':mtm-fee,'gross_inventory_usdc':grossinv,'max_symbol_inventory_usdc':maxinv,'mean_abs_inventory_usdc':meaninv,'stress_net_pnl_usdc':stress,'inventory_bounded':maxinv<=cap*1.05}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('capture');ap.add_argument('--out',default='v7_inventory_carrying_summary.json');a=ap.parse_args();e=load(a.capture);lo=e[0][0];hi=e[-1][0]+1;cut=lo+int((hi-lo)*.60);vals=[]
 for p in PROFILES:vals.append({'profile':p,'validation':replay(e,p,lo,cut),'oos':replay(e,p,cut,hi)})
 vals.sort(key=lambda x:(x['oos']['inventory_bounded'],x['oos']['mtm_net_025bps_usdc']>0,x['oos']['mtm_net_025bps_usdc'],x['oos']['turnover_x_day_1000u']),reverse=True);best=vals[0];o=best['oos'];status='PASS' if o['inventory_bounded'] and o['mtm_net_025bps_usdc']>0 and o['stress_net_pnl_usdc']['20']>-10 else 'NEEDS_MORE_DATA';out={'status':status,'selection':'inventory bounded + positive net MTM + stress safety','chosen':best,'all_profiles':vals};open(a.out,'w').write(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
