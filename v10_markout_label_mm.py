#!/usr/bin/env python3
import argparse,gzip,json,math,bisect
from collections import defaultdict,deque

NOTIONAL=5.0
HORIZONS=(100,500,1000,5000)
WEIGHTS={100:0.25,500:0.25,1000:0.30,5000:0.20}
HARD_CAP=15.0


def load(path):
    out=[]
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            x=json.loads(line)
            if x.get('type')=='EVENT': out.append(x)
    return out


def build_candidates(events):
    book={}; hist=defaultdict(lambda:deque(maxlen=60)); flow=defaultdict(lambda:deque(maxlen=80)); mids=defaultdict(list)
    inv=defaultdict(float); cash=defaultdict(float); rows=[]; maxinv=0.0
    for e in events:
        d=e['data']; s=d.get('s'); t=e['recv_ts_ms']
        if not s: continue
        if all(k in d for k in ('b','a','B','A')):
            b=float(d['b']); a=float(d['a']); B=float(d['B']); A=float(d['A']); mid=(a+b)/2
            book[s]=(b,a,B,A,mid,t); hist[s].append((t,mid)); mids[s].append((t,mid)); continue
        if d.get('e')!='trade' or s not in book: continue
        b,a,B,A,mid,_=book[s]; px=float(d['p']); qty=float(d.get('q',0)); bm=bool(d.get('m'))
        signed=(-qty if bm else qty); flow[s].append(signed)
        h=hist[s]; mom=0.0
        if len(h)>1 and h[0][1]>0: mom=(mid/h[0][1]-1)*10000
        denom=B+A; obi=(B-A)/denom if denom>0 else 0.0
        fsum=sum(flow[s]); fabs=sum(abs(x) for x in flow[s]); tfi=fsum/fabs if fabs>0 else 0.0
        spread=(a/b-1)*10000 if b>0 else 0.0
        side='BUY' if bm else 'SELL'; direction=1 if side=='SELL' else -1
        side_mom=direction*mom; side_tfi=direction*tfi; side_obi=direction*(-obi)
        fpx=b if side=='BUY' else a
        # candidate maker fill at BBO, subject only to inventory cap
        if side=='BUY':
            if inv[s]*mid+NOTIONAL > HARD_CAP+0.01: continue
        else:
            if inv[s]*mid-NOTIONAL < -HARD_CAP-0.01: continue
        q=NOTIONAL/fpx
        if side=='BUY': inv[s]+=q; cash[s]-=NOTIONAL
        else: inv[s]-=q; cash[s]+=NOTIONAL
        maxinv=max(maxinv,abs(inv[s]*mid))
        rows.append({'s':s,'t':t,'side':side,'px':fpx,'mid':mid,'mom':side_mom,'tfi':side_tfi,'obi':side_obi,'spread':spread})
    # post-fill signed markouts and composite label
    for r in rows:
        arr=mids[r['s']]; ts=[x[0] for x in arr]; marks={}
        comp=0.0; wsum=0.0
        for hz in HORIZONS:
            i=bisect.bisect_left(ts,r['t']+hz)
            if i<len(arr):
                future=arr[i][1]; raw=(future/r['px']-1)*10000; signed=raw if r['side']=='BUY' else -raw
                marks[hz]=signed; comp += WEIGHTS[hz]*signed; wsum += WEIGHTS[hz]
        r['marks']=marks; r['label']=comp/wsum if wsum>0 else None
    last={s:v[4] for s,v in book.items()}
    mtm=sum(cash[s]+inv[s]*last.get(s,0) for s in set(cash)|set(inv))
    return rows,mtm,maxinv


def quantiles(vals):
    if not vals: return [0,0,0,0]
    v=sorted(vals)
    def q(p): return v[min(len(v)-1,max(0,int(p*(len(v)-1))))]
    return [q(.2),q(.4),q(.6),q(.8)]


def bindex(x,cuts):
    return bisect.bisect_right(cuts,x)


def train_rules(rows):
    labeled=[r for r in rows if r['label'] is not None]
    cuts={k:quantiles([r[k] for r in labeled]) for k in ('mom','tfi','obi','spread')}
    stats=defaultdict(lambda:[0.0,0])
    for r in labeled:
        key=(bindex(r['mom'],cuts['mom']),bindex(r['tfi'],cuts['tfi']),bindex(r['obi'],cuts['obi']),bindex(r['spread'],cuts['spread']))
        stats[key][0]+=r['label']; stats[key][1]+=1
    table={k:(s/n,n) for k,(s,n) in stats.items() if n>=20}
    return cuts,table


def apply_rules(events,cuts,table,threshold,min_volume_frac=0.70):
    book={}; hist=defaultdict(lambda:deque(maxlen=60)); flow=defaultdict(lambda:deque(maxlen=80)); inv=defaultdict(float); cash=defaultdict(float); fills=[]; maxinv=0.; baseline=0
    for e in events:
        d=e['data']; s=d.get('s'); t=e['recv_ts_ms']
        if not s: continue
        if all(k in d for k in ('b','a','B','A')):
            b=float(d['b']); a=float(d['a']); B=float(d['B']); A=float(d['A']); mid=(a+b)/2
            book[s]=(b,a,B,A,mid); hist[s].append((t,mid)); continue
        if d.get('e')!='trade' or s not in book: continue
        b,a,B,A,mid=book[s]; px=float(d['p']); qty=float(d.get('q',0)); bm=bool(d.get('m')); flow[s].append(-qty if bm else qty)
        side='BUY' if bm else 'SELL'; direction=1 if side=='SELL' else -1
        h=hist[s]; mom=0.0
        if len(h)>1 and h[0][1]>0: mom=(mid/h[0][1]-1)*10000
        denom=B+A; obi=(B-A)/denom if denom>0 else 0.0
        fsum=sum(flow[s]); fabs=sum(abs(x) for x in flow[s]); tfi=fsum/fabs if fabs>0 else 0.0
        spread=(a/b-1)*10000 if b>0 else 0.0
        feat={'mom':direction*mom,'tfi':direction*tfi,'obi':direction*(-obi),'spread':spread}
        # baseline eligible fill at BBO
        fpx=b if side=='BUY' else a
        if side=='BUY': eligible=inv[s]*mid+NOTIONAL <= HARD_CAP+0.01
        else: eligible=inv[s]*mid-NOTIONAL >= -HARD_CAP-0.01
        if not eligible: continue
        baseline+=1
        key=(bindex(feat['mom'],cuts['mom']),bindex(feat['tfi'],cuts['tfi']),bindex(feat['obi'],cuts['obi']),bindex(feat['spread'],cuts['spread']))
        pred=table.get(key,(0.0,0))[0]
        # reject only bins with sufficiently negative expected markout
        if pred < threshold: continue
        q=NOTIONAL/fpx
        if side=='BUY': inv[s]+=q; cash[s]-=NOTIONAL
        else: inv[s]-=q; cash[s]+=NOTIONAL
        maxinv=max(maxinv,abs(inv[s]*mid)); fills.append((s,t,side,fpx,pred))
    last={s:v[4] for s,v in book.items()}; mtm=sum(cash[s]+inv[s]*last.get(s,0) for s in set(cash)|set(inv)); notional=len(fills)*NOTIONAL
    retained=(len(fills)/baseline) if baseline else 0.0; loss=max(0.0,-mtm); cpm=loss/notional*1e6 if notional else math.inf
    return {'threshold_bps':threshold,'fills':len(fills),'baseline_fills':baseline,'volume_retained':retained,'maker_notional':notional,'mtm_zero_fee':mtm,'loss_per_1m_volume':cpm,'max_inventory_usdc':maxinv,'inventory_cap_pass':maxinv<=HARD_CAP+0.01,'meets_volume_floor':retained>=min_volume_frac}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('--out',default='v10_markout_label_summary.json'); a=ap.parse_args(); ev=load(a.input)
    split=int(len(ev)*0.5); train_ev=ev[:split]; test_ev=ev[split:]
    train_rows,train_mtm,train_max=build_candidates(train_ev); cuts,table=train_rules(train_rows)
    thresholds=[-1.0,-0.75,-0.5,-0.35,-0.25,-0.15,0.0]
    results=[apply_rules(test_ev,cuts,table,x) for x in thresholds]
    eligible=[r for r in results if r['meets_volume_floor'] and r['maker_notional']>0]
    best=min(eligible,key=lambda r:r['loss_per_1m_volume']) if eligible else min(results,key=lambda r:r['loss_per_1m_volume'])
    labels=[r['label'] for r in train_rows if r['label'] is not None]
    positive=sum(1 for x in labels if x>=0); negative=sum(1 for x in labels if x<0)
    out={'objective':'learn toxic fill bins from post-fill markout labels, then apply only to future OOS','train_candidate_fills':len(train_rows),'train_baseline_mtm':train_mtm,'train_label_mean_bps':sum(labels)/len(labels) if labels else None,'train_positive_label_frac':positive/len(labels) if labels else None,'train_negative_label_frac':negative/len(labels) if labels else None,'learned_bins':len(table),'test_profiles':results,'selected':best,'pass':bool(best['inventory_cap_pass'] and best['meets_volume_floor'] and best['loss_per_1m_volume']<=10.0)}
    with open(a.out,'w') as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
