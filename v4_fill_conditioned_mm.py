#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import v2_datavision_validation as dv
import v2_maker_proxy as mp

LATENCY_MS=75
QUOTE_USDC=5.0
HALF_SPREAD_BPS=0.75
LEAD_SHIFT=0.50
QUEUE_MULT=3.0
FRICTION_BPS=0.20
INV_CAP=20.0
PAPER_CAPITAL=1000.0
FEATURES=['abs_score','lead_flow_aligned','lead_mom2_aligned','exec_flow_aligned','exec_mom1_aligned','exec_mom2_aligned','exec_mom5_aligned','mom_gap_aligned','basis_dev_aligned','exec_vol60_bps','tick_bps','log_exec_notional5','hour_sin','hour_cos']


def add_features(frame, lead_raw, exec_raw):
    f=frame.copy(); ep=f['exec_px'].astype(float); lp=f['lead_px'].astype(float)
    lr=np.log(ep/ep.shift(1))*1e4
    f['exec_mom1_bps']=lr; f['exec_mom2_bps']=np.log(ep/ep.shift(2))*1e4; f['exec_mom5_bps']=np.log(ep/ep.shift(5))*1e4
    f['exec_vol60_bps']=lr.rolling(60,min_periods=20).std(ddof=0)
    ea=dv.aggregate_lead(exec_raw); idx=f.index
    signed=ea['signed'].reindex(idx).fillna(0.0); absn=ea['absn'].reindex(idx).fillna(0.0)
    f['exec_flow2']=signed.rolling(2,min_periods=1).sum()/absn.rolling(2,min_periods=1).sum().replace(0,np.nan)
    f['log_exec_notional5']=np.log1p(absn.rolling(5,min_periods=1).sum())
    basis=np.log(ep/lp); basis_ema=basis.ewm(halflife=120,min_periods=30,adjust=False).mean(); f['basis_dev_bps']=(basis-basis_ema)*1e4
    return f.replace([np.inf,-np.inf],np.nan)


def generate_shadow_fills(symbol, frame, exec_raw, start_sec, end_sec, min_score=0.20):
    f=frame.loc[(frame.index>=start_sec)&(frame.index<=end_sec)].copy()
    f=f[(f.score.abs()>=min_score)&(f.lead_age<=2)&(f.exec_age<=2)]
    need=['score','exec_px','flow2','mom2_bps','exec_flow2','exec_mom1_bps','exec_mom2_bps','exec_mom5_bps','exec_vol60_bps','basis_dev_bps','log_exec_notional5']
    f=f.dropna(subset=need)
    raw=exec_raw[(exec_raw.ts>=start_sec*1000)&(exec_raw.ts<=(end_sec+35)*1000)]
    if f.empty or raw.empty: return pd.DataFrame()
    ts=raw.ts.to_numpy(np.int64); px=raw.price.to_numpy(float); qty=raw.qty.to_numpy(float); bm=raw.is_buyer_maker.to_numpy(bool); notion=px*qty
    tick=mp.infer_tick(px)
    if not np.isfinite(tick) or tick<=0: return pd.DataFrame()
    rows=[]
    for sec,row in f.iterrows():
        sec=int(sec); score=float(row.score); side=1 if score>0 else -1; p0=float(row.exec_px)
        qpx=p0*math.exp((LEAD_SHIFT*score-side*HALF_SPREAD_BPS)/1e4); qpx=min(qpx,p0-tick) if side>0 else max(qpx,p0+tick)
        if qpx<=0: continue
        active=(sec+1)*1000+LATENCY_MS; expire=active+1000
        i0=int(np.searchsorted(ts,active,'left')); i1=int(np.searchsorted(ts,expire,'left'))
        if i1<=i0: continue
        sl=slice(i0,i1); qual=(bm[sl]&(px[sl]<=qpx-tick+tick*1e-6)) if side>0 else ((~bm[sl])&(px[sl]>=qpx+tick-tick*1e-6))
        loc=np.flatnonzero(qual)
        if len(loc)==0: continue
        c=np.cumsum(notion[sl][loc]); k=int(np.searchsorted(c,QUOTE_USDC*(1+QUEUE_MULT),'left'))
        if k>=len(loc): continue
        j=i0+int(loc[k]); fill_ms=int(ts[j]); fill_px=float(qpx); marks={}
        for h in (1,5,30):
            fp=mp.last_price_at(ts,px,fill_ms+h*1000,2000); marks[h]=None if fp is None else side*math.log(fp/fill_px)*1e4
        if marks[5] is None: continue
        hour=(sec%86400)/3600.0
        rows.append({'symbol':symbol,'sec':sec,'fill_ms':fill_ms,'bucket5m':fill_ms//300000,'side':side,'fill_px':fill_px,
                     'mark1_bps':marks[1],'mark5_bps':marks[5],'mark30_bps':marks[30],'net5_bps':marks[5]-FRICTION_BPS,
                     'abs_score':abs(score),'lead_flow_aligned':side*float(row.flow2),'lead_mom2_aligned':side*float(row.mom2_bps),
                     'exec_flow_aligned':side*float(row.exec_flow2),'exec_mom1_aligned':side*float(row.exec_mom1_bps),
                     'exec_mom2_aligned':side*float(row.exec_mom2_bps),'exec_mom5_aligned':side*float(row.exec_mom5_bps),
                     'mom_gap_aligned':side*(float(row.mom2_bps)-float(row.exec_mom2_bps)),
                     'basis_dev_aligned':side*float(row.basis_dev_bps),'exec_vol60_bps':float(row.exec_vol60_bps),
                     'tick_bps':tick/p0*1e4,'log_exec_notional5':float(row.log_exec_notional5),
                     'hour_sin':math.sin(2*math.pi*hour/24),'hour_cos':math.cos(2*math.pi*hour/24)})
    return pd.DataFrame(rows)


def bootstrap(df,n=2000,seed=1):
    x=df.dropna(subset=['net5_bps'])
    if x.empty:return {'mean':None,'lcb90':None,'blocks':0,'fills':0}
    b=x.groupby('bucket5m').net5_bps.mean().to_numpy(float); rng=np.random.default_rng(seed); vals=np.empty(n)
    for i in range(n): vals[i]=rng.choice(b,size=len(b),replace=True).mean()
    return {'mean':float(x.net5_bps.mean()),'lcb90':float(np.quantile(vals,.05)),'blocks':len(b),'fills':len(x)}


def apply_inventory(df,cap=INV_CAP):
    if df.empty:return df.copy(), {'roundtrip_net_bps':None,'max_inventory_usdc':0.0,'matched_notional':0.0}
    kept=[]; lots={}; pos={}; maxinv=0.; realized=0.; matched=0.
    for r in df.sort_values('fill_ms').itertuples(index=False):
        sym=r.symbol; side=int(r.side); p=float(r.fill_px); q=QUOTE_USDC/p; inv=pos.get(sym,0.0)*p
        if side>0 and inv+QUOTE_USDC>cap: continue
        if side<0 and inv-QUOTE_USDC<-cap: continue
        dq=side*q; pos[sym]=pos.get(sym,0.0)+dq; maxinv=max(maxinv,abs(pos[sym]*p)); dqleft=abs(dq); dq_list=lots.setdefault(sym,deque())
        while dqleft>1e-15 and dq_list and np.sign(dq_list[0][0])!=side:
            lq,lp=dq_list[0]; m=min(dqleft,abs(lq))
            if lq>0 and side<0: realized+=(p-lp)*m; matched+=lp*m
            elif lq<0 and side>0: realized+=(lp-p)*m; matched+=lp*m
            lq-=math.copysign(m,lq); dqleft-=m
            if abs(lq)<1e-15:dq_list.popleft()
            else:dq_list[0]=(lq,lp)
        if dqleft>1e-15:dq_list.append((side*dqleft,p))
        kept.append(r)
    out=pd.DataFrame(kept,columns=df.columns) if kept else df.iloc[:0].copy()
    rt=(realized-(2*FRICTION_BPS/1e4)*matched)/matched*1e4 if matched>0 else None
    return out, {'roundtrip_net_bps':rt,'max_inventory_usdc':maxinv,'matched_notional':matched}


def metric(df,hours,seed=1):
    invdf,inv=apply_inventory(df); bs=bootstrap(invdf,3000,seed); maker=len(invdf)*QUOTE_USDC; days=max(hours/24,1e-9)
    return {'fills':len(invdf),'symbols':int(invdf.symbol.nunique()) if not invdf.empty else 0,'maker_notional_usdc':maker,
            'capital_turnover_x_day_1000u':maker/PAPER_CAPITAL/days,'mean_net5_bps':bs['mean'],'lcb90_net5_bps':bs['lcb90'],
            'bootstrap_blocks':bs['blocks'],'mark30_bps':float(invdf.mark30_bps.mean()) if not invdf.empty else None,
            'positive_net5_share':float((invdf.net5_bps>0).mean()) if not invdf.empty else None,
            'roundtrip_net_bps':inv['roundtrip_net_bps'],'max_inventory_usdc':inv['max_inventory_usdc']},invdf


def pass_gate(m):
    return bool(m['fills']>=500 and m['symbols']>=5 and m['mean_net5_bps'] is not None and m['mean_net5_bps']>=.30
                and m['lcb90_net5_bps'] is not None and m['lcb90_net5_bps']>=.10 and m['roundtrip_net_bps'] is not None
                and m['roundtrip_net_bps']>0 and m['mark30_bps'] is not None and m['mark30_bps']>=0)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start-date',required=True); ap.add_argument('--end-date',required=True)
    ap.add_argument('--output-dir',default='v4_fill_cond_out'); ap.add_argument('--min-exec-trades',type=int,default=3000); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); t0=time.time(); dates=mp.date_range(args.start_date,args.end_date)
    syms=dv.list_aggtrade_symbols(); sset=set(syms); bases=sorted({s[:-4] for s in syms if s.endswith('USDC') and s[:-4]+'USDT' in sset})
    frames={}; raws={}; skipped=[]
    for i,b in enumerate(bases,1):
        print(f'[{i}/{len(bases)}] {b}',flush=True)
        try: lead=mp.load_multi(b+'USDT',dates); exe=mp.load_multi(b+'USDC',dates)
        except Exception as e: skipped.append({'base':b,'reason':str(e)}); continue
        if len(exe)<args.min_exec_trades: continue
        fr=dv.build_frame(dv.aggregate_lead(lead),dv.aggregate_exec(exe),2)
        if len(fr)<10800: continue
        frames[b+'USDC']=add_features(fr,lead,exe); raws[b+'USDC']=exe
    common_start=max(int(f.index.min()) for f in frames.values()); common_end=min(int(f.index.max()) for f in frames.values()); span=common_end-common_start
    train=(common_start,common_start+int(span*.50)); val=(train[1]+1,common_start+int(span*.75)); test=(val[1]+1,common_end-31)
    print(f'eligible={len(frames)} train_h={(train[1]-train[0])/3600:.1f} val_h={(val[1]-val[0])/3600:.1f} test_h={(test[1]-test[0])/3600:.1f}',flush=True)
    parts={'train':[],'val':[],'test':[]}
    for sym in sorted(frames):
        print('fills',sym,flush=True)
        for name,period in [('train',train),('val',val),('test',test)]:
            x=generate_shadow_fills(sym,frames[sym],raws[sym],period[0],period[1])
            if not x.empty:parts[name].append(x)
    data={k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in parts.items()}
    for k,d in data.items(): print(k,'shadow_fills',len(d),'mean',None if d.empty else float(d.net5_bps.mean()),flush=True)
    tr=data['train'].dropna(subset=FEATURES+['net5_bps']).copy(); va=data['val'].dropna(subset=FEATURES+['net5_bps']).copy(); te=data['test'].dropna(subset=FEATURES+['net5_bps']).copy()
    model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.04,max_leaf_nodes=15,max_depth=3,min_samples_leaf=100,l2_regularization=3.0,random_state=20260818)
    model.fit(tr[FEATURES],tr.net5_bps); va['pred_net5']=model.predict(va[FEATURES]); te['pred_net5']=model.predict(te[FEATURES])
    gates=set([-1.0,-.5,0,.1,.2,.3,.4,.5,.75,1.0,1.5,2.0]); gates.update(float(x) for x in va.pred_net5.quantile([.5,.6,.7,.75,.8,.85,.9,.925,.95,.975]).values)
    vals=[]; val_h=(val[1]-val[0])/3600
    for g in sorted(gates):
        m,_=metric(va[va.pred_net5>=g],val_h,20260818); m['gate']=g; m['pass_gate']=pass_gate(m); vals.append(m)
    vdf=pd.DataFrame(vals); vdf.to_csv(out/'validation_gates.csv',index=False); passing=vdf[vdf.pass_gate==True]
    if not passing.empty: chosen=float(passing.sort_values('capital_turnover_x_day_1000u',ascending=False).iloc[0].gate); reason='max validation turnover among passing gates'
    else: chosen=float(vdf.sort_values(['lcb90_net5_bps','mean_net5_bps'],ascending=False).iloc[0].gate); reason='no validation gate passed; strongest diagnostic LCB'
    print('chosen_gate',chosen,reason,flush=True); vg=va[va.pred_net5>=chosen].copy(); srows=[]
    for sym,g in vg.groupby('symbol'):
        m,_=metric(g,val_h,20260821); m['symbol']=sym; srows.append(m)
    vs=pd.DataFrame(srows); vs.to_csv(out/'validation_symbols.csv',index=False)
    selected=set(vs[(vs.fills>=50)&(vs.mean_net5_bps>=.30)&(vs.mark30_bps>=0)].symbol.tolist()) if not vs.empty else set(); print('selected_symbols',sorted(selected),flush=True)
    tg=te[te.pred_net5>=chosen].copy(); test_h=(test[1]-test[0])/3600; all_m,all_f=metric(tg,test_h,20260822); all_m['pass_gate']=pass_gate(all_m)
    if selected: final_m,final_f=metric(tg[tg.symbol.isin(selected)],test_h,20260823); final_m['pass_gate']=pass_gate(final_m)
    else: final_m,final_f=all_m,all_f
    final_m.update({'status':'PASS_MAKER_PROXY' if final_m['pass_gate'] else 'FAIL_MAKER_PROXY','model':'HistGradientBoostingRegressor','prediction_gate_bps':chosen,
                    'selection_reason':reason,'selected_symbols':sorted(selected),'train_shadow_fills':len(tr),'validation_shadow_fills':len(va),'test_shadow_fills':len(te),
                    'oos_hours':test_h,'paper_capital_usdc':PAPER_CAPITAL,'elapsed_sec':time.time()-t0,
                    'limitations':['Fill-conditioned model uses only pre-placement features, but execution remains an aggTrades proxy, not L2 queue replay.',
                                   'Quotes use 75ms synthetic latency, 1-tick penetration, 3x synthetic queue ahead, 0 maker fee and 0.20bps friction.',
                                   'Model and gate are fit on train/validation only; final segment is untouched OOS.','Proxy PASS would require true L2/live-paper confirmation.']})
    va.to_csv(out/'validation_fill_predictions.csv',index=False); te.to_csv(out/'oos_fill_predictions.csv',index=False); final_f.to_csv(out/'oos_final_fills.csv',index=False)
    pd.DataFrame(skipped).to_csv(out/'skipped.csv',index=False); (out/'oos_final_summary.json').write_text(json.dumps(final_m,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'oos_all_summary.json').write_text(json.dumps(all_m,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n=== V4 FILL-CONDITIONED OOS ==='); print(json.dumps(final_m,ensure_ascii=False,indent=2)); print('\n=== ALL UNIVERSE OOS ==='); print(json.dumps(all_m,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
