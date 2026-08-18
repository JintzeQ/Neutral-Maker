#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import v2_datavision_validation as dv
import v2_maker_proxy as mp

PAPER_CAPITAL = 1000.0
LATENCY_MS = 75

@dataclass(frozen=True)
class Profile:
    name: str
    quote_usdc: float
    half_spread_bps: float
    queue_mult: float
    inv_cap_usdc: float
    inv_size_skew: float
    inv_price_shift_bps: float
    vol_widen_mult: float
    friction_bps: float = 0.10
    penetration_ticks: int = 1

PROFILES = [
    Profile('TIGHT',   5.0, 0.50, 1.0, 15.0, 0.80, 0.50, 0.00),
    Profile('CORE',    5.0, 0.75, 2.0, 20.0, 1.00, 0.75, 0.15),
    Profile('NEUTRAL', 5.0, 1.00, 3.0, 20.0, 1.25, 1.00, 0.20),
    Profile('WIDE',    7.0, 1.50, 3.0, 25.0, 1.25, 1.25, 0.30),
]


def load_multi(symbol, dates):
    parts=[]
    for d in dates:
        x,_=dv.load_day(symbol,d)
        if not x.empty: parts.append(x)
    if not parts: return pd.DataFrame()
    return pd.concat(parts,ignore_index=True).sort_values('ts',kind='stable').reset_index(drop=True)


def second_frame(raw):
    if raw.empty: return pd.DataFrame()
    x=raw.copy(); x['sec']=(x.ts//1000).astype(np.int64)
    g=x.groupby('sec').agg(last_px=('price','last'), notional=('price',lambda s:0.0))
    # rebuild notional cheaply from raw to avoid groupby lambda overhead
    n=(x.price.astype(float)*x.qty.astype(float)).groupby(x.sec).sum()
    g['notional']=n
    px=g.last_px.astype(float)
    g['ret1_bps']=np.log(px/px.shift(1))*1e4
    g['vol60_bps']=g.ret1_bps.rolling(60,min_periods=20).std(ddof=0)
    return g.replace([np.inf,-np.inf],np.nan)


def round_down_tick(p,tick): return math.floor(p/tick+1e-12)*tick

def round_up_tick(p,tick): return math.ceil(p/tick-1e-12)*tick


def fill_candidate(ts, px, qty, bm, start_ms, end_ms, side, qpx, tick, need_notional, penetration):
    i0=int(np.searchsorted(ts,start_ms,'left')); i1=int(np.searchsorted(ts,end_ms,'left'))
    if i1<=i0: return None
    sl=slice(i0,i1)
    if side>0:
        qual=bm[sl] & (px[sl] <= qpx-penetration*tick+tick*1e-6)
    else:
        qual=(~bm[sl]) & (px[sl] >= qpx+penetration*tick-tick*1e-6)
    loc=np.flatnonzero(qual)
    if len(loc)==0: return None
    notion=(px[sl]*qty[sl])[loc]
    c=np.cumsum(notion); k=int(np.searchsorted(c,need_notional,'left'))
    if k>=len(loc): return None
    j=i0+int(loc[k]); return int(ts[j])


def simulate_symbol(symbol, raw, sf, start_sec, end_sec, p:Profile):
    raw=raw[(raw.ts>=start_sec*1000)&(raw.ts<=(end_sec+2)*1000)].copy()
    f=sf.loc[(sf.index>=start_sec)&(sf.index<=end_sec)].copy()
    if raw.empty or f.empty: return {},pd.DataFrame(),pd.DataFrame()
    ts=raw.ts.to_numpy(np.int64); px=raw.price.to_numpy(float); qty=raw.qty.to_numpy(float); bm=raw.is_buyer_maker.to_numpy(bool)
    tick=mp.infer_tick(px)
    if not np.isfinite(tick) or tick<=0: return {},pd.DataFrame(),pd.DataFrame()

    lots=deque(); pos_qty=0.0; max_inv=0.0; inv_samples=[]; fills=[]; matches=[]; quote_count=0
    realized=0.0; matched_notional=0.0

    secs=f.index.to_numpy(np.int64); ref=f.last_px.to_numpy(float); vols=f.vol60_bps.fillna(0.0).to_numpy(float)
    for idx,sec in enumerate(secs):
        p0=float(ref[idx]);
        if not np.isfinite(p0) or p0<=0: continue
        inv_u=pos_qty*p0; inv_ratio=float(np.clip(inv_u/p.inv_cap_usdc,-1,1)); inv_samples.append(abs(inv_u))
        widen=max(0.0,p.vol_widen_mult*min(float(vols[idx]),8.0))
        reservation_shift=-p.inv_price_shift_bps*inv_ratio
        half=p.half_spread_bps+widen

        bid_mult=float(np.clip(1.0-p.inv_size_skew*inv_ratio,0.0,2.0))
        ask_mult=float(np.clip(1.0+p.inv_size_skew*inv_ratio,0.0,2.0))
        bid_u=p.quote_usdc*bid_mult; ask_u=p.quote_usdc*ask_mult
        if inv_u>=p.inv_cap_usdc-1e-9: bid_u=0.0
        if inv_u<=-p.inv_cap_usdc+1e-9: ask_u=0.0
        bid_u=min(bid_u,max(0.0,p.inv_cap_usdc-inv_u))
        ask_u=min(ask_u,max(0.0,p.inv_cap_usdc+inv_u))

        bid_target=p0*math.exp((reservation_shift-half)/1e4)
        ask_target=p0*math.exp((reservation_shift+half)/1e4)
        bid_px=min(round_down_tick(bid_target,tick),p0-tick)
        ask_px=max(round_up_tick(ask_target,tick),p0+tick)
        active=(int(sec)+1)*1000+LATENCY_MS; expire=active+1000

        candidates=[]
        if bid_u>=1.0 and bid_px>0:
            quote_count+=1
            t=fill_candidate(ts,px,qty,bm,active,expire,1,bid_px,tick,bid_u*(1+p.queue_mult),p.penetration_ticks)
            if t is not None: candidates.append((t,1,bid_px,bid_u))
        if ask_u>=1.0 and ask_px>0:
            quote_count+=1
            t=fill_candidate(ts,px,qty,bm,active,expire,-1,ask_px,tick,ask_u*(1+p.queue_mult),p.penetration_ticks)
            if t is not None: candidates.append((t,-1,ask_px,ask_u))

        for fill_ms,side,fill_px,quote_u in sorted(candidates,key=lambda z:z[0]):
            current_inv=pos_qty*fill_px
            if side>0 and current_inv+quote_u>p.inv_cap_usdc+1e-9: continue
            if side<0 and current_inv-quote_u<-p.inv_cap_usdc-1e-9: continue
            q=quote_u/fill_px; signed=side*q; rem=abs(signed)
            while rem>1e-15 and lots and np.sign(lots[0]['qty'])!=side:
                lot=lots[0]; m=min(rem,abs(lot['qty'])); pnl=0.0
                if lot['qty']>0 and side<0: pnl=(fill_px-lot['px'])*m
                elif lot['qty']<0 and side>0: pnl=(lot['px']-fill_px)*m
                entry_notional=lot['px']*m; realized+=pnl; matched_notional+=entry_notional
                net_pnl=pnl-(2*p.friction_bps/1e4)*entry_notional
                matches.append({'symbol':symbol,'close_ms':fill_ms,'bucket5m':fill_ms//300000,'entry_notional':entry_notional,'net_pnl':net_pnl,'rt_bps':net_pnl/entry_notional*1e4,'hold_sec':max(0,(fill_ms-lot['ts'])/1000)})
                lot['qty']-=math.copysign(m,lot['qty']); rem-=m
                if abs(lot['qty'])<1e-15: lots.popleft()
            if rem>1e-15: lots.append({'qty':side*rem,'px':fill_px,'ts':fill_ms})
            pos_qty+=signed; max_inv=max(max_inv,abs(pos_qty*fill_px))
            fills.append({'symbol':symbol,'fill_ms':fill_ms,'side':'BUY' if side>0 else 'SELL','fill_px':fill_px,'quote_usdc':quote_u,'inventory_usdc':pos_qty*fill_px})

    ff=pd.DataFrame(fills); mm=pd.DataFrame(matches)
    final_px=mp.last_price_at(ts,px,end_sec*1000,5000); unreal=0.0
    if final_px is not None:
        for lot in lots:
            unreal += (final_px-lot['px'])*lot['qty'] if lot['qty']>0 else (lot['px']-final_px)*abs(lot['qty'])
    fill_notional=float(ff.quote_usdc.sum()) if not ff.empty else 0.0
    friction=fill_notional*p.friction_bps/1e4
    mtm=realized+unreal-friction
    rt_net=float(mm.net_pnl.sum()) if not mm.empty else 0.0
    rt_bps=rt_net/matched_notional*1e4 if matched_notional>0 else None
    duration_days=max((end_sec-start_sec)/86400,1e-9)
    s={'symbol':symbol,'fills':len(ff),'quotes':quote_count,'fill_ratio':len(ff)/quote_count if quote_count else 0.0,
       'maker_notional_usdc':fill_notional,'daily_maker_notional_usdc':fill_notional/duration_days,
       'turnover_x_day_1000u':fill_notional/PAPER_CAPITAL/duration_days,'roundtrip_net_bps':rt_bps,
       'matched_notional_usdc':matched_notional,'mtm_net_pnl_usdc':mtm,'max_inventory_usdc':max_inv,
       'mean_abs_inventory_usdc':float(np.mean(inv_samples)) if inv_samples else 0.0,'ending_inventory_usdc':pos_qty*(final_px or (ref[-1] if len(ref) else 0.0)),
       'median_hold_sec':float(mm.hold_sec.median()) if not mm.empty else None,'cycles':len(mm)}
    return s,ff,mm


def bootstrap_matches(mm,n=2000,seed=1):
    if mm.empty:return {'mean_bps':None,'lcb90_bps':None,'blocks':0}
    b=mm.groupby('bucket5m').agg(pnl=('net_pnl','sum'),notional=('entry_notional','sum'))
    arr=b[['pnl','notional']].to_numpy(float); rng=np.random.default_rng(seed); vals=[]
    for _ in range(n):
        z=arr[rng.integers(0,len(arr),len(arr))]; den=z[:,1].sum(); vals.append(z[:,0].sum()/den*1e4 if den>0 else np.nan)
    vals=np.asarray(vals); return {'mean_bps':float(mm.net_pnl.sum()/mm.entry_notional.sum()*1e4),'lcb90_bps':float(np.nanquantile(vals,.05)),'blocks':len(arr)}


def pooled(raws,frames,period,p,allow=None,seed=1):
    rows=[]; fills=[]; matches=[]
    for sym in sorted(raws):
        if allow is not None and sym not in allow: continue
        s,f,m=simulate_symbol(sym,raws[sym],frames[sym],period[0],period[1],p)
        if s: rows.append(s)
        if not f.empty:fills.append(f)
        if not m.empty:matches.append(m)
    rdf=pd.DataFrame(rows); fdf=pd.concat(fills,ignore_index=True) if fills else pd.DataFrame(); mdf=pd.concat(matches,ignore_index=True) if matches else pd.DataFrame()
    boot=bootstrap_matches(mdf,2000,seed); days=max((period[1]-period[0])/86400,1e-9); maker=float(rdf.maker_notional_usdc.sum()) if not rdf.empty else 0.0
    out={'profile':p.name,**asdict(p),'symbols':int((rdf.fills>0).sum()) if not rdf.empty else 0,'fills':len(fdf),'cycles':len(mdf),
         'maker_notional_usdc':maker,'turnover_x_day_1000u':maker/PAPER_CAPITAL/days,'roundtrip_net_bps':boot['mean_bps'],'lcb90_roundtrip_bps':boot['lcb90_bps'],
         'bootstrap_blocks':boot['blocks'],'mtm_net_pnl_usdc':float(rdf.mtm_net_pnl_usdc.sum()) if not rdf.empty else 0.0,
         'max_symbol_inventory_usdc':float(rdf.max_inventory_usdc.max()) if not rdf.empty else 0.0,'mean_abs_inventory_usdc':float(rdf.mean_abs_inventory_usdc.mean()) if not rdf.empty else 0.0}
    return out,rdf,fdf,mdf


def pass_gate(r):
    return bool(r['fills']>=1000 and r['cycles']>=500 and r['symbols']>=5 and r['roundtrip_net_bps'] is not None and r['roundtrip_net_bps']>=0.20 and r['lcb90_roundtrip_bps'] is not None and r['lcb90_roundtrip_bps']>=0.05 and r['mtm_net_pnl_usdc']>0)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start-date',required=True); ap.add_argument('--end-date',required=True); ap.add_argument('--output-dir',default='v5_pure_mm_out'); ap.add_argument('--min-trades',type=int,default=5000); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); dates=mp.date_range(args.start_date,args.end_date); t0=time.time()
    syms=sorted(s for s in dv.list_aggtrade_symbols() if s.endswith('USDC')); raws={}; frames={}; skipped=[]
    for i,sym in enumerate(syms,1):
        print(f'[{i}/{len(syms)}] {sym}',flush=True)
        try:r=load_multi(sym,dates)
        except Exception as e: skipped.append({'symbol':sym,'reason':str(e)}); continue
        if len(r)<args.min_trades: skipped.append({'symbol':sym,'reason':f'trades {len(r)} < {args.min_trades}'}); continue
        sf=second_frame(r)
        if len(sf)<10800: skipped.append({'symbol':sym,'reason':'insufficient seconds'}); continue
        raws[sym]=r; frames[sym]=sf
    if not raws: raise SystemExit('no eligible USDC markets')
    common_start=max(int(f.index.min()) for f in frames.values()); common_end=min(int(f.index.max()) for f in frames.values()); span=common_end-common_start
    val=(common_start,common_start+int(span*.50)); test=(val[1]+1,common_end-2)
    print(f'eligible={len(raws)} val_h={(val[1]-val[0])/3600:.2f} test_h={(test[1]-test[0])/3600:.2f}',flush=True)
    vals=[]; per={}
    for p in PROFILES:
        print('VALIDATE',p.name,flush=True); r,ps,_,_=pooled(raws,frames,val,p,seed=20260818); r['pass_gate']=pass_gate(r); vals.append(r); per[p.name]=ps; print(json.dumps(r,ensure_ascii=False),flush=True)
    vdf=pd.DataFrame(vals); vdf.to_csv(out/'validation_profiles.csv',index=False); passing=vdf[vdf.pass_gate==True]
    if not passing.empty:
        chosen_name=str(passing.sort_values('turnover_x_day_1000u',ascending=False).iloc[0].profile); reason='max validation turnover among positive-EV inventory-neutral profiles'
    else:
        chosen_name=str(vdf.sort_values(['lcb90_roundtrip_bps','roundtrip_net_bps','turnover_x_day_1000u'],ascending=False).iloc[0].profile); reason='no profile passed; strongest validation roundtrip statistics'
    p=next(x for x in PROFILES if x.name==chosen_name); vps=per[chosen_name].copy(); selected=set()
    if not vps.empty:
        selected=set(vps[(vps.cycles>=25)&(vps.roundtrip_net_bps>0)&(vps.mtm_net_pnl_usdc>0)].symbol.tolist())
    print('CHOSEN',chosen_name,reason,'selected',sorted(selected),flush=True)
    all_r,all_ps,all_f,all_m=pooled(raws,frames,test,p,seed=20260819); all_r['pass_gate']=pass_gate(all_r)
    if selected:
        final,ps,ff,mm=pooled(raws,frames,test,p,allow=selected,seed=20260820); final['pass_gate']=pass_gate(final)
    else: final,ps,ff,mm=all_r,all_ps,all_f,all_m
    final.update({'status':'PASS_PURE_MM_PROXY' if final['pass_gate'] else 'FAIL_PURE_MM_PROXY','chosen_profile':chosen_name,'selection_reason':reason,'selected_symbols':sorted(selected),'oos_hours':(test[1]-test[0])/3600,'paper_capital_usdc':PAPER_CAPITAL,'elapsed_sec':time.time()-t0,
                  'limitations':['Pure two-sided USDC maker: no directional/USDT lead signal used.','Binance DataVision aggTrades proxy; no exact L2 price-time queue reconstruction.','Quotes activate next second +75ms, require >=1 tick trade-through and synthetic queue depletion.','Inventory is controlled only by maker-side skew/recycling; no routine taker rescue.','0.10bps per-fill execution-friction stress is included; funding is not modeled in this short-horizon proxy.','Proxy PASS would still require true L2/live-paper confirmation.']})
    vps.to_csv(out/'validation_per_symbol_chosen.csv',index=False); all_ps.to_csv(out/'oos_all_symbols.csv',index=False); ps.to_csv(out/'oos_final_symbols.csv',index=False); ff.to_csv(out/'oos_final_fills.csv',index=False); mm.to_csv(out/'oos_roundtrips.csv',index=False); pd.DataFrame(skipped).to_csv(out/'skipped.csv',index=False)
    (out/'oos_final_summary.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8'); (out/'oos_all_summary.json').write_text(json.dumps(all_r,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n=== V5 PURE INVENTORY-NEUTRAL MM OOS ==='); print(json.dumps(final,ensure_ascii=False,indent=2))
    if not ps.empty:
        cols=['symbol','fills','cycles','daily_maker_notional_usdc','turnover_x_day_1000u','roundtrip_net_bps','mtm_net_pnl_usdc','max_inventory_usdc','mean_abs_inventory_usdc','median_hold_sec']
        print(ps[cols].sort_values('daily_maker_notional_usdc',ascending=False).to_string(index=False))

if __name__=='__main__': main()
