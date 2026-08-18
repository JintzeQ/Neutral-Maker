#!/usr/bin/env python3
import argparse, gzip, json, math
from collections import defaultdict, deque

PROFILES = [
    {"name":"V8_VOLUME","toxic_bps":3.0,"widen_bps":0.5,"hard_cap":15.0},
    {"name":"V8_BALANCED","toxic_bps":1.5,"widen_bps":1.0,"hard_cap":15.0},
    {"name":"V8_DEFENSIVE","toxic_bps":0.75,"widen_bps":2.0,"hard_cap":15.0},
]
NOTIONAL=5.0

def load(path):
    ev=[]
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            x=json.loads(line)
            if x.get("type")=="EVENT": ev.append(x)
    return ev

def replay(events,p):
    book={}; hist=defaultdict(lambda:deque(maxlen=30)); inv=defaultdict(float)
    cash=defaultdict(float); fills=[]; max_inv=0.0; rejected=0
    for e in events:
        d=e["data"]; s=d.get("s"); t=e["recv_ts_ms"]
        if not s: continue
        if "b" in d and "a" in d and "B" in d and "A" in d:
            b=float(d["b"]); a=float(d["a"]); mid=(a+b)/2
            book[s]=(b,a,mid,t); hist[s].append((t,mid))
            continue
        if d.get("e")!="trade" or s not in book: continue
        b,a,mid,bt=book[s]; px=float(d["p"])
        h=hist[s]
        momentum=0.0
        if len(h)>=2 and h[0][1]>0: momentum=(mid/h[0][1]-1)*10000
        # Directional toxicity: positive momentum makes asks toxic; negative makes bids toxic.
        buyer_maker=bool(d.get("m"))
        side="SELL" if not buyer_maker else "BUY"
        toxic = momentum if side=="SELL" else -momentum
        widen = p["widen_bps"] if toxic>p["toxic_bps"] else 0.0
        qbid=b*(1-widen/10000); qask=a*(1+widen/10000)
        fill=False; fpx=None
        if side=="BUY" and px<=qbid:
            if (inv[s]*mid)+NOTIONAL <= p["hard_cap"]+1e-9: fill=True; fpx=qbid
            else: rejected+=1
        elif side=="SELL" and px>=qask:
            if (inv[s]*mid)-NOTIONAL >= -p["hard_cap"]-1e-9: fill=True; fpx=qask
            else: rejected+=1
        if not fill: continue
        qty=NOTIONAL/fpx
        if side=="BUY": inv[s]+=qty; cash[s]-=NOTIONAL
        else: inv[s]-=qty; cash[s]+=NOTIONAL
        max_inv=max(max_inv,abs(inv[s]*mid))
        fills.append({"s":s,"t":t,"side":side,"px":fpx,"mid":mid})
    last={s:v[2] for s,v in book.items()}
    mtm=sum(cash[s]+inv[s]*last.get(s,0) for s in set(cash)|set(inv))
    notional=len(fills)*NOTIONAL
    loss=max(0.0,-mtm)
    cost_per_m=loss/notional*1e6 if notional else math.inf
    # zero-fee VIP objective; markout loss is embedded in MTM.
    return {"profile":p["name"],"fills":len(fills),"maker_notional":notional,"mtm_zero_fee":mtm,
            "loss_per_1m_volume":cost_per_m,"max_inventory_usdc":max_inv,"hard_cap":p["hard_cap"],
            "inventory_cap_pass":max_inv<=p["hard_cap"]+1e-6,"cap_rejections":rejected}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("--out",default="v8_vip_efficient_summary.json")
    a=ap.parse_args(); ev=load(a.input); n=len(ev); split=int(n*.6)
    val=ev[:split]; oos=ev[split:]
    vr=[replay(val,p) for p in PROFILES]
    # Pareto-like score: prefer volume, heavily penalize losses.
    best=max(vr,key=lambda x: x["maker_notional"]/(1+max(0,-x["mtm_zero_fee"])*10000))
    pp=next(p for p in PROFILES if p["name"]==best["profile"])
    oo=replay(oos,pp)
    out={"objective":"maximize qualified maker volume while minimizing adverse-selection loss",
         "validation":vr,"selected":best["profile"],"oos":oo,
         "pass": bool(oo["inventory_cap_pass"] and oo["maker_notional"]>0 and oo["mtm_zero_fee"]>=0)}
    with open(a.out,"w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
