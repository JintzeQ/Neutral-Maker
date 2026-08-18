#!/usr/bin/env python3
import argparse, asyncio, gzip, json, time
from pathlib import Path
import websockets

DEFAULT_SYMBOLS = ["BTCUSDC","ETHUSDC","BNBUSDC","SOLUSDC","XRPUSDC","ZECUSDC","DOGEUSDC"]

async def capture(symbols, seconds, out):
    streams=[]
    for s in symbols:
        x=s.lower()
        streams += [f"{x}@bookTicker", f"{x}@aggTrade"]
    url="wss://fstream.binance.com/stream?streams=" + "/".join(streams)
    path=Path(out); path.parent.mkdir(parents=True, exist_ok=True)
    counts={s:{"bookTicker":0,"aggTrade":0} for s in symbols}
    t0=time.monotonic(); deadline=t0+seconds
    with gzip.open(path,"wt",encoding="utf-8",compresslevel=3) as f:
        f.write(json.dumps({"type":"META","symbols":symbols,"started_ms":int(time.time()*1000),"source":"fstream.binance.com","streams":["bookTicker","aggTrade"]})+"\n")
        async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=5, max_queue=10000) as ws:
            while time.monotonic()<deadline:
                timeout=max(0.1, min(5.0, deadline-time.monotonic()))
                try:
                    raw=await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    continue
                recv_ms=int(time.time()*1000)
                obj=json.loads(raw)
                data=obj.get("data",obj)
                sym=data.get("s")
                et=data.get("e")
                if sym in counts:
                    if et=="aggTrade": counts[sym]["aggTrade"]+=1
                    elif ("b" in data and "B" in data and "a" in data and "A" in data): counts[sym]["bookTicker"]+=1
                f.write(json.dumps({"type":"EVENT","recv_ts_ms":recv_ms,"stream":obj.get("stream"),"data":data},separators=(",",":"))+"\n")
    summary={"elapsed_sec":time.monotonic()-t0,"counts":counts,"output":str(path)}
    print(json.dumps(summary,indent=2))
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seconds",type=int,default=90)
    ap.add_argument("--output",default="v6_bbo_capture.jsonl.gz")
    ap.add_argument("--symbols",default=",".join(DEFAULT_SYMBOLS))
    args=ap.parse_args()
    syms=[x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    asyncio.run(capture(syms,args.seconds,args.output))

if __name__=="__main__":
    main()
