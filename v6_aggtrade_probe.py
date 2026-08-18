#!/usr/bin/env python3
import asyncio, json, time
import websockets

STREAMS=["btcusdt@aggTrade","btcusdc@aggTrade","ethusdt@aggTrade","ethusdc@aggTrade"]

async def main():
    url="wss://fstream.binance.com/stream?streams="+"/".join(STREAMS)
    print("URL",url,flush=True)
    counts={s:0 for s in STREAMS}
    async with websockets.connect(url,ping_interval=20,ping_timeout=20,close_timeout=5) as ws:
        end=time.monotonic()+30
        while time.monotonic()<end:
            try:
                msg=await asyncio.wait_for(ws.recv(),timeout=2)
            except asyncio.TimeoutError:
                continue
            o=json.loads(msg)
            st=o.get("stream")
            if st in counts: counts[st]+=1
            if sum(counts.values())<=12:
                print(msg[:500],flush=True)
    print(json.dumps(counts,indent=2),flush=True)

if __name__=="__main__": asyncio.run(main())
