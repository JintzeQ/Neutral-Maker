#!/usr/bin/env python3
import asyncio,json,time,websockets
STREAMS=[
"btcusdt@aggTrade","btcusdt@aggtrade","btcusdt@trade",
"btcusdc@aggTrade","btcusdc@aggtrade","btcusdc@trade",
]
async def main():
 url="wss://fstream.binance.com/stream?streams="+"/".join(STREAMS)
 print(url,flush=True); c={s:0 for s in STREAMS}; other={}
 async with websockets.connect(url,ping_interval=20,ping_timeout=20,close_timeout=5) as ws:
  end=time.monotonic()+20
  while time.monotonic()<end:
   try:m=await asyncio.wait_for(ws.recv(),timeout=2)
   except asyncio.TimeoutError:continue
   o=json.loads(m); st=o.get('stream'); c[st]=c.get(st,0)+1
   if c[st]<=2: print(m[:500],flush=True)
 print(json.dumps(c,indent=2),flush=True)
if __name__=='__main__':asyncio.run(main())
