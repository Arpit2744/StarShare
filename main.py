from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json, os

app = FastAPI(title="StarShare")
APP_DIR = os.path.dirname(os.path.abspath(__file__))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")

rooms = {}

@app.get("/")
async def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    try:
        room = rooms.setdefault(room_id, {"clients": set(), "names": set()})
        room["clients"].add(websocket)

        async def broadcast(event_type, data=None, exclude=None):
            payload = json.dumps({"type": event_type, **(data or {})})
            dead = []
            for ws in list(room["clients"]):
                if exclude and ws == exclude:
                    continue
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for d in dead:
                room["clients"].discard(d)

        await broadcast("presence", {"event": "join"})

        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            t = data.get("type")

            if t == "chat":
                await broadcast("chat", {"name": data.get("name", ""), "text": data.get("text", ""), "ts": data.get("ts", "")})
            elif t in ("offer", "answer", "ice"):
                await broadcast(t, {"from": data.get("from"), "payload": data.get("payload")}, exclude=websocket)
            elif t == "set_name":
                name = data.get("name")
                if name:
                    room["names"].add(name)
                    await broadcast("users", {"list": sorted(list(room["names"]))})
            elif t == "leave":
                await broadcast("presence", {"event": "leave"}, exclude=websocket)

    except WebSocketDisconnect:
        pass
    finally:
        if room_id in rooms:
            room = rooms[room_id]
            room["clients"].discard(websocket)
            if not room["clients"]:
                rooms.pop(room_id, None)
            else:
                await broadcast("presence", {"event": "leave"})
