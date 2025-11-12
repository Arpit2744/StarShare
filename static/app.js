const $ = (s) => document.querySelector(s);
const roomInput = $('#room'), nameInput = $('#name'), joinBtn = $('#join'),
      joinCard = $('#join-card'), roomUi = $('#room-ui'), shareBtn = $('#share'),
      stopShareBtn = $('#stopShare'), leaveBtn = $('#leave'), endCallBtn = $('#endCall'),
      remoteVideo = $('#remoteVideo'), msgInput = $('#msginput'), sendBtn = $('#send'),
      messagesEl = $('#messages'), statusBar = $('#statusBar');
let ws = null, pc = null, localStream = null, roomId = null, yourName = null;

function setStatus(connected, text) {
  statusBar.textContent = text;
  statusBar.className = connected ? 'status connected' : 'status waiting';
}

function appendMsg(name, text) {
  const d = document.createElement('div');
  d.textContent = name ? `${name}: ${text}` : text;
  messagesEl.appendChild(d);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function connectWS(rid) {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${protocol}://${location.host}/ws/${encodeURIComponent(rid)}`);

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'set_name', name: yourName }));
    setStatus(true, '🟢 Connected to room');
  };

  ws.onmessage = async (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === 'chat') return appendMsg(data.name, data.text);
    if (data.type === 'presence') {
      if (data.event === 'join') setStatus(true, '🟢 Partner joined');
      if (data.event === 'leave') setStatus(false, '🔴 Partner left');
    }
    if (data.type === 'offer') {
      if (!pc) await preparePeer();
      await pc.setRemoteDescription(new RTCSessionDescription(data.payload));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      ws.send(JSON.stringify({ type: 'answer', payload: answer }));
    }
    if (data.type === 'answer') await pc?.setRemoteDescription(new RTCSessionDescription(data.payload));
    if (data.type === 'ice') await pc?.addIceCandidate(new RTCIceCandidate(data.payload)).catch(e=>console.warn(e));
  };

  ws.onclose = () => setStatus(false, '🔴 Disconnected');
}

async function preparePeer() {
  pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
  pc.ontrack = (ev) => { remoteVideo.srcObject = ev.streams[0]; };
  pc.onicecandidate = (ev) => { if (ev.candidate) ws.send(JSON.stringify({ type: 'ice', payload: ev.candidate })); };
}

joinBtn.addEventListener('click', async () => {
  roomId = roomInput.value.trim();
  yourName = nameInput.value.trim() || 'Guest';
  if (!roomId) return alert('Room ID required');
  connectWS(roomId);
  joinCard.classList.add('hidden'); roomUi.classList.remove('hidden');
});

sendBtn.addEventListener('click', () => {
  const t = msgInput.value.trim(); if (!t || !ws) return;
  ws.send(JSON.stringify({ type: 'chat', name: yourName, text: t, ts: new Date().toISOString() }));
  msgInput.value = '';
});

shareBtn.addEventListener('click', async () => {
  try { localStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true }); }
  catch (e) { return alert('Screen share denied'); }
  if (!pc) await preparePeer();
  localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws.send(JSON.stringify({ type: 'offer', from: yourName, payload: offer }));
  shareBtn.classList.add('hidden'); stopShareBtn.classList.remove('hidden');
});

stopShareBtn.addEventListener('click', () => {
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  if (pc) pc.getSenders().forEach(s => { try { pc.removeTrack(s); } catch {} });
  shareBtn.classList.remove('hidden'); stopShareBtn.classList.add('hidden');
});

endCallBtn.addEventListener('click', () => {
  if (pc) pc.close(); pc = null;
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  ws?.send(JSON.stringify({ type: 'leave' }));
  setStatus(false, '🔴 Call ended');
});

leaveBtn.addEventListener('click', () => {
  ws?.close(); ws = null;
  if (pc) pc.close(); pc = null;
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  roomUi.classList.add('hidden'); joinCard.classList.remove('hidden');
  setStatus(false, '🔴 Left room');
});

window.addEventListener('beforeunload', () => { ws?.close(); pc?.close(); localStream?.getTracks().forEach(t => t.stop()); });
