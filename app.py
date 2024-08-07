from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import json
from chat99 import chat_with_99

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat99 Interface</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #chat-box { height: 400px; border: 1px solid #ccc; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
            #user-input { width: 100%; padding: 5px; }
            .user-message { color: blue; }
            .assistant-message { color: green; }
        </style>
    </head>
    <body>
        <h1>Chat99 Interface</h1>
        <div id="chat-box"></div>
        <input type="text" id="user-input" placeholder="Type your message here...">
        <button onclick="sendMessage()">Send</button>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('chat-box');
                var data = JSON.parse(event.data);
                var message = document.createElement('div');
                if (data.error) {
                    message.innerHTML = '<span style="color: red;">Error: ' + data.error + '</span>';
                } else {
                    message.innerHTML = '<span class="assistant-message">Chat99: ' + data.message + '</span>';
                }
                messages.appendChild(message);
                messages.scrollTop = messages.scrollHeight;
            };
            function sendMessage() {
                var input = document.getElementById("user-input");
                var message = input.value;
                if (message.trim() !== "") {
                    ws.send(JSON.stringify({message: message}));
                    var messages = document.getElementById('chat-box');
                    messages.innerHTML += '<div class="user-message">You: ' + message + '</div>';
                    input.value = '';
                    messages.scrollTop = messages.scrollHeight;
                }
            }
            document.getElementById("user-input").addEventListener("keypress", function(event) {
                if (event.key === "Enter") {
                    event.preventDefault();
                    sendMessage();
                }
            });
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    conversation_history = []

    while True:
        try:
            data = await websocket.receive_text()
            message = json.loads(data)
            user_input = message['message']

            response = chat_with_99(user_input, conversation_history)

            await websocket.send_json({"message": response})
        except Exception as e:
            await websocket.send_json({"error": f"An error occurred: {str(e)}"})