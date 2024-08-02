from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
from chat99 import generate_response, advanced_router

app = FastAPI()

class ChatMessage(BaseModel):
    message: str

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    conversation = []

    while True:
        try:
            data = await websocket.receive_text()
            message = json.loads(data)
            user_input = message['message']

            conversation.append({"role": "user", "content": user_input})

            route_config = advanced_router.route(user_input, conversation)
            model = route_config['model']
            max_tokens = route_config['max_tokens']
            temperature = route_config['temperature']
            response_strategy = route_config['response_strategy']

            content = generate_response(model, conversation, max_tokens, temperature, response_strategy)

            if content:
                response = {
                    "message": content,
                    "model": model,
                    "strategy": response_strategy
                }
                await websocket.send_json(response)
                conversation.append({"role": "assistant", "content": content})
            else:
                await websocket.send_json({"error": "Failed to get a response. Please try again."})
        except Exception as e:
            await websocket.send_json({"error": f"An error occurred: {str(e)}"})

# Mount static files for UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")