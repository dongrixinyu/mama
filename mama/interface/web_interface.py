from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.chat_service import ChatService
from backends.mock_ai_backend import MockAIBackend

app = FastAPI()

chat_service = ChatService(MockAIBackend())


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>AI Chat Web Demo</title>
    </head>
    <body>
        <h2>简易 AI 网页聊天</h2>
        <div id="chat-box" style="width:500px;height:300px;border:1px solid #ccc;overflow:auto;padding:10px;"></div>
        <br />
        <input id="msg" type="text" style="width:400px;" placeholder="请输入消息" />
        <button onclick="sendMessage()">发送</button>

        <script>
            async function sendMessage() {
                const input = document.getElementById("msg");
                const chatBox = document.getElementById("chat-box");
                const text = input.value.trim();
                if (!text) return;

                chatBox.innerHTML += "<p><b>你:</b> " + text + "</p>";
                input.value = "";

                const resp = await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({message: text})
                });

                const data = await resp.json();
                chatBox.innerHTML += "<p><b>AI:</b> " + data.reply + "</p>";
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """


@app.post("/chat")
async def chat(req: ChatRequest):
    reply = await chat_service.send_user_message(req.message)
    return {"reply": reply}

