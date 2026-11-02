import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from openai import OpenAI

from agent_engine import agent_chat

app = FastAPI(title="FinAgent-Pro API", description="金融投研Agent核心引擎")

client = OpenAI(
    api_key="API-key",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 加入 session_id
class ChatRequest(BaseModel):
    query: str
    session_id: str = str(uuid.uuid4())

# 1. 核心 Agent 接口
@app.post("/agent_chat")
def agent_endpoint(request: ChatRequest):
    try:
        answer = agent_chat(request.query, request.session_id)
        return {"code": 200, "session_id": request.session_id, "reply": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. 流式输出接口
@app.post("/agent_stream")
async def agent_stream_endpoint(request: ChatRequest):
    def generate():
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": request.query}],
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield f"data: {chunk.choices[0].delta.content}\n\n"
        except Exception as e:
            yield f"data: [Error] {str(e)}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")