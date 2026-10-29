from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import ask_pdf
from agent_engine import agent_chat  # 引入 Agent
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


app = FastAPI(title="FinAgent-Pro API", description="金融投研Agent核心引擎")

# 定义请求体，加入 session_id
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

# 纯 RAG 接口
@app.post("/chat/rag")
async def rag_endpoint(request: ChatRequest):
    try:
        return {"code": 200, "reply": ask_pdf(request.query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 核心 Agent 接口
@app.post("/agent_chat")
async def agent_endpoint(request: ChatRequest):
    try:
        # 调用 agent_engine.py 里的核心函数
        answer = agent_chat(request.query, request.session_id)
        return {"code": 200, "session_id": request.session_id, "reply": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))