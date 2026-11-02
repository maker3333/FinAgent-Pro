markdown
# FinAgent-Pro 金融投研 Agent 核心引擎

## 📌 项目简介
FinAgent-Pro 是一个面向金融投研场景的纯后端 LLM Agent 引擎。基于 FastAPI 构建，结合 RAG（检索增强生成）与手写 ReAct 循环，解决金融文档解析困难、大模型幻觉及外部工具调度等问题。项目提供标准的 RESTful API，支持无前端接入，方便快速集成到现有业务流中。

## 🛠️ 技术栈
- **后端框架**：FastAPI, Uvicorn, Pydantic
- **大模型**：智谱 GLM-4-Flash (兼容 OpenAI SDK)
- **RAG 组件**：LangChain, FAISS, 本地 BGE-M3 (Embedding), pdfplumber
- **Agent 调度**：纯手写 ReAct 循环 (Thought-Action-Observation)
- **工具与重试**：akshare (行情获取), tenacity (指数退避重试)

## 📂 项目结构
```text
FinAgent-Pro/
├── main.py                 # FastAPI 服务入口与路由定义
├── agent_engine.py         # 手写 ReAct 调度核心与状态流转
├── rag_engine.py           # PDF 解析、向量库构建与防幻觉问答
├── tools.py                # 外部工具集（行情、情绪分析）与重试逻辑
├── embedding_config.py     # 本地 BGE-M3 模型单例加载配置
├── requirements.txt        # 项目依赖清单
└── README.md
✨ 核心特性
手写 ReAct 调度器：不依赖黑盒 Agent 框架，手写 While 循环实现“思考-行动-观察”闭环。精准控制工具调用参数，内置 try-except 异常捕获与 JSON 解析纠错机制。

RAG 防幻觉与解析优化：针对财报 PDF 表格乱码痛点，引入 pdfplumber 提取结构化数据；设计强约束 Prompt，强制模型“无法回答时拒绝编造”，显著降低幻觉率。

本地模型与单例管理：将向量模型切换为本地 BGE-M3 并采用单例模式全局加载，避免高并发下的内存溢出（OOM），实现零 API 调用成本。

高可用工程化设计：基于 FastAPI 异步接口；为第三方金融 API 提供 tenacity 指数退避重试；对模型返回的非标准 JSON 做了严格的类型防御与正则清洗；针对服务降级设计了带明确警告的 Mock 数据兜底。

🚀 快速启动
1. 安装依赖
bash
pip install -r requirements.txt
2. 配置 API Key
在 agent_engine.py、rag_engine.py 和 tools.py 中，将 api_key="YOUR_API_KEY" 替换为你自己的智谱 API Key。

3. 准备本地模型 (可选)
默认代码会从本地 models/bge-m3 加载模型，如果未找到，会自动从 HuggingFace 下载 BAAI/bge-m3。

4. 构建向量库并启动服务
首次运行需取消 rag_engine.py 底部 build_vector_store("report.pdf") 的注释以构建知识库。
启动 FastAPI 服务：

bash
uvicorn main:app --reload
5. 访问接口文档
在浏览器打开 http://127.0.0.1:8000/docs，即可在 Swagger UI 中调试 /agent_chat 接口。

📡 核心接口
POST /agent_chat：核心 Agent 接口，接收用户 query 与 session_id，自动进行意图识别、工具路由与多轮对话记忆。

Request 示例：

json
{
  "query": "东方财富2025年度的营业总收入是多少？",
  "session_id": "user_01"
}
