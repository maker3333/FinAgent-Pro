# FinAgent-Pro 金融投研 Agent 核心引擎

## 📌 项目简介
FinAgent-Pro 是一个面向金融投研场景的纯后端 LLM Agent 引擎。基于 FastAPI 构建，结合 RAG（检索增强生成）与手写 ReAct 循环，解决金融文档解析困难、大模型幻觉及外部工具调度等问题。项目提供标准的 RESTful API，支持无前端接入。

## 🛠️ 技术栈
- **后端框架**：FastAPI, Uvicorn, Pydantic
- **大模型**：智谱 GLM-4-Flash
- **RAG 组件**：LangChain, FAISS, 本地 BGE-M3 (Embedding), pdfplumber
- **Agent 调度**：纯手写 ReAct 循环 (Thought-Action-Observation)
- **工具与重试**：akshare (行情获取), tenacity (指数退避重试)

## ✨ 核心特性
1. **手写 ReAct 调度器**：不依赖黑盒 Agent 框架，手写 While 循环实现“思考-行动-观察”闭环。精准控制工具调用参数，内置异常捕获与自我纠错机制。
2. **RAG 防幻觉与解析优化**：针对财报 PDF 表格乱码痛点，引入 `pdfplumber` 提取结构化数据；设计强约束 Prompt，强制模型“无法回答时拒绝编造”，显著降低幻觉率。
3. **本地模型与单例管理**：将向量模型切换为本地 `BGE-M3` 并采用单例模式全局加载，避免高并发下的内存溢出（OOM），实现零 API 调用成本。
4. **高可用工程化设计**：基于 FastAPI 异步接口；为第三方金融 API 提供 `tenacity` 指数退避重试；对模型返回的非标准 JSON 做了严格的类型防御与正则清洗。

## 🚀 快速启动
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
