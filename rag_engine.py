import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# 导入单例模型
from embedding_config import global_embeddings

# 初始化智谱大模型
client = OpenAI(
    api_key="API-key",  # 记得替换
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


def build_vector_store(pdf_path):
    print(f"正在加载并解析 PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 提取表格并转为 Markdown 格式
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(3, len(pdf.pages))):  # 取前三页测试
            tables = pdf.pages[i].extract_tables()
            for table in tables:
                table_text = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
                if table_text.strip():
                    from langchain_core.documents import Document
                    documents.append(Document(page_content=f"[表格数据]\n{table_text}",
                                              metadata={"source": pdf_path, "page": i + 1}))

    # 切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    #print(f"切分完成，共 {len(chunks)} 个文本块。")

    # 分批向量化（防止接口限流）
    #print("开始分批向量化（每次最多64条）...")
    batch_size = 64
    vector_store = None

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        #print(f"正在处理第 {i} 到 {min(i + batch_size, len(chunks))} 个块...")
        if vector_store is None:
            vector_store = FAISS.from_documents(batch_chunks, global_embeddings)
        else:
            vector_store.add_documents(batch_chunks)

    vector_store.save_local("faiss_index")
    #print("向量库已保存到本地。")


def ask_pdf(question):
    # 加载本地向量库
    vector_store = FAISS.load_local("faiss_index", global_embeddings, allow_dangerous_deserialization=True)

    # 检索
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 防幻觉 Prompt
    prompt = f"""你是一个专业的金融分析师。
请严格根据以下【参考信息】回答用户的问题。
如果在【参考信息】中找不到答案，请直接回答“根据现有资料，我无法回答该问题”，绝对不允许自己编造事实。

【参考信息】：
{context}

用户问题：{question}
"""
    # 调大模型
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # 第一次运行解开这行建库，之后注释掉
    build_vector_store("report.pdf")

    # 测试问答
    ans = ask_pdf("这家公司去年的营业收入是多少？")
    print("==== 大模型基于 RAG 的回答 ====")
    print(ans)