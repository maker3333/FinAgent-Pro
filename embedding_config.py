from langchain_community.embeddings import HuggingFaceEmbeddings

print("正在加载本地 BGE-M3 模型（只需加载一次，请稍候）...")

# 全局只实例化这一次
global_embeddings = HuggingFaceEmbeddings(
    model_name=r"D:\products\FinAgent\models\bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("✅ BGE-M3 模型加载成功，常驻内存。")