import os
from langchain_community.embeddings import HuggingFaceEmbeddings

# 动态获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models", "bge-m3")

# 如果本地没有模型，就回退到默认下载
if not os.path.exists(model_path):
    model_path = "BAAI/bge-m3"

global_embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)