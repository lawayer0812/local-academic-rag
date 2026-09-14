import chromadb

from sentence_transformers import SentenceTransformer


# ==============================
# 基础设置
# ==============================

CHROMA_FOLDER = "chroma_db"

COLLECTION_NAME = "academic_papers"

EMBEDDING_MODEL = "BAAI/bge-m3"

TOP_K = 5


# ==============================
# 1. 加载 Embedding 模型
# ==============================

print("正在加载 Embedding 模型...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding 模型加载完成！")


# ==============================
# 2. 连接 Chroma 数据库
# ==============================

client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"知识库加载完成，共有 "
    f"{collection.count()} 个文本块。"
)


# ==============================
# 3. 输入问题
# ==============================

query = input(
    "\n请输入你想检索的问题：\n"
)


# ==============================
# 4. 将问题转成向量
# ==============================

query_embedding = model.encode(
    query,
    normalize_embeddings=True
)


# ==============================
# 5. 在 Chroma 中检索
# ==============================

results = collection.query(

    query_embeddings=[
        query_embedding.tolist()
    ],

    n_results=TOP_K,

    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


# ==============================
# 6. 显示结果
# ==============================

print("\n")
print("=" * 70)
print("检索结果")
print("=" * 70)


documents = results["documents"][0]

metadatas = results["metadatas"][0]

distances = results["distances"][0]


for i in range(len(documents)):

    document = documents[i]

    metadata = metadatas[i]

    distance = distances[i]

    # Chroma 的 cosine distance 越小越相似
    similarity = 1 - distance

    print("\n" + "=" * 70)

    print(
        f"【结果 {i + 1}】"
    )

    print(
        f"论文：{metadata['source']}"
    )

    print(
        f"页码：{metadata['page']}"
    )

    print(
        f"本页 Chunk："
        f"{metadata['chunk_index']}"
    )

    print(
        f"相似度：{similarity:.4f}"
    )

    print("\n原文：")

    print(document)