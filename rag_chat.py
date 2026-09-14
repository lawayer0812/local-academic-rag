import os
import chromadb

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ==============================
# 基础设置
# ==============================

CHROMA_FOLDER = "chroma_db"

COLLECTION_NAME = "academic_papers"

EMBEDDING_MODEL = "BAAI/bge-m3"

TOP_K = 5


# ==============================
# 1. 读取 .env
# ==============================

load_dotenv()

deepseek_api_key = os.getenv(
    "DEEPSEEK_API_KEY"
)

if not deepseek_api_key:
    raise ValueError(
        "没有找到 DEEPSEEK_API_KEY，请检查 .env 文件。"
    )


# ==============================
# 2. 初始化 DeepSeek
# ==============================

client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)


# ==============================
# 3. 加载 Embedding 模型
# ==============================

print("正在加载 Embedding 模型...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding 模型加载完成！")


# ==============================
# 4. 连接 Chroma
# ==============================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"知识库加载完成，共有 "
    f"{collection.count()} 个文本块。"
)


# ==============================
# 5. 检索函数
# ==============================

def retrieve(query, top_k=TOP_K):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    seen_keys = set()

    for i in range(len(documents)):

        similarity = 1 - distances[i]

        main_source = metadatas[i]["source"]
        main_page = metadatas[i]["page"]
        main_chunk_index = metadatas[i]["chunk_index"]

        # 先加入主命中 chunk
        main_key = (
            main_source,
            main_page,
            main_chunk_index
        )

        if main_key not in seen_keys:

            retrieved.append({
                "text": documents[i],
                "source": main_source,
                "page": main_page,
                "chunk_index": main_chunk_index,
                "similarity": similarity,
                "type": "main"
            })

            seen_keys.add(main_key)

        # 再加入前后相邻 chunk
        for neighbor_index in [
            main_chunk_index - 1,
            main_chunk_index + 1
        ]:

            if neighbor_index < 1:
                continue

            neighbor_results = collection.get(
                where={
                    "$and": [
                        {"source": main_source},
                        {"page": main_page},
                        {"chunk_index": neighbor_index}
                    ]
                },
                include=[
                    "documents",
                    "metadatas"
                ]
            )

            if (
                neighbor_results["documents"]
                and len(neighbor_results["documents"]) > 0
            ):

                neighbor_doc = (
                    neighbor_results["documents"][0]
                )

                neighbor_meta = (
                    neighbor_results["metadatas"][0]
                )

                neighbor_key = (
                    neighbor_meta["source"],
                    neighbor_meta["page"],
                    neighbor_meta["chunk_index"]
                )

                if neighbor_key not in seen_keys:

                    retrieved.append({
                        "text": neighbor_doc,
                        "source":
                            neighbor_meta["source"],
                        "page":
                            neighbor_meta["page"],
                        "chunk_index":
                            neighbor_meta["chunk_index"],
                        "similarity":
                            similarity,
                        "type": "neighbor"
                    })

                    seen_keys.add(
                        neighbor_key
                    )

    return retrieved

# ==============================
# 6. 构造 Context
# ==============================

def build_context(retrieved_chunks):

    context_parts = []

    for i, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        evidence_type = (
            "主检索证据"
            if item["type"] == "main"
            else "相邻补充证据"
        )

        context_part = f"""
【证据 {i}】

类型：
{evidence_type}

文献：
{item["source"]}

页码：
{item["page"]}

本页 Chunk：
{item["chunk_index"]}

相似度：
{item["similarity"]:.4f}

原文：
{item["text"]}
"""

        context_parts.append(
            context_part
        )

    return "\n".join(
        context_parts
    )


# ==============================
# 7. 调用 DeepSeek
# ==============================

def ask_deepseek(
    question,
    context
):

    system_prompt = """
你是一名严格的社会科学文献证据核验助手。

你的唯一信息来源是系统提供给你的“检索证据”。

请严格遵守以下规则：

1. 只能使用检索证据中明确出现的信息。
2. 禁止使用你自己的背景知识、常识或训练数据补充内容。
3. 不得根据一般方法论知识推测论文的局限、假设或研究设计。
4. 不得因为当前检索片段没有出现某项分析，就断言整篇论文没有进行该分析。
5. 如果当前证据没有显示某项信息，只能说：
   “当前检索到的证据中未显示该信息。”
   不得说：
   “该论文没有……”
6. 不得编造作者、年份、变量、系数、模型、样本、页码或研究结论。
7. 必须严格区分：
   A. 文献明确报告的研究发现
   B. 文献作者提出的解释
   C. 根据现有证据可以做出的谨慎概括
8. 如果用户询问因果关系：
   只有当检索证据明确使用 causal、causal effect、
   difference-in-differences、实验、准实验等因果识别表述时，
   才可以描述为因果效应。
9. 如果证据只是相关关系，不得改写成因果关系。
10. 每一个重要判断后都必须标明证据：
   【文件名，第X页】
11. 如果多个证据之间存在冲突，必须明确指出。
12. 如果证据不足，请明确写：
   “根据当前检索到的证据，暂时无法判断。”
13. 不要为了让答案完整而补充证据之外的信息。
14. “不知道”或“当前证据不足”是完全可接受的回答。
15. 精确性和可追溯性高于答案的完整性。
"""

    user_prompt = f"""
用户问题：

{question}


以下是知识库检索到的文献证据：

{context}


请严格根据以上证据回答。

请按照以下结构：

一、核心结论
用1—3句话直接回答问题。

二、直接证据
逐条列出与问题直接相关的文献发现。
每一条必须注明：
【文件名，第X页】

三、可能的作用机制
只有当检索证据明确讨论机制时才写。
如果证据没有明确讨论，请写：
“当前检索证据不足以判断作用机制。”

四、证据状态
请区分：
- 直接支持
- 间接支持
- 当前证据不足

不得根据“当前没有检索到”推断“原论文不存在”。

五、来源
列出本次实际使用的文献及页码。
"""

    response = client.chat.completions.create(
        model="deepseek-chat",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ==============================
# 8. 主程序
# ==============================

print("\n" + "=" * 70)
print("学术 RAG 已启动")
print("=" * 70)

question = input(
    "\n请输入你的问题：\n"
)


print("\n正在检索相关文献...")

retrieved_chunks = retrieve(
    question
)


# ==============================
# 9. 先显示检索结果
# ==============================

print("\n")
print("=" * 70)
print("检索到的文献证据")
print("=" * 70)

for i, item in enumerate(
    retrieved_chunks,
    start=1
):

    print(
        f"\n【证据 {i}】"
    )

    print(
        f"论文：{item['source']}"
    )

    print(
        f"页码：{item['page']}"
    )

    print(
        f"相似度："
        f"{item['similarity']:.4f}"
    )


# ==============================
# 10. 构造 Context
# ==============================

context = build_context(
    retrieved_chunks
)


# ==============================
# 11. DeepSeek 回答
# ==============================

print(
    "\n正在调用 DeepSeek..."
)

answer = ask_deepseek(
    question,
    context
)


print("\n")
print("=" * 70)
print("DeepSeek 回答")
print("=" * 70)

print(answer)