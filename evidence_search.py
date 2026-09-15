import os
import chromadb

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# =========================================================
# 基础设置
# =========================================================

CHROMA_FOLDER = "chroma_db"
COLLECTION_NAME = "academic_papers"
EMBEDDING_MODEL = "BAAI/bge-m3"

# 先检索 8 个位置，后面让 DeepSeek 筛选
TOP_K = 8


# =========================================================
# 1. DeepSeek
# =========================================================

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    raise ValueError(
        "没有找到 DEEPSEEK_API_KEY，请检查 .env 文件。"
    )

deepseek_client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)


# =========================================================
# 2. Embedding
# =========================================================

print("正在加载 BGE-M3...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("BGE-M3 加载完成。")


# =========================================================
# 3. Chroma
# =========================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

print(
    f"知识库加载完成，共有 "
    f"{collection.count()} 个文本块。"
)

# =========================================================
# 查找指定论文
# =========================================================

def find_sources(keyword):

    # 读取知识库中的 metadata
    data = collection.get(
        include=["metadatas"]
    )

    sources = set()

    for metadata in data["metadatas"]:

        if metadata and "source" in metadata:
            sources.add(
                metadata["source"]
            )

    # -----------------------------------------------------
    # 支持多个关键词
    # 例如：
    # 刘欣 2001
    # Wei 2022
    # -----------------------------------------------------

    keywords = [
        word.strip().lower()
        for word in keyword.split()
        if word.strip()
    ]

    matched = []

    for source in sorted(sources):

        source_lower = source.lower()

        # 所有关键词都必须出现在文件名中
        if all(
            word in source_lower
            for word in keywords
        ):
            matched.append(source)

    return matched

# =========================================================
# 4. 检索观点
# =========================================================

def retrieve(
    query,
    top_k=TOP_K,
    source_filter=None
):

    # 每次检索时重新连接 Chroma，
    # 避免其他进程刚完成入库后仍使用旧的 HNSW reader
    fresh_client = chromadb.PersistentClient(
        path=CHROMA_FOLDER
    )

    fresh_collection = fresh_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 空知识库保护
    if fresh_collection.count() == 0:
        return []

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    # =====================================================
    # 是否限定某一篇论文
    # =====================================================

    if source_filter:

        results = fresh_collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            where={
                "source": source_filter
            },
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    else:

        results = fresh_collection.query(
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
    seen_keys = set()

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):

        source = metadatas[i]["source"]
        page = metadatas[i]["page"]
        chunk_index = metadatas[i]["chunk_index"]

        similarity = 1 - distances[i]

        # 主检索结果
        main_key = (
            source,
            page,
            chunk_index
        )

        if main_key not in seen_keys:

            retrieved.append({
                "text": documents[i],
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
                "similarity": similarity,
                "type": "main"
            })

            seen_keys.add(main_key)

        # -------------------------------------------------
        # 自动加入前后相邻 Chunk
        # -------------------------------------------------

        for neighbor_index in [
            chunk_index - 1,
            chunk_index + 1
        ]:

            if neighbor_index < 1:
                continue

            neighbor_results = fresh_collection.get(
                where={
                    "$and": [
                        {"source": source},
                        {"page": page},
                        {
                            "chunk_index":
                            neighbor_index
                        }
                    ]
                },
                include=[
                    "documents",
                    "metadatas"
                ]
            )

            if (
                neighbor_results["documents"]
                and len(
                    neighbor_results["documents"]
                ) > 0
            ):

                neighbor_doc = (
                    neighbor_results[
                        "documents"
                    ][0]
                )

                neighbor_meta = (
                    neighbor_results[
                        "metadatas"
                    ][0]
                )

                neighbor_key = (
                    neighbor_meta["source"],
                    neighbor_meta["page"],
                    neighbor_meta[
                        "chunk_index"
                    ]
                )

                if neighbor_key not in seen_keys:

                    retrieved.append({
                        "text": neighbor_doc,
                        "source":
                            neighbor_meta["source"],
                        "page":
                            neighbor_meta["page"],
                        "chunk_index":
                            neighbor_meta[
                                "chunk_index"
                            ],
                        "similarity":
                            similarity,
                        "type":
                            "neighbor"
                    })

                    seen_keys.add(
                        neighbor_key
                    )

    return retrieved


# =========================================================
# 5. 构造证据文本
# =========================================================

def build_context(retrieved):

    parts = []

    for i, item in enumerate(
        retrieved,
        start=1
    ):

        evidence_type = (
            "主检索结果"
            if item["type"] == "main"
            else "相邻上下文"
        )

        part = f"""
【证据 {i}】

类型：{evidence_type}

论文：
{item["source"]}

页码：
{item["page"]}

Chunk：
{item["chunk_index"]}

向量相似度：
{item["similarity"]:.4f}

原文：
{item["text"]}
"""

        parts.append(part)

    return "\n".join(parts)


# =========================================================
# 6. DeepSeek 整理证据
# =========================================================

def analyze_evidence(query, context):

    system_prompt = """
你是一名社会科学学术文献检索助手。

用户会提供一个需要寻找文献依据的观点。
系统已经从用户自己的论文知识库中检索了一组相关原文。

你的任务不是写综述，而是帮助用户快速定位
“这个观点在什么论文、什么页码、什么原文中出现”。

必须遵守：

1. 只能依据提供的检索证据。
2. 不得使用自己的知识补充事实。
3. 不得编造论文、作者、页码或参考文献。
4. 优先寻找与用户观点最直接对应的证据。
5. 不要因为语义相似就判断为直接支持。
6. 必须区分：
   - 高度相关：原文直接表达或明确支持该观点
   - 部分相关：只支持观点的一部分
   - 背景相关：只是讨论相同主题
7. 优先展示“高度相关”的结果。
8. 相同论文、相同页码、内容高度重叠的证据应合并。
9. 必须保留最关键的原文，方便用户人工核对。
10. 不得把原文中没有的内容加入“原文”。
11. 如果原文中明确出现括号引用，
    例如：
    (Knight & Gunatilaka, 2010b; Wang, 2017)
    应单独提取出来。
12. 只能提取检索证据中实际出现的参考文献。
13. 不得根据作者姓氏自行补全参考文献题名。
14. 如果当前片段只出现作者和年份，
    就只输出作者和年份。
15. 如果没有发现直接证据，应明确告诉用户，
    不要为了给出答案而强行匹配。
16. 输出的重点是“定位”，不是长篇解释。
"""

    user_prompt = f"""
【需要定位的观点】

{query}


【知识库检索证据】

{context}


请筛选真正有用的证据，并按照以下格式回答：

## 定位结论

用1—2句话说明：
当前知识库是否找到能够支持或讨论该观点的文献。

## 最相关证据

### 证据1

相关程度：
高度相关 / 部分相关 / 背景相关

论文：
文件名

页码：
第X页

关键原文：
只摘录最关键的相关原文。

说明：
用中文简要说明这段原文与用户观点的关系。

涉及的参考文献：
只列该段原文中明确出现的“作者 + 年份”。
如果没有，写“未发现”。

---

继续列出其他真正有价值的证据。
不要为了凑数量列出明显不相关结果。

## 建议继续追溯的参考文献

汇总上述关键证据中明确出现、
并且与用户观点直接相关的参考文献。

只允许列出检索原文实际出现的作者和年份。

如果没有，写“当前检索证据中未发现”。

## 使用建议

简要告诉用户：
哪些证据最适合直接引用；
哪些参考文献值得进一步查找原文。

原则上只保留最有价值的5条证据。
只有在第6条及之后包含前5条没有的重要信息时，才继续列出。
优先级为：
直接对应观点的原文 > 实证结果 > 理论解释 > 背景讨论。
"""

    response = (
        deepseek_client
        .chat.completions
        .create(
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

            temperature=0.0
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# =========================================================
# 7. 主程序
# =========================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("学术观点证据定位器")
    print("=" * 70)

    print("""
    请选择检索模式：

    1. 全库检索
    ——不知道应该引用哪篇文献时使用

    2. 指定文献检索
    ——已经有目标文献，想核对观点和出处时使用
    """)

    mode = input(
        "请输入 1 或 2："
    ).strip()


    # =========================================================
    # 模式 1：全库检索
    # =========================================================

    if mode == "1":

        source_filter = None

        query = input(
            "\n请输入需要查找文献依据的观点：\n"
        )

        print(
            "\n正在全部论文中检索..."
        )


    # =========================================================
    # 模式 2：指定论文
    # =========================================================

    elif mode == "2":

        print(
            "\n请输入作者、年份或论文题目关键词。"
        )

        print(
            "例如：刘欣 2001"
        )

        print(
            "或者：Wei 2022"
        )

        while True:

            source_keyword = input(
                "\n请输入："
            ).strip()

            if source_keyword:
                break
            print(
                "关键词不能为空，请输入作者、年份或题目关键词。"
            )

        matched_sources = find_sources(
        source_keyword
    )


        # -----------------------------------------------------
        # 没找到
        # -----------------------------------------------------

        if len(matched_sources) == 0:

            print(
                "\n没有找到匹配的论文。"
            )

            print(
                "请检查关键词，"
                "或者确认 PDF 是否已经加入 papers/ 并完成入库。"
            )

            raise SystemExit


        # -----------------------------------------------------
        # 只找到一篇
        # -----------------------------------------------------

        elif len(matched_sources) == 1:

            source_filter = (
                matched_sources[0]
            )

            print(
                "\n找到目标论文："
            )

            print(
                source_filter
            )


        # -----------------------------------------------------
        # 找到多篇
        # -----------------------------------------------------

        else:

            print(
                "\n找到多篇匹配论文：\n"
            )

            for i, source in enumerate(
                matched_sources,
                start=1
            ):

                print(
                    f"{i}. {source}"
                )


            while True:

                choice = input(
                    "\n请选择论文编号："
                ).strip()

                if (
                    choice.isdigit()
                    and 1 <= int(choice)
                    <= len(matched_sources)
                ):

                    source_filter = (
                        matched_sources[
                            int(choice) - 1
                        ]
                    )

                    break

                print(
                    "输入无效，请重新输入。"
                )


            print(
                "\n已选择："
            )

            print(
                source_filter
            )


        query = input(
            "\n请输入需要在该论文中定位的观点：\n"
        )

        print(
            "\n正在指定论文中检索..."
        )


    # =========================================================
    # 输入错误
    # =========================================================

    else:

        print(
            "\n输入无效，请重新运行程序，"
            "并输入 1 或 2。"
        )

        raise SystemExit


    # =========================================================
    # 8. 执行检索
    # =========================================================

    retrieved = retrieve(
        query,
        source_filter=source_filter
    )


    # =========================================================
    # 9. 显示原始 Top K
    # =========================================================

    print("\n" + "=" * 70)
    print("原始检索位置")
    print("=" * 70)

    number = 0

    for item in retrieved:

        if item["type"] == "main":

            number += 1

            print(
                f"\n【{number}】"
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


    # =========================================================
    # 10. DeepSeek 整理证据
    # =========================================================

    context = build_context(
        retrieved
    )

    print(
        "\n正在整理最相关证据..."
    )

    answer = analyze_evidence(
        query,
        context
    )


    # =========================================================
    # 11. 输出
    # =========================================================

    print("\n")
    print("=" * 70)

    if mode == "1":

        print(
            "全库观点证据定位结果"
        )

    else:

        print(
            "指定文献观点定位结果"
        )

        print(
            f"\n目标论文：{source_filter}"
        )

    print("=" * 70)

    print(answer)

