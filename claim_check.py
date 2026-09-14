import os
import json
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

# 每一个 Claim 独立检索多少个主 Chunk
TOP_K = 5


# =========================================================
# 1. 环境变量
# =========================================================

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    raise ValueError(
        "没有找到 DEEPSEEK_API_KEY，请检查 .env 文件。"
    )


# =========================================================
# 2. DeepSeek
# =========================================================

deepseek_client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)


# =========================================================
# 3. Embedding
# =========================================================

print("正在加载 BGE-M3...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("BGE-M3 加载完成。")


# =========================================================
# 4. Chroma
# =========================================================

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


# =========================================================
# 5. DeepSeek 通用调用
# =========================================================

def call_deepseek(system_prompt, user_prompt):

    response = deepseek_client.chat.completions.create(
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

    return response.choices[0].message.content


# =========================================================
# 6. 自动拆分 Claim
# =========================================================

def split_claims(statement):

    system_prompt = """
你是一名社会科学论文编辑。

你的任务是把一段学术论述拆分为若干个可以分别进行
文献证据核验的最小事实性主张（Claim）。

规则：

1. 一个 Claim 尽量只包含一个可以被文献支持或反驳的判断。
2. 不要改变原文意思。
3. 不要添加原文没有的信息。
4. 不要评价 Claim 是否正确。
5. 不要寻找证据。
6. 不要把纯粹的连接词或修辞表达单独作为 Claim。
7. 如果一句话包含多个事实判断，应拆开。
8. 保留原文中的关键限定词，例如：
   “可能”“主要”“持续”“显著”“返乡后”等。
9. 输出必须是合法 JSON。
10. 不要输出 Markdown 代码框。
11. 不要输出任何 JSON 之外的文字。

格式必须严格为：

{
  "claims": [
    {
      "id": 1,
      "claim": "第一个主张"
    },
    {
      "id": 2,
      "claim": "第二个主张"
    }
  ]
}
"""

    user_prompt = f"""
请拆分下面这段学术论述：

{statement}
"""

    result = call_deepseek(
        system_prompt,
        user_prompt
    )

    # 防止模型偶尔返回 ```json
    result = (
        result
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    data = json.loads(result)

    return data["claims"]


# =========================================================
# 7. 检索单个 Claim
# =========================================================

def retrieve_claim(claim, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        claim,
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
    seen_keys = set()

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):

        source = metadatas[i]["source"]
        page = metadatas[i]["page"]
        chunk_index = metadatas[i]["chunk_index"]

        similarity = 1 - distances[i]

        # -------------------------
        # 主命中 Chunk
        # -------------------------

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

        # -------------------------
        # 相邻 Chunk
        # -------------------------

        for neighbor_index in [
            chunk_index - 1,
            chunk_index + 1
        ]:

            if neighbor_index < 1:
                continue

            neighbor_results = collection.get(
                where={
                    "$and": [
                        {
                            "source": source
                        },
                        {
                            "page": page
                        },
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
# 8. 构造单个 Claim 的证据 Context
# =========================================================

def build_context(retrieved):

    parts = []

    for i, item in enumerate(
        retrieved,
        start=1
    ):

        evidence_type = (
            "主检索证据"
            if item["type"] == "main"
            else "相邻补充证据"
        )

        part = f"""
【证据 {i}】

类型：{evidence_type}

文献：{item["source"]}

页码：{item["page"]}

本页 Chunk：{item["chunk_index"]}

相似度：{item["similarity"]:.4f}

原文：
{item["text"]}
"""

        parts.append(part)

    return "\n".join(parts)


# =========================================================
# 9. 核验单个 Claim
# =========================================================

def verify_claim(claim, context):

    system_prompt = """
你是一名社会科学论文的学术主张拆分助手。

你的唯一任务，是把用户输入的一段学术论述拆分为
若干个可以独立进行文献检索和证据核验的最小事实性主张
（Atomic Claims）。

必须严格遵守以下规则：

1. 每一个 Claim 只能包含一个核心事实判断。

2. 如果一句话同时包含两个可以分别成立或不成立的事实，
必须拆成两个 Claim。

例如：

原文：
“迁移提高了收入，但降低了主观社会地位。”

必须拆成：
Claim 1：迁移提高了收入。
Claim 2：迁移降低了主观社会地位。

不能把两者放在同一个 Claim 中。

3. 如果一个因果机制包含三个独立环节，例如：

A 导致 B，B 导致 C。

原则上拆成：
Claim 1：A 导致 B。
Claim 2：B 导致 C。
Claim 3：A 通过 B 影响 C。

这样既核验各个环节，也核验完整机制。

4. 时间持续性必须单独拆分。

例如：
“这种影响在返乡后仍然持续。”

应该成为独立 Claim。

5. 群体差异、时间差异、方向差异、效应大小等，
只要能够独立验证，都应单独拆分。

6. 保留原文的重要限定词，例如：
“可能”“显著”“主要”“部分”“持续”“返乡后”等。

7. 不允许增加原文不存在的事实。

8. 不允许改变原文的因果方向。

9. 不允许在拆分阶段判断 Claim 是否正确。

10. 不允许寻找证据或解释文献。

11. Claim 应尽量保持原文措辞，
不要主动把原文改写成更强或更弱的判断。

12. 每一个 Claim 应当能够单独输入搜索引擎，
并具有明确、完整的语义。

13. 输出必须是合法 JSON。

14. 不要输出 Markdown 代码框。

15. 不要输出 JSON 之外的任何文字。

格式：

{
  "claims": [
    {
      "id": 1,
      "claim": "第一个最小事实主张"
    },
    {
      "id": 2,
      "claim": "第二个最小事实主张"
    }
  ]
}
"""

    user_prompt = f"""
【待核验 Claim】

{claim}


【当前检索证据】

{context}


请严格按照下面格式回答：

判断：
只能填写“直接支持 / 间接支持 / 证据不足 / 相冲突”之一。

理由：
简洁说明为什么作出这一判断。

关键原文：
摘录最关键的原文证据。
如果没有足够证据，请写“当前检索证据不足”。

来源：
【文献文件名，第X页】

表述风险：
说明 Claim 是否存在过度概括、因果强化、
范围扩大、时间限定不准确等问题。
如果没有明显问题，写“未发现明显表述风险”。

建议：
如果原 Claim 可以直接使用，写“原表述可以保留”。
如果需要修改，给出一个更严格、更贴近证据的中文表述。
"""

    return call_deepseek(
        system_prompt,
        user_prompt
    )


# =========================================================
# 10. 总体评价
# =========================================================

def final_review(statement, reports):

    combined_reports = "\n\n".join(
        reports
    )

    system_prompt = """
你是一名社会科学论文编辑。

你需要根据已经完成的逐条 Claim 核验结果，
对原始论述做最终审查。

只能依据提供的核验结果。

不要重新引入任何外部知识。

重点判断：

1. 原论述整体是否得到文献支持。
2. 哪些部分证据最强。
3. 哪些部分需要删除、弱化或改写。
4. 引用该文献是否合适。
5. 给出一版尽可能保留原意、但与证据严格一致的修改稿。

不要因为一句话中的部分 Claim 得到支持，
就认为整句话完全成立。
"""

    user_prompt = f"""
【原始论述】

{statement}


【逐条 Claim 核验结果】

{combined_reports}


请输出：

一、总体支持程度
从以下四种中选择一个：
强支持 / 部分支持 / 证据不足 / 存在冲突

二、主要问题
指出需要特别注意的 Claim。

三、引用建议
判断现有文献是否适合作为这段论述的引用来源。

四、建议修改稿
给出一版符合当前证据的学术化中文表述。
"""

    return call_deepseek(
        system_prompt,
        user_prompt
    )


# =========================================================
# 11. 主程序
# =========================================================

print("\n" + "=" * 70)
print("学术论述 Claim 核验系统")
print("=" * 70)

statement = input(
    "\n请输入需要核验的学术论述：\n"
)

print("\n正在拆分 Claim...")

claims = split_claims(statement)


print("\n" + "=" * 70)
print("Claim 拆分结果")
print("=" * 70)

for item in claims:

    print(
        f"\nClaim {item['id']}："
        f"{item['claim']}"
    )


reports = []


# =========================================================
# 逐个 Claim 检索 + 核验
# =========================================================

for item in claims:

    claim_id = item["id"]
    claim_text = item["claim"]

    print("\n" + "=" * 70)
    print(
        f"正在核验 Claim {claim_id}"
    )
    print("=" * 70)

    print(
        f"\n{claim_text}"
    )

    print("\n正在检索证据...")

    retrieved = retrieve_claim(
        claim_text
    )

    # 显示主检索结果
    print("\n主检索结果：")

    main_number = 0

    for evidence in retrieved:

        if evidence["type"] == "main":

            main_number += 1

            print(
                f"\n{main_number}. "
                f"{evidence['source']}"
            )

            print(
                f"   第{evidence['page']}页"
            )

            print(
                f"   相似度："
                f"{evidence['similarity']:.4f}"
            )

    context = build_context(
        retrieved
    )

    print("\n正在让 DeepSeek 核验证据...")

    report = verify_claim(
        claim_text,
        context
    )

    formatted_report = f"""
==============================
Claim {claim_id}
==============================

原始 Claim：
{claim_text}

{report}
"""

    reports.append(
        formatted_report
    )

    print(
        "\n" + formatted_report
    )


# =========================================================
# 总体审查
# =========================================================

print("\n" + "=" * 70)
print("正在生成总体核验报告...")
print("=" * 70)

final_result = final_review(
    statement,
    reports
)

print("\n")
print("=" * 70)
print("最终核验报告")
print("=" * 70)

print(final_result)