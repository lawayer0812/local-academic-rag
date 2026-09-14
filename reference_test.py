import json
import os
from functools import lru_cache

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# 基础设置
# =========================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_config():
    config = {
        "papers_folder": "papers",
    }
    config_path = os.path.join(PROJECT_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            if isinstance(user_config, dict):
                config.update(user_config)
        except Exception:
            # 配置文件读取失败时仍使用默认值，避免影响主程序。
            pass
    return config


CONFIG = _load_config()
PAPERS_FOLDER = os.path.join(
    PROJECT_DIR,
    CONFIG.get("papers_folder", "papers"),
)

load_dotenv(os.path.join(PROJECT_DIR, ".env"))
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件。")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

REFERENCE_KEYWORDS = [
    "参考文献",
    "参考 文献",
    "References",
    "REFERENCES",
    "Bibliography",
    "BIBLIOGRAPHY",
]


# =========================================================
# 工具函数
# =========================================================
def _safe_json_loads(text):
    """尽量把模型返回内容解析为 JSON。"""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


@lru_cache(maxsize=64)
def get_reference_section(source):
    """
    读取一篇 PDF 的参考文献部分。

    返回：
    {
        "found": bool,
        "source": str,
        "start_page": int | None,
        "reference_text": str,
        "message": str,
    }
    """
    pdf_path = os.path.join(PAPERS_FOLDER, source)

    if not os.path.exists(pdf_path):
        return {
            "found": False,
            "source": source,
            "start_page": None,
            "reference_text": "",
            "message": "未找到原始 PDF 文件。",
        }

    doc = pymupdf.open(pdf_path)
    try:
        reference_start_page = None

        for page_index in range(len(doc)):
            text = doc[page_index].get_text("text") or ""
            text_lower = text.lower()

            for keyword in REFERENCE_KEYWORDS:
                if keyword.lower() in text_lower:
                    reference_start_page = page_index
                    break

            if reference_start_page is not None:
                break

        if reference_start_page is None:
            return {
                "found": False,
                "source": source,
                "start_page": None,
                "reference_text": "",
                "message": "没有自动识别到参考文献部分。",
            }

        reference_pages = []
        for page_index in range(reference_start_page, len(doc)):
            text = doc[page_index].get_text("text").strip()
            reference_pages.append(
                f"【PDF第{page_index + 1}页】\n{text}"
            )

        return {
            "found": True,
            "source": source,
            "start_page": reference_start_page + 1,
            "reference_text": "\n\n".join(reference_pages),
            "message": "已找到参考文献部分。",
        }
    finally:
        doc.close()


def extract_explicit_citations(evidence_text, max_citations=6):
    """
    只从检索到的原文片段中提取明确出现的“作者 + 年份”引用。
    不允许模型凭知识补充。
    """
    prompt = f"""
你是严谨的学术引文抽取助手。

请从下面的“原文证据”中，只提取正文里明确出现的作者—年份引用。
例如：
(Knight & Gunatilaka, 2010b; Wang, 2017)
Wei (2022)
吴愈晓（2013）

严格规则：
1. 只能提取原文中实际出现的引用。
2. 不得补全题名、期刊或其他信息。
3. 不要把 PDF 文件名、页眉、参考文献条目本身当作正文引用。
4. 去重。
5. 最多返回 {max_citations} 条，优先保留与当前证据内容最直接相关的引用。
6. 如果没有明确的作者—年份引用，返回空数组。
7. 只返回严格 JSON，不要输出解释或 Markdown。

返回格式：
{{"citations": ["Knight & Gunatilaka, 2010b", "Wang, 2017"]}}

【原文证据】
{evidence_text}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你只能抽取用户提供文本中明确出现的引文，"
                    "不得使用外部知识补全。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    data = _safe_json_loads(response.choices[0].message.content)
    citations = data.get("citations", []) if isinstance(data, dict) else []

    cleaned = []
    seen = set()
    for citation in citations:
        citation = str(citation).strip()
        key = citation.lower()
        if citation and key not in seen:
            cleaned.append(citation)
            seen.add(key)
        if len(cleaned) >= max_citations:
            break

    return cleaned


def match_citations_in_references(citations, reference_text):
    """在一篇论文自己的参考文献区中批量匹配若干作者—年份引用。"""
    citation_block = "\n".join(
        f"{i}. {citation}"
        for i, citation in enumerate(citations, start=1)
    )

    prompt = f"""
你是一个学术文献参考文献匹配助手。

你的任务是从我提供的论文“参考文献原文”中，
为每一个目标引用找到对应的完整参考文献条目。

【目标引用】
{citation_block}

【论文参考文献原文】
{reference_text}

严格规则：
1. 只能使用上面提供的参考文献原文。
2. 禁止使用外部知识补全作者、题名、期刊、出版社、页码等信息。
3. PDF 跨行造成的断行可以合并，但不能增加原文不存在的信息。
4. 必须逐条处理所有目标引用。
5. 明确找到时写“匹配成功”；找不到写“未找到”；有多个可能项写“存在多个候选”。
6. 匹配成功时必须保留完整参考文献原文，并注明它所在的 PDF 页码。
7. 不要虚构任何参考文献。

请按照下面的 Markdown 格式输出：

### 引用1：作者 + 年份
**匹配结果：** 匹配成功 / 未找到 / 存在多个候选

**完整参考文献：**
原文条目；若未找到则写“未找到”。

**PDF页码：** 第X页；若未找到则写“—”。

**说明：** 一句话说明匹配依据。

依次处理所有目标引用。
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是严谨的参考文献匹配助手。"
                    "只能依据用户提供的参考文献原文，绝不能凭知识补全。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def trace_references(retrieved, max_sources=3, max_citations_per_source=6):
    """
    对当前 RAG 检索证据做可选的参考文献追溯。

    流程：
    1. 按来源论文聚合当前检索证据；
    2. 从证据中提取明确出现的作者—年份引用；
    3. 打开该来源论文自己的参考文献区；
    4. 批量匹配完整参考文献条目。
    """
    grouped = {}
    source_scores = {}

    for item in retrieved:
        source = item.get("source")
        if not source:
            continue

        grouped.setdefault(source, []).append(item)
        if item.get("type") == "main":
            source_scores[source] = max(
                source_scores.get(source, -1),
                float(item.get("similarity", 0)),
            )

    ranked_sources = sorted(
        grouped,
        key=lambda s: source_scores.get(s, 0),
        reverse=True,
    )[:max_sources]

    results = []

    for source in ranked_sources:
        evidence_parts = []
        for item in grouped[source]:
            evidence_parts.append(
                f"【PDF第{item.get('page')}页】\n{item.get('text', '')}"
            )

        evidence_text = "\n\n".join(evidence_parts)
        citations = extract_explicit_citations(
            evidence_text,
            max_citations=max_citations_per_source,
        )

        if not citations:
            results.append({
                "source": source,
                "citations": [],
                "reference_start_page": None,
                "status": "no_citations",
                "message": "当前检索证据中未发现明确的作者—年份引用。",
                "answer": "",
            })
            continue

        ref_data = get_reference_section(source)

        if not ref_data["found"]:
            results.append({
                "source": source,
                "citations": citations,
                "reference_start_page": None,
                "status": "no_reference_section",
                "message": ref_data["message"],
                "answer": "",
            })
            continue

        answer = match_citations_in_references(
            citations,
            ref_data["reference_text"],
        )

        results.append({
            "source": source,
            "citations": citations,
            "reference_start_page": ref_data["start_page"],
            "status": "ok",
            "message": "参考文献追溯完成。",
            "answer": answer,
        })

    return results


# =========================================================
# 保留命令行测试入口
# =========================================================
if __name__ == "__main__":
    print("参考文献追溯测试")
    print("=" * 70)

    source = input("请输入 PDF 完整文件名：\n").strip()
    citation = input("请输入需要查找的作者和年份（例如 Wright 1978）：\n").strip()

    if not source or not citation:
        raise SystemExit("输入不能为空。")

    ref_data = get_reference_section(source)
    if not ref_data["found"]:
        raise SystemExit(ref_data["message"])

    answer = match_citations_in_references(
        [citation],
        ref_data["reference_text"],
    )
    print(answer)
