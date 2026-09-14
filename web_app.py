import json
import os
import re
import socket
import time
import sys
import subprocess
from urllib.parse import quote

import chromadb
import streamlit as st

from evidence_search import retrieve, build_context, analyze_evidence
from reference_test import trace_references


# =========================================================
# 1. 页面与配置
# =========================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    config = {
        "app_name": "学术观点证据定位器",
        "papers_folder": "papers",
        "chroma_folder": "chroma_db",
        "collection_name": "academic_papers",
        "pdf_port": 8502,
    }

    config_path = os.path.join(PROJECT_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            if isinstance(user_config, dict):
                config.update(user_config)
        except Exception as e:
            st.warning(f"config.json 读取失败，暂时使用默认配置：{e}")

    return config


CONFIG = load_config()
APP_NAME = CONFIG.get("app_name", "学术观点证据定位器")
PAPERS_FOLDER = os.path.join(
    PROJECT_DIR,
    CONFIG.get("papers_folder", "papers"),
)
CHROMA_FOLDER = os.path.join(
    PROJECT_DIR,
    CONFIG.get("chroma_folder", "chroma_db"),
)
COLLECTION_NAME = CONFIG.get("collection_name", "academic_papers")
PDF_PORT = int(CONFIG.get("pdf_port", 8502))

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📚",
    layout="wide",
)

st.title(f"📚 {APP_NAME}")
st.caption("在本地论文知识库中快速定位学术观点、出处、页码与原文证据")


def _port_is_in_use(port):
    """检查某个本地端口是否已经被占用。"""
    try:
        with socket.create_connection(
            ("127.0.0.1", port),
            timeout=0.25,
        ):
            return True
    except OSError:
        return False


def _find_free_port():
    """自动寻找一个当前可用的本地端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _pdf_server_is_running(port):
    """检查指定端口上的 PDF 服务是否已经启动。"""
    try:
        with socket.create_connection(
            ("127.0.0.1", port),
            timeout=0.25,
        ):
            return True
    except OSError:
        return False


@st.cache_resource
def ensure_pdf_server():
    """自动启动本地 PDF 浏览服务，并自动避开端口冲突。"""
    os.makedirs(PAPERS_FOLDER, exist_ok=True)

    # 优先使用 config.json 中设置的端口。
    # 如果该端口已经被其他程序占用，则自动寻找一个空闲端口。
    active_port = PDF_PORT

    if _port_is_in_use(active_port):
        active_port = _find_free_port()

    try:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(active_port),
                "--bind",
                "127.0.0.1",
                "--directory",
                PAPERS_FOLDER,
            ],
            cwd=PROJECT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        for _ in range(10):
            time.sleep(0.15)

            if _pdf_server_is_running(active_port):
                return active_port

    except Exception:
        return None

    return None


ACTIVE_PDF_PORT = ensure_pdf_server()
PDF_SERVER_OK = ACTIVE_PDF_PORT is not None


def build_pdf_url(source, page):
    encoded_name = quote(source)

    if ACTIVE_PDF_PORT is None:
        return "#"

    return (
        f"http://127.0.0.1:{ACTIVE_PDF_PORT}/"
        f"{encoded_name}#page={page}"
    )


def extract_successful_reference_matches(trace_results):
    """只保留真正匹配成功的完整参考文献条目。"""
    matched_blocks = []

    for result in trace_results:
        if result.get("status") != "ok":
            continue

        answer = result.get("answer", "") or ""
        source = result.get("source", "")

        # reference_test.py 的输出以 "### 引用..." 分块。
        parts = answer.split("### 引用")
        for part in parts[1:]:
            block = "### 引用" + part.strip()
            if "匹配结果：** 匹配成功" not in block and "匹配结果： 匹配成功" not in block:
                continue

            # 只保留“完整参考文献”这一项，界面尽量简洁。
            marker = "**完整参考文献：**"
            if marker not in block:
                continue

            ref_text = block.split(marker, 1)[1]
            # 到下一个字段前截断。
            for stop_marker in ["**PDF页码：**", "**说明：**"]:
                if stop_marker in ref_text:
                    ref_text = ref_text.split(stop_marker, 1)[0]
            ref_text = ref_text.strip()

            if ref_text and ref_text != "未找到":
                matched_blocks.append(
                    f"**完整参考文献：** {ref_text}"
                )

    # 去重，保持原顺序。
    unique = []
    seen = set()
    for item in matched_blocks:
        key = item.lower()
        if key not in seen:
            unique.append(item)
            seen.add(key)

    return unique


def insert_reference_matches(answer, matches):
    """
    简化展示：
    1. 删除单独的“建议继续追溯的参考文献”章节；
    2. 将匹配成功的完整参考文献直接放到“最相关证据”中；
    3. 找不到完整参考文献时不额外显示。
    """
    # 删除“建议继续追溯的参考文献”章节，但保留后面的“使用建议”。
    answer = re.sub(
        r"\n*## 建议继续追溯的参考文献.*?(?=\n## 使用建议|\Z)",
        "",
        answer,
        flags=re.S,
    )

    if not matches:
        return answer.strip()

    refs_block = (
        "\n\n### 相关参考文献\n\n"
        + "\n\n".join(matches)
        + "\n"
    )

    usage_heading = "## 使用建议"
    if usage_heading in answer:
        before, after = answer.split(usage_heading, 1)
        return (
            before.rstrip()
            + refs_block
            + "\n"
            + usage_heading
            + after
        )

    return answer.rstrip() + refs_block


# =========================================================
# 2. 论文上传
# =========================================================
with st.expander("📥 添加新论文到知识库"):
    st.write("上传 PDF 后，系统会自动保存到 papers 文件夹，并加入本地向量知识库。")

    uploaded_file = st.file_uploader("选择 PDF 文件", type=["pdf"])

    if uploaded_file is not None:
        st.info(f"已选择：{uploaded_file.name}")

        if st.button("➕ 加入知识库", type="primary"):
            os.makedirs(PAPERS_FOLDER, exist_ok=True)
            save_path = os.path.join(PAPERS_FOLDER, uploaded_file.name)

            if os.path.exists(save_path):
                st.warning("papers 文件夹中已经存在同名 PDF。")
            else:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success("PDF 已保存，正在加入知识库……")

                with st.spinner("正在解析 PDF 并生成向量，请稍候……"):
                    result = subprocess.run(
                        [sys.executable, os.path.join(PROJECT_DIR, "app.py")],
                        cwd=PROJECT_DIR,
                        capture_output=True,
                        text=True,
                    )

                if result.returncode == 0:
                    st.success("✅ 新论文已经成功加入知识库！")
                    with st.expander("查看入库运行记录"):
                        st.code(result.stdout)

                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.info("请点击下方按钮刷新网页，刷新后即可检索新论文。")

                    if st.button("🔄 刷新知识库"):
                        st.rerun()
                else:
                    st.error("论文入库过程中出现错误。")
                    st.code(result.stderr)


# =========================================================
# 3. 读取知识库
# =========================================================
@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_FOLDER)
    return client.get_collection(name=COLLECTION_NAME)


collection = load_collection()


def get_publication_year(source):
    """
    从文件名中提取四位年份。
    若出现多个合法年份，取最后一个；没有年份则排在最后。
    """
    years = re.findall(r"(?<!\\d)((?:19|20)\\d{2})(?!\\d)", source)
    if not years:
        return 0
    return int(years[-1])


@st.cache_data
def get_sources():
    data = collection.get(include=["metadatas"])

    source_set = {
        metadata["source"]
        for metadata in data["metadatas"]
        if metadata and metadata.get("source")
    }

    return sorted(
        source_set,
        key=lambda source: (
            -get_publication_year(source),
            source.lower(),
        ),
    )


def render_library_directory(sources):
    """显示当前知识库目录，点击文献名打开 PDF 第1页。"""
    st.subheader("📚 当前文献库目录")
    st.caption(
        f"共 {len(sources)} 篇论文，按文件名中的年份从近到远排列。"
        "点击文献名可直接打开 PDF。"
    )

    if not PDF_SERVER_OK:
        st.warning(
            f"PDF 浏览服务未能自动启动（端口 {PDF_PORT}）。"
            "请重新启动应用后再试。"
        )

    if not sources:
        st.info("当前文献库中还没有论文。")
        return

    for i, source in enumerate(sources, start=1):
        year = get_publication_year(source)

        col_num, col_title = st.columns([0.08, 0.92])

        with col_num:
            st.write(f"{i}.")

        with col_title:
            pdf_path = os.path.join(PAPERS_FOLDER, source)
            label = f"{source}"
            if year:
                label = f"{year} · {source}"

            if os.path.exists(pdf_path):
                st.markdown(
                    f'<a href="{build_pdf_url(source, 1)}" '
                    f'target="_blank">{label}</a>',
                    unsafe_allow_html=True,
                )
            else:
                st.write(label)
                st.caption("未找到原始 PDF 文件。")

    st.divider()


sources = get_sources()
paper_count = len(sources)
chunk_count = collection.count()

col1, col2 = st.columns(2)

with col1:
    show_library = st.query_params.get("library", "0")
    library_href = "?library=0" if show_library == "1" else "?library=1"

    st.markdown(
        f"""
        <a href="{library_href}" target="_self"
           style="text-decoration:none;color:inherit;">
            <div style="
                border:1px solid rgba(128,128,128,0.25);
                border-radius:0.5rem;
                padding:0.75rem 1rem;
                min-height:88px;
            ">
                <div style="font-size:0.875rem;color:rgba(120,120,120,0.95);">
                    论文数量（点击查看目录）
                </div>
                <div style="font-size:2rem;font-weight:600;line-height:1.35;">
                    {paper_count}
                </div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.metric("文本块数量", chunk_count)

if st.query_params.get("library", "0") == "1":
    render_library_directory(sources)

st.divider()


# =========================================================
# 4. 检索设置
# =========================================================
st.subheader("检索设置")

mode = st.radio(
    "选择检索模式",
    ["全库检索", "指定文献检索"],
    horizontal=True,
)

source_filter = None
if mode == "指定文献检索":
    source_filter = st.selectbox("选择目标论文", sources)

query = st.text_area(
    "需要定位的学术观点",
    placeholder="例如：中国大陆城市居民的阶层意识受什么因素影响？",
    height=120,
)

search_button = st.button(
    "🔍 开始检索",
    type="primary",
    use_container_width=True,
)


# =========================================================
# 5. 执行检索，并保存到 session_state
# =========================================================
if search_button:
    if not query.strip():
        st.warning("请输入需要查找的学术观点。")
        st.stop()

    with st.spinner("正在知识库中检索相关文献……"):
        retrieved = retrieve(
            query=query,
            source_filter=source_filter,
        )

    if not retrieved:
        st.warning("当前知识库没有找到相关结果。")
        st.stop()

    context = build_context(retrieved)

    with st.spinner("DeepSeek 正在核对并整理最相关证据……"):
        try:
            answer = analyze_evidence(query, context)

            # 自动追溯当前证据中明确出现的作者—年份引用。
            # 只有真正匹配到完整参考文献时才插入结果；
            # 找不到时不额外显示任何内容。
            trace_results = trace_references(retrieved)
            reference_matches = extract_successful_reference_matches(
                trace_results
            )
            answer = insert_reference_matches(
                answer,
                reference_matches,
            )

        except Exception as e:
            # 主证据整理失败时仍按原逻辑报错。
            st.error("DeepSeek 调用失败。")
            st.exception(e)
            st.stop()

    st.session_state["rag_query"] = query
    st.session_state["rag_retrieved"] = retrieved
    st.session_state["rag_answer"] = answer


# =========================================================
# 6. 展示最近一次检索结果
# =========================================================
if "rag_retrieved" in st.session_state:
    retrieved = st.session_state["rag_retrieved"]
    answer = st.session_state.get("rag_answer", "")

    st.divider()
    st.subheader("🔎 原始检索位置")
    st.caption(
        "以下为 BGE-M3 找到的主要相关位置。"
        "点击“打开原文”可在浏览器中直接核对对应 PDF 页码；"
        "DeepSeek 会继续结合相邻文本筛选证据。"
    )

    main_results = [
        item for item in retrieved
        if item.get("type") == "main"
    ]

    for i, item in enumerate(main_results, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {item['source']}**")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(f"📄 PDF 第 {item['page']} 页")
            with col2:
                st.write(f"相似度 {item['similarity']:.4f}")

            pdf_path = os.path.join(PAPERS_FOLDER, item["source"])
            if os.path.exists(pdf_path):
                st.link_button(
                    f"📖 打开原文第 {item['page']} 页",
                    build_pdf_url(item["source"], item["page"]),
                )
            else:
                st.warning("未找到原始 PDF 文件。")

    st.divider()
    st.subheader("🤖 学术证据整理")
    st.markdown(answer)

