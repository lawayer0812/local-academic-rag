import os
import hashlib
import pymupdf
import chromadb

from sentence_transformers import SentenceTransformer


# =========================================================
# 基础设置
# =========================================================

PDF_FOLDER = "papers"
CHROMA_FOLDER = "chroma_db"

COLLECTION_NAME = "academic_papers"
EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 200


# =========================================================
# 1. 文本切块
# =========================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_SIZE:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# =========================================================
# 2. 为 Chunk 创建稳定 ID
# =========================================================

def make_chunk_id(source, page, chunk_index):

    raw_id = (
        f"{source}|{page}|{chunk_index}"
    )

    return hashlib.md5(
        raw_id.encode("utf-8")
    ).hexdigest()


# =========================================================
# 3. 读取一篇 PDF
# =========================================================

def process_pdf(pdf_path, filename):

    document = pymupdf.open(pdf_path)

    chunks = []

    for page_number in range(
        len(document)
    ):

        page = document[page_number]

        text = page.get_text(
            "text"
        ).strip()

        if not text:
            continue

        page_chunks = split_text(
            text
        )

        for chunk_index, chunk in enumerate(
            page_chunks,
            start=1
        ):

            chunks.append({
                "id": make_chunk_id(
                    filename,
                    page_number + 1,
                    chunk_index
                ),

                "text": chunk,

                "metadata": {
                    "source": filename,
                    "page": page_number + 1,
                    "chunk_index": chunk_index
                }
            })

    document.close()

    return chunks


# =========================================================
# 4. 加载 BGE-M3
# =========================================================

print("\n正在加载 BGE-M3...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("BGE-M3 加载完成。")


# =========================================================
# 5. 连接 Chroma
# =========================================================

client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)

try:

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"已有知识库加载完成，"
        f"当前共有 {collection.count()} 个文本块。"
    )

except:

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    print(
        "未发现已有知识库，"
        "已创建新的知识库。"
    )


# =========================================================
# 6. 扫描 papers 文件夹
# =========================================================

pdf_files = sorted([
    filename
    for filename in os.listdir(PDF_FOLDER)
    if filename.lower().endswith(".pdf")
])

print("\n" + "=" * 70)
print("扫描论文文件夹")
print("=" * 70)

print(
    f"\npapers 文件夹中共有 "
    f"{len(pdf_files)} 篇 PDF。"
)


# =========================================================
# 7. 找出已经入库的论文
# =========================================================

existing_sources = set()

if collection.count() > 0:

    existing_data = collection.get(
        include=["metadatas"]
    )

    for metadata in existing_data[
        "metadatas"
    ]:

        if (
            metadata
            and "source" in metadata
        ):

            existing_sources.add(
                metadata["source"]
            )


print(
    f"知识库中已经存在 "
    f"{len(existing_sources)} 篇论文。"
)


# =========================================================
# 8. 找出新增 PDF
# =========================================================

new_files = [
    filename
    for filename in pdf_files
    if filename not in existing_sources
]

skipped_files = [
    filename
    for filename in pdf_files
    if filename in existing_sources
]


print(
    f"已存在，将跳过："
    f"{len(skipped_files)} 篇"
)

print(
    f"发现新增论文："
    f"{len(new_files)} 篇"
)


# =========================================================
# 9. 如果没有新增论文
# =========================================================

if len(new_files) == 0:

    print("\n" + "=" * 70)

    print(
        "没有发现需要新增的论文。"
    )

    print(
        f"当前知识库共有 "
        f"{collection.count()} 个文本块。"
    )

    print("=" * 70)

    raise SystemExit


# =========================================================
# 10. 逐篇处理新增论文
# =========================================================

total_new_chunks = 0


for file_number, filename in enumerate(
    new_files,
    start=1
):

    print("\n" + "-" * 70)

    print(
        f"[{file_number}/{len(new_files)}] "
        f"正在处理："
    )

    print(filename)

    pdf_path = os.path.join(
        PDF_FOLDER,
        filename
    )

    # -----------------------------------------------------
    # 提取并切块
    # -----------------------------------------------------

    chunks = process_pdf(
        pdf_path,
        filename
    )

    if len(chunks) == 0:

        print(
            "没有提取到有效文本，"
            "已跳过。"
        )

        continue


    print(
        f"提取完成："
        f"{len(chunks)} 个文本块。"
    )


    # -----------------------------------------------------
    # Embedding
    # -----------------------------------------------------

    texts = [
        item["text"]
        for item in chunks
    ]

    print(
        "正在生成 Embedding..."
    )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )


    # -----------------------------------------------------
    # 写入 Chroma
    # -----------------------------------------------------

    ids = [
        item["id"]
        for item in chunks
    ]

    metadatas = [
        item["metadata"]
        for item in chunks
    ]


    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )


    total_new_chunks += len(chunks)


    print(
        f"已成功加入知识库："
        f"{len(chunks)} 个文本块。"
    )


# =========================================================
# 11. 最终统计
# =========================================================

final_sources = set()

final_data = collection.get(
    include=["metadatas"]
)

for metadata in final_data[
    "metadatas"
]:

    if (
        metadata
        and "source" in metadata
    ):

        final_sources.add(
            metadata["source"]
        )


print("\n" + "=" * 70)

print("增量入库完成")

print("=" * 70)

print(
    f"\n本次新增论文："
    f"{len(new_files)} 篇"
)

print(
    f"本次新增文本块："
    f"{total_new_chunks} 个"
)

print(
    f"知识库论文总数："
    f"{len(final_sources)} 篇"
)

print(
    f"知识库文本块总数："
    f"{collection.count()} 个"
)

print("\n完成。")