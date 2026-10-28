"""
Modul RAG — mengubah knowledge base menjadi embedding, menyimpannya ke FAISS,
melakukan semantic search, lalu menghasilkan jawaban menggunakan LLM.

Alur ini serupa dengan yang sudah dibangun di Assignment 3 (Semantic Search &
LLM Application) dan Sesi 11 (Building a Simple RAG System) — di assignment ini,
alur tersebut dibungkus menjadi fungsi yang dipanggil oleh router.py.
"""

"""
Modul RAG untuk IT Helpdesk Assistant.

Alur:
1. Membaca knowledge base dari CSV.
2. Membuat embedding untuk setiap FAQ.
3. Menyimpan embedding ke index FAISS.
4. Mencari FAQ yang relevan dengan pertanyaan user.
5. Menggunakan Gemini untuk menyusun jawaban berdasarkan FAQ.
"""

import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


# Memuat konfigurasi dari file .env di root project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Pastikan API key tersedia sebelum menghubungi Gemini.
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY or API_KEY == "your_api_key_here":
    raise ValueError(
        "GEMINI_API_KEY belum diatur. Periksa file .env di root project."
    )

_client = genai.Client(api_key=API_KEY)

# Model embedding dan model untuk menghasilkan jawaban.
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.5-flash-lite"

KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT / "data" / "it_helpdesk_knowledge_base.csv"
)

# Variabel untuk menyimpan index dan dokumen agar tidak dibuat
# ulang pada setiap pertanyaan.
_index = None
_documents = None
_df = None


def _get_embedding(text: str, task_type: str) -> np.ndarray:
    """
    Mengubah teks menjadi embedding atau representasi angka.

    task_type:
    - RETRIEVAL_DOCUMENT untuk dokumen FAQ.
    - RETRIEVAL_QUERY untuk pertanyaan user.
    """
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type
        ),
    )

    vector = np.asarray(
        result.embeddings[0].values,
        dtype=np.float32,
    )

    # Normalisasi agar perbandingan vektor lebih konsisten.
    vector = vector.reshape(1, -1)
    faiss.normalize_L2(vector)

    return vector


def build_index():
    """
    Membaca knowledge base, membuat embedding, dan membangun index FAISS.

    Fungsi ini cukup dipanggil sekali selama aplikasi berjalan.
    """
    global _index, _documents, _df

    # Hindari membangun ulang index jika sudah tersedia.
    if _index is not None:
        return _index

    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base tidak ditemukan: {KNOWLEDGE_BASE_PATH}"
        )

    _df = pd.read_csv(KNOWLEDGE_BASE_PATH)

    required_columns = {"faq_id", "category", "question", "answer"}
    missing_columns = required_columns - set(_df.columns)

    if missing_columns:
        raise ValueError(
            f"Kolom CSV tidak lengkap: {sorted(missing_columns)}"
        )

    if _df.empty:
        raise ValueError("Knowledge base tidak memiliki data FAQ.")

    _documents = []

    # Gabungkan pertanyaan, kategori, dan jawaban agar konteks FAQ
    # lebih lengkap ketika dicari secara semantik.
    for _, row in _df.iterrows():
        document = (
            f"Kategori: {row['category']}\n"
            f"Pertanyaan: {row['question']}\n"
            f"Jawaban: {row['answer']}"
        )

        _documents.append(document)

    # Buat embedding untuk setiap dokumen FAQ.
    vectors = []

    for i, document in enumerate(_documents, start=1):
        print(f"Membuat embedding FAQ {i}/{len(_documents)}")

        vector = _get_embedding(
            document,
            task_type="RETRIEVAL_DOCUMENT",
        )
        vectors.append(vector)

    # Gabungkan seluruh vektor menjadi satu matriks NumPy.
    embedding_matrix = np.vstack(vectors).astype(np.float32)

    # IndexFlatIP menghitung inner product.
    # Karena vektor sudah dinormalisasi, hasilnya setara
    # dengan cosine similarity.
    _index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    _index.add(embedding_matrix)

    print(f"Index FAISS selesai. Jumlah FAQ: {_index.ntotal}")

    return _index


def semantic_search(query: str, top_k: int = 3) -> list[dict]:
    """
    Mencari FAQ yang paling relevan dengan pertanyaan user.

    Hasil dikembalikan dalam urutan relevansi tertinggi.
    """
    if not query or not query.strip():
        return []

    index = build_index()

    # Batasi jumlah hasil agar tidak melebihi jumlah dokumen.
    k = max(1, min(top_k, index.ntotal))

    query_vector = _get_embedding(
        query,
        task_type="RETRIEVAL_QUERY",
    )

    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = _df.iloc[int(idx)]

        results.append(
            {
                "faq_id": int(row["faq_id"]),
                "category": str(row["category"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "score": float(score),
            }
        )

    return results


async def answer_from_knowledge_base(query: str) -> str:
    """
    Mengambil FAQ yang relevan lalu meminta Gemini menyusun jawaban.

    Gemini diarahkan untuk menggunakan konteks FAQ saja.
    """
    retrieved_docs = semantic_search(query, top_k=3)

    if not retrieved_docs:
        return (
            "Maaf, saya belum menemukan informasi yang relevan "
            "di knowledge base. Silakan hubungi tim IT Helpdesk."
        )

    # Susun konteks dari hasil retrieval.
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        context_parts.append(
            f"FAQ {i}\n"
            f"Kategori: {doc['category']}\n"
            f"Pertanyaan: {doc['question']}\n"
            f"Jawaban: {doc['answer']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
PERSONA:
Kamu adalah asisten IT Helpdesk internal perusahaan.
Jawab dengan ramah, jelas, dan profesional dalam bahasa Indonesia.

TASK:
Jawab pertanyaan karyawan berdasarkan konteks FAQ yang diberikan.

CONTEXT:
{context}

USER QUESTION:
{query}

RULES:
1. Gunakan hanya informasi yang didukung oleh konteks FAQ.
2. Jangan mengarang kebijakan, prosedur, atau informasi tambahan.
3. Jika konteks tidak cukup untuk menjawab pertanyaan,
   katakan bahwa informasi belum ditemukan di knowledge base
   dan sarankan menghubungi tim IT Helpdesk.
4. Jawab secara ringkas dan mudah dipahami.
5. Jangan menampilkan skor similarity atau detail teknis internal.

ANSWER:
""".strip()

    response = _client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    if not response.text:
        return (
            "Maaf, saya belum dapat menghasilkan jawaban saat ini. "
            "Silakan coba kembali atau hubungi tim IT Helpdesk."
        )

    return response.text.strip()