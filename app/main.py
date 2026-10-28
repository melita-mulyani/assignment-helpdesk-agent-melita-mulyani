
"""
Entry point aplikasi FastAPI — IT Helpdesk Assistant.

Alur:
1. Menerima pertanyaan dari user.
2. Meneruskan pertanyaan ke router.
3. Router memilih RAG, ticket tool, atau unknown.
4. Mengembalikan hasil dalam format terstruktur.
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.router import handle_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IT Helpdesk Assistant API")


class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Pertanyaan atau permintaan karyawan",
        examples=["Bagaimana cara mengajukan cuti?"],
    )


class AskResponse(BaseModel):
    status: str
    intent: str
    data: Any


@app.get("/health")
async def health():
    """Memastikan service berjalan."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Memproses pertanyaan melalui router."""
    try:
        query = request.query.strip()

        if not query:
            raise HTTPException(
                status_code=422,
                detail="Pertanyaan tidak boleh kosong.",
            )

        result = await handle_query(query)
        return AskResponse(**result)

    except HTTPException:
        raise

    except Exception:
        logger.exception("Terjadi kesalahan saat memproses permintaan.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Terjadi kesalahan saat memproses permintaan. "
                "Silakan coba kembali atau hubungi IT Helpdesk."
            ),
        )
