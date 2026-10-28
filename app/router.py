
"""
Router untuk menentukan apakah pertanyaan membutuhkan RAG,
pemeriksaan status tiket, atau berada di luar cakupan assistant.
"""

import re

from app.rag import answer_from_knowledge_base
from app.tools import get_ticket_status


# Kata kunci baseline untuk mengenali pertanyaan IT Helpdesk.
# Daftar ini dapat disesuaikan dengan isi knowledge base.
FAQ_KEYWORDS = {
    "cuti", "absensi", "kehadiran", "izin",
    "password", "kata sandi", "akun", "login",
    "akses", "permission", "vpn", "internet",
    "wifi", "wi-fi", "jaringan", "email",
    "e-mail", "laptop", "komputer", "monitor",
    "mouse", "keyboard", "printer", "hardware",
    "software", "aplikasi", "instalasi", "install",
    "perangkat", "sistem", "hr", "portal",
    "karyawan", "helpdesk", "it", "error",
    "gangguan", "reset", "google analytics",
}


TICKET_KEYWORDS = {
    "status tiket",
    "cek tiket",
    "check ticket",
    "ticket status",
    "tiket saya",
    "status ticket",
    "nomor tiket",
    "id tiket",
}


def extract_ticket_id(query: str) -> str:
    """
    Mengambil ID tiket berformat TICKET-XXXX dari pertanyaan.
    Mengembalikan string kosong jika ID tidak ditemukan.
    """
    match = re.search(r"\bTICKET-\d{4}\b", query, re.IGNORECASE)

    if match:
        return match.group(0).upper()

    return ""


async def detect_intent(query: str) -> str:
    """
    Mengklasifikasikan pertanyaan menjadi:
      - ticket_status: pemeriksaan status tiket
      - faq: informasi atau kebijakan IT Helpdesk
      - unknown: di luar cakupan atau tidak jelas
    """
    if not query or not query.strip():
        return "unknown"

    normalized_query = query.casefold()

    # ID tiket merupakan petunjuk kuat bahwa user ingin
    # memeriksa tiket tertentu.
    if extract_ticket_id(query):
        return "ticket_status"

    # Kenali permintaan status tiket meskipun ID belum diberikan.
    if any(keyword in normalized_query for keyword in TICKET_KEYWORDS):
        return "ticket_status"

    # Baseline keyword untuk pertanyaan informasi IT Helpdesk.
    if any(keyword in normalized_query for keyword in FAQ_KEYWORDS):
        return "faq"

    return "unknown"


async def handle_query(query: str) -> dict:
    """
    Menentukan intent lalu meneruskan pertanyaan
    ke RAG atau Ticket Status Tool.
    """
    intent = await detect_intent(query)

    if intent == "ticket_status":
        ticket_id = extract_ticket_id(query)

        # Tool menangani ID kosong, tiket ditemukan,
        # dan tiket yang tidak ditemukan.
        result = get_ticket_status(ticket_id)

        # Pertahankan status asli dari tool.
        return result

    if intent == "faq":
        answer = await answer_from_knowledge_base(query)

        return {
            "status": "success",
            "intent": "faq",
            "data": answer,
        }

    return {
        "status": "error",
        "intent": "unknown",
        "data": {
            "message": (
                "Maaf, saya belum dapat mengenali permintaan tersebut. "
                "Silakan tanyakan informasi IT Helpdesk atau "
                "berikan ID tiket untuk memeriksa status tiket."
            )
        },
    }
