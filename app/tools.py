
"""
Tool untuk membaca status tiket IT Helpdesk
dari file data/mock_tickets.csv.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TICKETS_PATH = PROJECT_ROOT / "data" / "mock_tickets.csv"


def get_ticket_status(ticket_id: str) -> dict:
    """
    Mencari tiket berdasarkan ticket_id.

    Mengembalikan detail tiket jika ditemukan,
    atau pesan yang sesuai jika ID tidak tersedia.
    """

    # 1. Validasi ID tiket
    if not ticket_id or not ticket_id.strip():
        return {
            "status": "error",
            "intent": "ticket_status",
            "data": {
                "message": (
                    "Mohon sertakan ID tiket, "
                    "contohnya TICKET-0001."
                )
            },
        }

    # 2. Pastikan file data tersedia
    if not TICKETS_PATH.exists():
        return {
            "status": "error",
            "intent": "ticket_status",
            "data": {
                "message": "File data tiket tidak ditemukan."
            },
        }

    # 3. Baca data tiket
    df = pd.read_csv(TICKETS_PATH)

    # 4. Pastikan kolom yang diperlukan tersedia
    if not {"ticket_id", "status"}.issubset(df.columns):
        return {
            "status": "error",
            "intent": "ticket_status",
            "data": {
                "message": "Format file data tiket tidak sesuai."
            },
        }

    # 5. Cari tiket tanpa membedakan huruf besar/kecil
    ticket_id = ticket_id.strip().upper()

    matched = df[
        df["ticket_id"].astype(str).str.strip().str.upper()
        == ticket_id
    ]

    # 6. Jika tiket tidak ditemukan
    if matched.empty:
        return {
            "status": "not_found",
            "intent": "ticket_status",
            "data": {
                "ticket_id": ticket_id,
                "message": (
                    f"Tiket {ticket_id} tidak ditemukan. "
                    "Periksa kembali ID tiket Anda."
                ),
            },
        }

    # 7. Ambil data tiket yang ditemukan
    row = matched.iloc[0]

    ticket_details = {
        column: (
            None if pd.isna(row[column]) else str(row[column])
        )
        for column in df.columns
    }

    return {
        "status": "success",
        "intent": "ticket_status",
        "data": {
            "message": (
                f"Status tiket {ticket_id}: {row['status']}."
            ),
            "ticket": ticket_details,
        },
    }
