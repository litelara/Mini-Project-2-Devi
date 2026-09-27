"""
Versi web dari RAG chatbot, dibangun dengan Streamlit.

File ini hanya mengurus TAMPILAN. Seluruh "mesin" chatbot (memuat dokumen,
vector store, RAG chain) tetap ada di rag_chatbot.py dan dipinjam lewat import.

Cara jalanin di laptop:
    streamlit run app.py
(bukan "python app.py")
"""

import os

import streamlit as st
from dotenv import load_dotenv

from rag_chatbot import (
    KNOWLEDGE_DIR,
    SYSTEM_PROMPT_PATH,
    TOP_K,
    buat_model,
    muat_dokumen,
    bangun_vectorstore,
    muat_system_prompt,
    buat_rag_chain,
)


# ============================================================
# 1. PENGATURAN HALAMAN
# ============================================================
# Wajib jadi perintah Streamlit pertama: judul tab browser dan ikonnya.

st.set_page_config(
    page_title="Helmet Papa | Customer Care",
    page_icon="🪖",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 980px; padding: 2rem 1.5rem 5rem; }
    :root {
        --hp-blue: #2563eb;
        --hp-blue-light: #60a5fa;
        --hp-blue-pale: #eff6ff;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 50% -18%, rgba(59, 130, 246, .10), transparent 46%),
            var(--background-color);
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(59, 130, 246, .12);
    }
    [data-testid="stChatMessage"] {
        padding: .55rem .8rem;
        border: 1px solid rgba(59, 130, 246, .12);
        border-radius: 18px;
        background: rgba(59, 130, 246, .035);
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 16px;
        transition: border-color .2s ease, box-shadow .2s ease;
    }
    [data-testid="stChatInput"] {
        border-color: rgba(37, 99, 235, .32);
        box-shadow: 0 8px 28px rgba(37, 99, 235, .08);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, .14);
    }
    [data-testid="stButton"] button {
        border-radius: 13px;
        border-color: rgba(59, 130, 246, .2);
        transition: transform .18s ease, box-shadow .18s ease,
                    border-color .18s ease, background .18s ease;
    }
    [data-testid="stButton"] button:hover {
        border-color: rgba(37, 99, 235, .58);
        color: #1d4ed8;
        box-shadow: 0 7px 20px rgba(37, 99, 235, .12);
        transform: translateY(-1px);
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.1rem;
        margin: .35rem 0 1.35rem;
        border: 1px solid rgba(147, 197, 253, .28);
        border-radius: 26px;
        background:
            radial-gradient(ellipse at 88% 12%, rgba(125, 211, 252, .32), transparent 30%),
            radial-gradient(ellipse at 10% 110%, rgba(96, 165, 250, .36), transparent 38%),
            linear-gradient(125deg, #172554 0%, #1d4ed8 54%, #0284c7 100%);
        color: #fff;
        box-shadow: 0 20px 48px rgba(37, 99, 235, .22);
    }
    .hero::after {
        position: absolute;
        top: -42px;
        right: -34px;
        width: 190px;
        height: 190px;
        border: 1px solid rgba(255, 255, 255, .16);
        border-radius: 50%;
        box-shadow: 0 0 0 24px rgba(255, 255, 255, .045),
                    0 0 0 50px rgba(255, 255, 255, .035);
        content: "";
        pointer-events: none;
    }
    .hero .eyebrow {
        margin: 0 0 .55rem;
        color: #bfdbfe;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 { margin: 0; color: #fff; font-size: clamp(1.8rem, 4vw, 2.5rem); }
    .hero p { margin: .6rem 0 0; color: #dbeafe; font-size: 1rem; }
    .topic-card {
        min-height: 150px;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(59, 130, 246, .14);
        border-radius: 19px;
        background: linear-gradient(145deg, rgba(59, 130, 246, .075), rgba(14, 165, 233, .025));
        box-shadow: 0 7px 22px rgba(30, 64, 175, .045);
        transition: transform .2s ease, border-color .2s ease,
                    box-shadow .2s ease;
    }
    .topic-card:hover {
        transform: translateY(-3px);
        border-color: rgba(37, 99, 235, .38);
        box-shadow: 0 13px 28px rgba(37, 99, 235, .12);
    }
    .topic-icon { margin-bottom: .45rem; font-size: 1.35rem; }
    .topic-title { margin: 0; font-weight: 700; }
    .topic-copy { margin: .3rem 0 0; color: #64748b; font-size: .86rem; }
    .section-label {
        margin: 1.45rem 0 .65rem;
        font-size: 1.02rem;
        font-weight: 700;
    }
    .trust-note {
        margin-top: 1.5rem;
        padding: .8rem 1rem;
        border: 1px solid rgba(59, 130, 246, .13);
        border-left: 3px solid #3b82f6;
        border-radius: 0 12px 12px 0;
        background: linear-gradient(100deg, rgba(59, 130, 246, .075), rgba(14, 165, 233, .025));
        color: #64748b;
        font-size: .88rem;
    }
    @media (max-width: 640px) {
        .block-container { padding: 1.2rem .9rem 4rem; }
        .hero { padding: 1.4rem 1.25rem; border-radius: 19px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. CEK API KEY
# ============================================================
# Di laptop, GROQ_API_KEY dibaca dari file .env.
# Di Streamlit Cloud, GROQ_API_KEY diisi lewat menu Secrets, dan Streamlit
# otomatis menjadikannya environment variable. Jadi kode yang sama ini
# jalan di dua tempat tanpa perlu diubah.

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY belum diisi. Cek file .env (di laptop) "
        "atau menu Secrets (di Streamlit Cloud)."
    )
    st.stop()


# ============================================================
# 3. SIAPKAN MESIN CHATBOT (sekali saja, lalu disimpan)
# ============================================================
# Streamlit menjalankan ulang SELURUH file ini dari atas setiap kali
# pengguna berinteraksi (misalnya mengirim pertanyaan).
# @st.cache_resource membuat fungsi di bawah ini cukup dijalankan SEKALI.
# Hasilnya disimpan, lalu dipakai ulang, sehingga dokumen tidak dimuat
# ulang dan vector store tidak dibangun ulang di setiap pertanyaan.

@st.cache_resource(show_spinner="Menyiapkan chatbot, mohon tunggu sebentar...")
def siapkan_chatbot():
    model = buat_model()
    dokumen = muat_dokumen(KNOWLEDGE_DIR)
    vectorstore = bangun_vectorstore(dokumen)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    system_prompt = muat_system_prompt(SYSTEM_PROMPT_PATH)
    return buat_rag_chain(retriever, model, system_prompt)


try:
    rag_chain = siapkan_chatbot()
    chatbot_error = None
except (FileNotFoundError, RuntimeError, OSError) as error:
    rag_chain = None
    chatbot_error = str(error)


# ============================================================
# 4. BUKU CATATAN PERCAKAPAN
# ============================================================
# st.session_state adalah tempat menyimpan data yang tidak ikut hilang
# saat file ini dijalankan ulang. Di sini dipakai untuk mencatat riwayat
# percakapan: siapa yang bicara ("user" atau "assistant") dan isinya.
# Sama saja dengan menjaga percakapan terus muncul di atas chat baru

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

if "pertanyaan_cepat" not in st.session_state:
    st.session_state.pertanyaan_cepat = None


# ============================================================
# 5. TAMPILAN
# ============================================================

with st.sidebar:
    st.markdown("## 🪖 Helmet Papa")
    st.caption("CUSTOMER CARE VIRTUAL")
    st.divider()
    st.subheader("Kami bisa bantu")
    st.write(
        "Informasi produk, panduan ukuran, pesanan, garansi, retur, "
        "penukaran, dan perawatan helm."
    )
    st.caption(
        "Jawaban mengacu pada panduan dan informasi Helmet Papa yang tersedia."
    )
    st.divider()
    if st.button("Mulai percakapan baru"):
        st.session_state.riwayat = []
        st.session_state.pertanyaan_cepat = None
        st.rerun()

st.markdown(
    """
    <div class="hero">
        <p class="eyebrow">HELMET PAPA · CUSTOMER CARE</p>
        <h1>Halo, Bro/Sis! 👋</h1>
        <p>Ada yang bisa kami bantu seputar helm dan layanan Helmet Papa?</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if chatbot_error:
    st.error("Layanan chat belum siap.")
    st.info(
        "Periksa konfigurasi GROQ_API_KEY dan pastikan folder `knowledge_docs/` "
        "berisi setidaknya satu dokumen PDF."
    )
    with st.expander("Detail masalah"):
        st.code(chatbot_error)
    st.stop()

if not st.session_state.riwayat:
    st.markdown(
        '<p class="section-label">Pilih topik yang ingin ditanyakan</p>',
        unsafe_allow_html=True,
    )
    topik = st.columns(3)
    kartu_topik = [
        ("📏", "Ukuran & produk", "Cari panduan memilih helm yang pas."),
        ("🛡️", "Garansi & retur", "Tanyakan syarat dan alur layanan."),
        ("✨", "Perawatan helm", "Cari tips menjaga helm tetap terawat."),
    ]
    for kolom, (ikon, judul, deskripsi) in zip(topik, kartu_topik):
        with kolom:
            st.markdown(
                f"""
                <div class="topic-card">
                    <div class="topic-icon">{ikon}</div>
                    <p class="topic-title">{judul}</p>
                    <p class="topic-copy">{deskripsi}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<p class="section-label">Pertanyaan yang sering ditanyakan</p>',
        unsafe_allow_html=True,
    )
    pertanyaan_cepat = [
        "Panduan memilih ukuran helm?",
        "Syarat klaim garansi atau retur?",
        "Cara merawat helm?",
    ]
    kolom = st.columns(len(pertanyaan_cepat))
    for index, teks in enumerate(pertanyaan_cepat):
        if kolom[index].button(teks, key=f"prompt-{index}", use_container_width=True):
            st.session_state.pertanyaan_cepat = teks
            st.rerun()
else:
    st.markdown(
        '<p class="section-label">Percakapan Anda</p>',
        unsafe_allow_html=True,
    )

# Tampilkan ulang seluruh riwayat percakapan dari buku catatan.
for pesan in st.session_state.riwayat:
    with st.chat_message(pesan["role"]):
        st.markdown(pesan["isi"])


# ============================================================
# 6. TANYA JAWAB
# ============================================================

pertanyaan = st.chat_input("Tulis pertanyaan tentang produk atau layanan Helmet Papa...")
pertanyaan = pertanyaan or st.session_state.pertanyaan_cepat
st.session_state.pertanyaan_cepat = None

if pertanyaan and pertanyaan.strip():
    pertanyaan = pertanyaan.strip()
    # Tampilkan pertanyaan, lalu catat ke buku catatan.
    with st.chat_message("user"):
        st.markdown(pertanyaan)
    st.session_state.riwayat.append({"role": "user", "isi": pertanyaan})

    # Minta jawaban ke mesin RAG. .stream() + st.write_stream() membuat
    # jawaban muncul bertahap, kata demi kata, seperti sedang diketik.
    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban di dokumen..."):
            try:
                jawaban = st.write_stream(rag_chain.stream(pertanyaan))
            except (ConnectionError, OSError, RuntimeError, ValueError) as error:
                jawaban = "Maaf Kak, jawaban belum dapat dibuat saat ini. Silakan coba lagi."
                st.error(f"Terjadi kendala saat memproses pertanyaan: {error}")
    st.session_state.riwayat.append({"role": "assistant", "isi": jawaban})

st.markdown(
    """
    <div class="trust-note">
        💬 Jawaban diberikan berdasarkan informasi yang tersedia. Untuk proses
        klaim atau kendala pesanan khusus, tim admin Helmet Papa siap membantu.
    </div>
    """,
    unsafe_allow_html=True,
)
