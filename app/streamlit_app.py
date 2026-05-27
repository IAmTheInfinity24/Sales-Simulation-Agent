import time
import streamlit as st
import pandas as pd

from graph.agentic_graph import app as sales_app
from services.pinecone_service import (
    get_pinecone_index,
    upsert_vectors,
    describe_stats,
    DEFAULT_NAMESPACE,
)
from services.embedding_service import generate_embeddings_batch
from utils.product_formatter import build_products_texts_batch
from agents.feedback_agent import process_feedback
from memory.user_store import profile_summary
from tools.catalog_sync import CatalogSync

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Sales Recommendation Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# STYLES
# =========================================================
st.markdown("""
<style>
/* ── global ── */
.stApp { background-color: #05081a; color: #e2e8f0; }
.block-container { padding-top: 0 !important; }

/* ── top bar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 24px 13px;
    border-bottom: 1px solid rgba(0,245,255,0.12);
    margin-bottom: 22px;
}
.topbar-title { font-size: 20px; font-weight: 600; color: #00F5FF; letter-spacing: -0.3px; }
.topbar-sub   { font-size: 11px; color: #475569; margin-top: 3px; }
.topbar-badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.topbar-badge {
    font-size: 11px; background: rgba(0,245,255,0.07);
    border: 1px solid rgba(0,245,255,0.18); color: #7dd3fc;
    border-radius: 20px; padding: 4px 12px;
}
.badge-memory {
    font-size: 11px; background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.25); color: #c4b5fd;
    border-radius: 20px; padding: 4px 12px;
}
.badge-boosted {
    font-size: 11px; background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25); color: #6ee7b7;
    border-radius: 20px; padding: 4px 12px;
}
.badge-fallback {
    font-size: 11px; background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25); color: #fcd34d;
    border-radius: 20px; padding: 4px 12px;
}
.badge-sync-warn {
    font-size: 11px; background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.30); color: #fcd34d;
    border-radius: 20px; padding: 4px 12px;
}

/* ── panel title ── */
.ptitle {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: #64748b;
    margin-bottom: 16px; display: flex; align-items: center; gap: 7px;
}

/* ── divider ── */
.pdiv { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 16px 0; }

/* ── stat pills ── */
.stat-row { display: flex; gap: 10px; margin-bottom: 16px; }
.stat-pill {
    flex: 1; background: #111c36;
    border: 1px solid rgba(0,245,255,0.09); border-radius: 10px; padding: 10px 13px;
}
.stat-pill .num { font-size: 20px; font-weight: 600; color: #00F5FF; }
.stat-pill .lbl { font-size: 10px; color: #64748b; margin-top: 2px; }

/* ── sync status box ── */
.sync-box {
    background: #0a1220; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 9px; padding: 10px 14px; margin-bottom: 12px;
    font-size: 11px; color: #64748b; line-height: 1.9;
}
.sync-box .sync-title { font-size: 11px; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.sync-ok   { color: #34d399; }
.sync-warn { color: #fbbf24; }

/* ── upload zone ── */
.upload-hint {
    border: 1px dashed rgba(0,245,255,0.18); border-radius: 9px;
    padding: 22px 14px; text-align: center;
    color: #475569; font-size: 12px; background: rgba(0,245,255,0.015);
    margin-bottom: 14px;
}
.upload-hint span { color: #64748b; font-weight: 500; font-size: 13px; display: block; margin-bottom: 5px; }

/* ── namespace pill ── */
.ns-box {
    background: #0a1220; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 7px; padding: 7px 11px;
    font-size: 12px; color: #7dd3fc; margin-bottom: 14px;
}

/* ── memory profile card ── */
.mem-card {
    background: #0d1a30; border: 1px solid rgba(167,139,250,0.15);
    border-radius: 10px; padding: 11px 14px; margin-bottom: 14px;
}
.mem-card .mem-title { font-size: 10px; font-weight: 600; color: #7c3aed;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 7px; }
.mem-stat { display: inline-block; background: #111c36;
    border-radius: 6px; padding: 4px 9px; font-size: 11px;
    color: #a78bfa; margin-right: 6px; margin-bottom: 5px; }
.mem-cats { font-size: 11px; color: #64748b; margin-top: 5px; }

/* ── intent tags ── */
.intent-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 12px; }
.intent-tag {
    background: #111c36; border: 1px solid rgba(0,245,255,0.22);
    color: #7dd3fc; border-radius: 20px;
    padding: 3px 11px; font-size: 11px; font-weight: 500;
}

/* ── AI explanation ── */
.explain-block {
    background: #0a1628; border-left: 3px solid #00F5FF;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    margin-bottom: 14px; font-size: 12px; color: #94a3b8; line-height: 1.75;
}
.explain-block b { color: #cbd5e1; }

/* ── signal grid ── */
.sig-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin: 8px 0; }
.sig-item { background: #0a1628; border-radius: 7px; padding: 7px 10px; }
.sig-lbl { font-size: 9px; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; }
.sig-val { font-size: 12px; color: #94a3b8; font-weight: 500; margin-top: 2px; }

/* ── product cards ── */
.prod-card {
    background: #111c36; border: 1px solid rgba(0,245,255,0.09);
    border-radius: 11px; padding: 13px 15px; margin-bottom: 4px;
    transition: border-color 0.15s;
}
.prod-card:hover { border-color: rgba(0,245,255,0.28); }
.prod-name  { font-size: 13px; font-weight: 600; color: #e2e8f0; margin-bottom: 3px; }
.prod-meta  { font-size: 11px; color: #64748b; margin-bottom: 5px; }
.prod-price { font-size: 13px; font-weight: 600; color: #34d399; }
.prod-rating{ font-size: 11px; color: #f59e0b; margin-left: 9px; }
.prod-desc  { font-size: 11px; color: #64748b; line-height: 1.55; margin-top: 7px; }
.disc-badge {
    background: #064e3b; color: #34d399; border-radius: 5px;
    padding: 1px 6px; font-size: 10px; font-weight: 600; margin-left: 7px;
}

/* ── feedback row ── */
.fb-row { display: flex; gap: 8px; margin-top: 8px; padding-top: 8px;
    border-top: 1px solid rgba(255,255,255,0.05); }

/* ── feedback saved banner ── */
.fb-saved {
    background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.2);
    border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;
    font-size: 12px; color: #6ee7b7;
}

/* ── empty state ── */
.empty-state {
    color: #334155; font-size: 12px; line-height: 1.8; padding-top: 6px;
}
.empty-state strong { color: #475569; }

/* ── streamlit widget overrides ── */
.stButton > button {
    background: rgba(0,245,255,0.07) !important;
    border: 1px solid rgba(0,245,255,0.25) !important;
    color: #00F5FF !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
.stButton > button:hover {
    background: rgba(0,245,255,0.14) !important;
    border-color: rgba(0,245,255,0.45) !important;
}
.stTextInput  > div > div > input,
.stNumberInput > div > div > input {
    background: #0a1220 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
    font-size: 13px !important;
}
.stTextInput  > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(0,245,255,0.38) !important;
    box-shadow: 0 0 0 2px rgba(0,245,255,0.07) !important;
}
label, .stTextInput label, .stNumberInput label {
    color: #64748b !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.stProgress > div > div > div > div { background-color: #00F5FF !important; }
.stAlert { border-radius: 8px !important; font-size: 13px !important; }
.stExpander { border: 1px solid rgba(0,245,255,0.08) !important; border-radius: 8px !important; }
.stExpander summary { color: #64748b !important; font-size: 12px !important; }
div[data-testid="stFileUploader"] > div {
    background: #0a1220 !important;
    border: 1px dashed rgba(0,245,255,0.18) !important;
    border-radius: 9px !important;
}
div[data-testid="stFileUploader"] label { color: #64748b !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE INIT
# =========================================================
for key, default in {
    "last_result"     : None,     # full graph response
    "feedback"        : {},       # {product_name: "like" | "dislike"}
    "feedback_saved"  : False,
    "last_persona_name": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =========================================================
# SERVICES
# =========================================================
index       = get_pinecone_index()
catalog_sync = CatalogSync()
sync_status  = catalog_sync.get_status()

# =========================================================
# TOP BAR
# =========================================================
# Build dynamic badge HTML
badges_html = '<span class="topbar-badge">⬤ &nbsp;Index live</span>'

if sync_status["needs_sync"]:
    badges_html += '<span class="badge-sync-warn">⚠ Catalog update detected</span>'

# Memory badge for last-run user (if any)
if st.session_state.last_persona_name:
    summary = profile_summary(st.session_state.last_persona_name)
    if summary:
        badges_html += (
            f'<span class="badge-memory">🧠 Memory · '
            f'{summary["interaction_count"]} sessions</span>'
        )

if st.session_state.last_result:
    strategy = st.session_state.last_result.get("retrieval_strategy", "standard")
    if strategy == "preference_boosted":
        badges_html += '<span class="badge-boosted">✦ Preference-boosted</span>'
    if st.session_state.last_result.get("fallback_applied"):
        badges_html += '<span class="badge-fallback">↩ Fallback applied</span>'

st.markdown(f"""
<div class="topbar">
    <div>
        <div class="topbar-title">AI Sales Recommendation Agent</div>
        <div class="topbar-sub">LangGraph &nbsp;·&nbsp; Pinecone &nbsp;·&nbsp; BGE Embeddings &nbsp;·&nbsp; LLaMA3 &nbsp;·&nbsp; Adaptive Memory</div>
    </div>
    <div class="topbar-badges">{badges_html}</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TWO-COLUMN LAYOUT
# =========================================================
col_catalog, col_reco = st.columns([1, 1.4], gap="large")


# =========================================================
# LEFT — CATALOG MANAGEMENT
# =========================================================
with col_catalog:
    st.markdown('<div class="ptitle">📦 &nbsp;Catalog management</div>', unsafe_allow_html=True)

    stats      = describe_stats(namespace=DEFAULT_NAMESPACE)
    total_vecs = stats.get("total_vector_count", 0)
    fullness   = round(stats.get("index_fullness", 0) * 100, 1)
    dim        = stats.get("dimension", 384)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill"><div class="num">{total_vecs:,}</div><div class="lbl">Products indexed</div></div>
        <div class="stat-pill"><div class="num">{dim}</div><div class="lbl">Vector dims</div></div>
        <div class="stat-pill"><div class="num">{fullness}%</div><div class="lbl">Index fullness</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sync status ──────────────────────────────────────────────────────
    st.markdown('<hr class="pdiv">', unsafe_allow_html=True)
    st.markdown('<div class="ptitle">🔄 &nbsp;Live catalog sync</div>', unsafe_allow_html=True)

    sync_color  = "sync-warn" if sync_status["needs_sync"] else "sync-ok"
    sync_icon   = "⚠" if sync_status["needs_sync"] else "✓"
    sync_label  = "Update detected — re-index recommended" if sync_status["needs_sync"] else "Index is up to date"
    auto_label  = "ON" if sync_status["auto_sync_enabled"] else "OFF"

    st.markdown(f"""
    <div class="sync-box">
        <div class="sync-title">Sync status</div>
        <span class="{sync_color}">{sync_icon} {sync_label}</span><br>
        Last synced: {sync_status["last_synced_at"]}<br>
        CSV modified: {sync_status["csv_modified_at"]}<br>
        Auto-sync: <b>{auto_label}</b>
    </div>
    """, unsafe_allow_html=True)

    auto_sync_toggle = st.toggle(
        "Enable auto-sync on CSV change",
        value=sync_status["auto_sync_enabled"],
        key="auto_sync_toggle",
    )
    if auto_sync_toggle != sync_status["auto_sync_enabled"]:
        catalog_sync.set_auto_sync(auto_sync_toggle)
        st.rerun()

    st.markdown('<hr class="pdiv">', unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload product catalog CSV",
        type=["csv"],
        help="Required columns: product, brand, category, sub_category, type, sale_price, market_price, rating, description",
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df = df.fillna("")
        before = len(df)
        df = df.drop_duplicates(subset=["product", "brand"])
        removed = before - len(df)

        st.success(f"**{len(df):,}** unique products · {removed} duplicates removed")

        with st.expander("Preview first 10 rows"):
            st.dataframe(df.head(10), use_container_width=True)

        if total_vecs > 0:
            st.warning(f"Index already has {total_vecs:,} vectors. Re-uploading will overwrite.")
            force = st.checkbox("Confirm re-upload")
        else:
            force = True

        if force:
            if st.button("Upload to Pinecone", key="upload_btn"):
                rows  = df.to_dict("records")
                total = len(rows)
                prog     = st.progress(0)
                stat_txt = st.empty()
                t0 = time.time()

                stat_txt.info("Building semantic product texts…")
                texts = build_products_texts_batch(rows)

                stat_txt.info(f"Generating embeddings for {total:,} products…")
                all_emb = generate_embeddings_batch(texts, is_query=False, show_progress=False)

                BATCH = 256
                vectors, upserted = [], 0

                for i, (row, emb) in enumerate(zip(rows, all_emb)):
                    vectors.append({
                        "id"    : str(row.get("index", i)),
                        "values": emb,
                        "metadata": {
                            "product"     : str(row.get("product",      "")),
                            "brand"       : str(row.get("brand",        "")),
                            "category"    : str(row.get("category",     "")),
                            "sub_category": str(row.get("sub_category", "")),
                            "type"        : str(row.get("type",         "")),
                            "sale_price"  : str(row.get("sale_price",   "")),
                            "market_price": str(row.get("market_price", "")),
                            "rating"      : str(row.get("rating",       "")),
                            "description" : str(row.get("description",  "")),
                        },
                    })
                    if len(vectors) >= BATCH:
                        upsert_vectors(vectors, namespace=DEFAULT_NAMESPACE)
                        upserted += len(vectors)
                        vectors = []
                        prog.progress(min(int((i + 1) / total * 95), 95))
                        stat_txt.info(f"Upserted {upserted:,} / {total:,}…")

                if vectors:
                    upsert_vectors(vectors, namespace=DEFAULT_NAMESPACE)
                    upserted += len(vectors)

                prog.progress(100)
                elapsed = round(time.time() - t0, 1)
                stat_txt.empty()

                # Mark sync state
                catalog_sync.mark_synced(upserted)

                st.success(f"{upserted:,} products uploaded in {elapsed}s")
                st.rerun()

    else:
        st.markdown("""
        <div class="upload-hint">
            <span>Drop your CSV here or click Browse</span>
            product · brand · category · sub_category · type<br>
            sale_price · market_price · rating · description
        </div>
        """, unsafe_allow_html=True)

        if total_vecs == 0:
            st.info("No catalog indexed yet. Upload a CSV to get started.")
        else:
            st.success(f"Catalog ready — {total_vecs:,} products in Pinecone.")

        # Auto-sync: trigger if toggle is on and CSV has changed
        if sync_status["auto_sync_enabled"] and sync_status["needs_sync"] and sync_status["csv_exists"]:
            st.info("Auto-sync: new version of products.csv detected. Re-upload the file to sync.")

    st.markdown('<hr class="pdiv">', unsafe_allow_html=True)
    st.markdown('<div class="ptitle">Namespace</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ns-box">{DEFAULT_NAMESPACE}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px; color:#334155; line-height:1.8;">
        Batch embedding enabled · BGE-small model<br>
        ~15–20× faster than row-by-row ingestion.<br>
        Re-uploading overwrites the existing namespace.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# RIGHT — CUSTOMER RECOMMENDATIONS
# =========================================================
with col_reco:
    st.markdown('<div class="ptitle">🎯 &nbsp;Customer recommendations</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        name   = st.text_input("Name",   placeholder="e.g. Rahul Sharma")
        age    = st.number_input("Age",  min_value=10, max_value=100, value=29)
    with f2:
        income    = st.text_input("Income",    placeholder="e.g. 12 LPA")
        interests = st.text_input("Interests", placeholder="fitness, travel, bike")

    purchase_history = st.text_input(
        "Purchase history",
        placeholder="protein bars, almond milk, oats, helmet",
    )

    # Show memory card if profile exists for this name
    if name.strip():
        summary = profile_summary(name.strip())
        if summary:
            top_cats = ", ".join(summary["top_categories"]) or "—"
            mem_html = (
                f'<div class="mem-card">'
                f'<div class="mem-title">🧠 Memory loaded</div>'
                f'<span class="mem-stat">{summary["interaction_count"]} sessions</span>'
                f'<span class="mem-stat">{summary["liked_count"]} liked products</span>'
            )
            if summary["has_embedding"]:
                mem_html += '<span class="mem-stat">✦ Preference vector active</span>'
            mem_html += f'<div class="mem-cats">Top categories: {top_cats}</div></div>'
            st.markdown(mem_html, unsafe_allow_html=True)

    run_btn = st.button("Generate recommendations", key="reco_btn")

    st.markdown('<hr class="pdiv">', unsafe_allow_html=True)

    # =========================================================
    # RUN PIPELINE
    # =========================================================
    if run_btn:
        if not name.strip():
            st.error("Please enter a customer name.")
            st.stop()
        if not interests.strip() and not purchase_history.strip():
            st.error("Add at least one interest or a purchase history item.")
            st.stop()

        persona = {
            "name"            : name.strip(),
            "age"             : int(age),
            "income"          : income.strip(),
            "interests"       : [x.strip() for x in interests.split(",")  if x.strip()],
            "purchase_history": [x.strip() for x in purchase_history.split(",") if x.strip()],
        }

        with st.spinner("Analysing persona…"):
            response = sales_app.invoke({"persona": persona})

        # Store for feedback
        st.session_state.last_result      = response
        st.session_state.last_persona_name = name.strip()
        st.session_state.feedback          = {}
        st.session_state.feedback_saved    = False

        st.rerun()   # re-render with top-bar badges updated

    # =========================================================
    # RENDER RESULTS (persisted in session_state)
    # =========================================================
    result = st.session_state.last_result
    if result is None:
        st.markdown("""
        <div class="empty-state">
            Fill in the customer persona above and hit
            <strong>Generate recommendations</strong>
            to see AI-ranked products with explanations.<br><br>
            Tip: the more interests and purchase history you add,
            the more precise the recommendations.<br>
            Feedback you give is remembered across sessions.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if result.get("error"):
        st.error(result["error"])
        st.stop()

    recommendations  = result.get("recommendations", [])
    ai_explanation   = result.get("ai_explanation", "")
    intents          = result.get("intents", [])
    customer_signals = result.get("customer_signals", {})
    strategy         = result.get("retrieval_strategy", "standard")
    fallback_applied = result.get("fallback_applied", False)
    confidence       = result.get("confidence_score", 0.0)

    # ── Intent tags ──────────────────────────────────────────────────────
    if intents:
        tags = "".join(f'<span class="intent-tag">#{i}</span>' for i in intents)
        st.markdown(f'<div class="intent-row">{tags}</div>', unsafe_allow_html=True)

    # ── Status badges ─────────────────────────────────────────────────────
    badge_row = ""
    if strategy == "preference_boosted":
        badge_row += '<span class="badge-boosted">✦ Preference-boosted results</span>&nbsp;'
    if fallback_applied:
        badge_row += '<span class="badge-fallback">↩ Fallback applied</span>&nbsp;'
    if badge_row:
        st.markdown(f'<div style="margin-bottom:10px">{badge_row}</div>', unsafe_allow_html=True)

    # ── Customer signals ──────────────────────────────────────────────────
    if customer_signals:
        fields = [
            ("Customer ID",   "customer_id"),
            ("City",          "city"),
            ("Loyalty pts",   "loyalty_points"),
            ("Avg order",     "avg_order_value"),
            ("Credit score",  "credit_score"),
            ("Churn risk",    "churn_risk"),
            ("Fav. category", "favorite_category"),
            ("Subscribed",    "subscription"),
        ]
        items = "".join(
            f'<div class="sig-item"><div class="sig-lbl">{lbl}</div>'
            f'<div class="sig-val">{customer_signals.get(key, "—")}</div></div>'
            for lbl, key in fields
            if customer_signals.get(key) not in ("", None)
        )
        if items:
            with st.expander("Customer signals from database"):
                st.markdown(f'<div class="sig-grid">{items}</div>', unsafe_allow_html=True)

    # ── AI explanation ────────────────────────────────────────────────────
    if ai_explanation:
        lines    = [l.strip() for l in ai_explanation.strip().splitlines() if l.strip()]
        rendered = ""
        for line in lines:
            if line and line[0].isdigit() and ". " in line:
                parts = line.split(". ", 1)
                if len(parts) == 2:
                    num, rest = parts
                    if ": " in rest:
                        pname, reason = rest.split(": ", 1)
                        rendered += f'<div><b>{num}. {pname}:</b> {reason}</div>'
                    else:
                        rendered += f'<div><b>{line}</b></div>'
                else:
                    rendered += f'<div>{line}</div>'
            else:
                rendered += f'<div>{line}</div>'
        st.markdown(f'<div class="explain-block">{rendered}</div>', unsafe_allow_html=True)

    # ── Feedback saved confirmation ───────────────────────────────────────
    if st.session_state.feedback_saved:
        liked_names    = [k for k, v in st.session_state.feedback.items() if v == "like"]
        disliked_names = [k for k, v in st.session_state.feedback.items() if v == "dislike"]
        parts = []
        if liked_names:
            parts.append(f"👍 Liked: {', '.join(liked_names[:3])}")
        if disliked_names:
            parts.append(f"👎 Skipped: {', '.join(disliked_names[:3])}")
        st.markdown(
            f'<div class="fb-saved">✓ Feedback saved — {" &nbsp;|&nbsp; ".join(parts)}'
            f'<br><span style="font-size:10px;color:#4ade80;opacity:0.6;">Profile updated · affects next session\'s ranking</span></div>',
            unsafe_allow_html=True,
        )

    # ── Product cards + feedback buttons ─────────────────────────────────
    if not recommendations:
        st.info("No products matched this persona. Try adding more interests.")
    else:
        for p in recommendations:
            pname = p.get("product", "")

            try:
                sale   = float(str(p.get("sale_price",   "0")).replace("₹","").strip())
                market = float(str(p.get("market_price", "0")).replace("₹","").strip())
                disc   = round((1 - sale/market)*100) if market > 0 and sale < market else 0
            except Exception:
                disc = 0

            disc_html = f'<span class="disc-badge">-{disc}%</span>' if disc > 0 else ""
            cat_str   = p.get("category","") + (f" › {p.get('sub_category','')}" if p.get("sub_category") else "")
            desc      = str(p.get("description",""))[:220]

            st.markdown(f"""
            <div class="prod-card">
                <div class="prod-name">{pname} {disc_html}</div>
                <div class="prod-meta">{p.get("brand","")}&nbsp;·&nbsp;{cat_str}</div>
                <span class="prod-price">₹{p.get("sale_price","")}</span>
                <span class="prod-rating">★ {p.get("rating","")}</span>
                <div class="prod-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # Feedback buttons
            current_signal = st.session_state.feedback.get(pname)
            like_style    = "✅ Liked" if current_signal == "like"    else "👍"
            dislike_style = "❌ Skip"  if current_signal == "dislike" else "👎"

            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
            with fb_col1:
                if st.button(like_style,    key=f"like_{pname}"):
                    st.session_state.feedback[pname] = "like"
                    st.session_state.feedback_saved   = False
                    st.rerun()
            with fb_col2:
                if st.button(dislike_style, key=f"dislike_{pname}"):
                    st.session_state.feedback[pname] = "dislike"
                    st.session_state.feedback_saved   = False
                    st.rerun()

        # ── Save feedback button ──────────────────────────────────────────
        if st.session_state.feedback and not st.session_state.feedback_saved:
            st.markdown('<div style="margin-top:8px"></div>', unsafe_allow_html=True)
            if st.button("💾 Save feedback & update my profile", key="save_fb"):
                with st.spinner("Updating preference profile…"):
                    process_feedback(
                        name         = st.session_state.last_persona_name,
                        all_products = recommendations,
                        feedback     = st.session_state.feedback,
                    )
                st.session_state.feedback_saved = True
                st.rerun()

    # ── Debug ─────────────────────────────────────────────────────────────
    with st.expander("Debug — intents, signals & routing"):
        st.json({
            "retrieval_strategy" : strategy,
            "confidence_score"   : round(confidence, 4),
            "fallback_applied"   : fallback_applied,
            "intents"            : intents,
            "customer_signals"   : customer_signals,
            "persona"            : result.get("persona", {}),
        })
