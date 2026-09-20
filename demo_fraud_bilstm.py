"""
Demo Streamlit — Attention-BiLSTM Customer Behavior Fraud Detector
===================================================================

Cara pakai
----------
1. Pastikan file berikut ada di folder yang sama dengan script ini
   (atau isi path-nya / upload lewat sidebar):
     - bilstm_fraud_detector.pth   (hasil sel terakhir notebook, §23)
     - fraudTest.csv               (atau CSV lain dengan skema yang sama)
2. Install dependensi:
     pip install streamlit torch pandas numpy scikit-learn
3. Jalankan:
     streamlit run demo_fraud_bilstm.py

Isi demo
--------
Tab 1  Skor transaksi historis : pilih customer -> lihat skor tiap transaksi,
                                 bobot attention, dan window transaksi sebelumnya.
Tab 2  Simulasi transaksi baru : tambahkan transaksi hipotetis ke riwayat
                                 seorang customer lalu lihat skor fraud-nya.
Tab 3  Batch & evaluasi        : skor banyak customer sekaligus, hitung
                                 Precision / Recall / F1 / ROC-AUC / PR-AUC.

Catatan
-------
Preprocessing (fitur statis, vocabulary encoding, fitur temporal, sliding
window, scaling) direplikasi persis dari notebook
`fraud_bilstm_behavior_embedding_v3-2-3.ipynb` (§7-§10, §12, §23). Fitur
temporal (time_since_last_txn_hours, roll_mean/std_amt_5, txn_count_24h)
hanya melihat riwayat yang ada di CSV yang dimuat — di notebook riwayat
customer bisa menyeberang dari fraudTrain ke fraudTest, jadi beberapa
transaksi awal di CSV demo bisa memberi fitur temporal yang sedikit
berbeda dari saat model dilatih.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REQUIRED_RAW_COLS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "lat", "long", "city_pop", "job", "dob", "merch_lat", "merch_long",
]


# =============================================================================
# 1. Arsitektur model (identik dengan notebook §12)
# =============================================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context_vec = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor, mask: torch.Tensor):
        energy = torch.tanh(self.attn(lstm_out))
        scores = self.context_vec(energy).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context, weights


class CustomerBehaviorBiLSTM(nn.Module):
    def __init__(
        self,
        cat_cardinalities: List[int],
        cat_emb_dims: List[int],
        num_numeric: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        dense_units: int = 64,
        behavior_dim: int = 64,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, dim, padding_idx=0)
            for cardinality, dim in zip(cat_cardinalities, cat_emb_dims)
        ])
        lstm_input_dim = sum(cat_emb_dims) + num_numeric
        self.input_norm = nn.LayerNorm(lstm_input_dim)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2
        self.lstm_out_norm = nn.LayerNorm(lstm_out_dim)
        if use_attention:
            self.attention = TemporalAttention(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_behavior_dense = nn.Linear(lstm_out_dim, dense_units)
        self.pre_behavior_norm = nn.LayerNorm(dense_units)
        self.activation = nn.GELU()
        self.behavior_embedding = nn.Linear(dense_units, behavior_dim)
        self.residual_proj = (nn.Linear(lstm_out_dim, behavior_dim)
                              if lstm_out_dim != behavior_dim else nn.Identity())
        self.output_dense = nn.Linear(behavior_dim, 1)

    def forward(self, cat_seq, num_seq, mask=None):
        emb_list = [emb(cat_seq[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(emb_list + [num_seq], dim=-1)
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_out_norm(lstm_out)
        if self.use_attention:
            pooled, attn_weights = self.attention(lstm_out, mask)
        else:
            pooled = lstm_out[:, -1, :]
            attn_weights = None
        residual_in = pooled
        z = self.dropout(pooled)
        z = self.activation(self.pre_behavior_dense(z))
        z = self.pre_behavior_norm(z)
        behavior_emb = self.behavior_embedding(z) + self.residual_proj(residual_in)
        logit = self.output_dense(behavior_emb).squeeze(-1)
        return logit, behavior_emb, attn_weights


# =============================================================================
# 2. Load checkpoint (kunci-kunci sesuai notebook §23)
# =============================================================================
@dataclass
class Bundle:
    model: CustomerBehaviorBiLSTM
    threshold: float
    vocabularies: Dict[str, Dict[str, int]]
    scaler: object
    seq_len: int
    numeric_cols: List[str]
    cat_cols: List[str]

    @property
    def cat_enc_cols(self) -> List[str]:
        return [c + "_enc" for c in self.cat_cols]


def load_bundle(source) -> Bundle:
    """source: path (str) ke .pth, atau bytes hasil upload."""
    buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    # weights_only=False karena checkpoint memuat StandardScaler (objek sklearn)
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    hp = ckpt["best_hp"]
    model = CustomerBehaviorBiLSTM(
        cat_cardinalities=list(ckpt["cat_cardinalities"]),
        cat_emb_dims=[int(hp["embedding_merchant"]), int(hp["embedding_category"]),
                      int(hp["embedding_job"]), int(hp["embedding_gender"])],
        num_numeric=int(ckpt["num_numeric"]),
        hidden_size=int(hp["hidden_size"]),
        num_layers=int(hp["num_layers"]),
        dropout=float(hp["dropout"]),
        dense_units=int(hp["dense_units"]),
        behavior_dim=64,
        use_attention=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return Bundle(
        model=model,
        threshold=float(ckpt["threshold"]),
        vocabularies=ckpt["vocabularies"],
        scaler=ckpt["numeric_scaler"],
        seq_len=int(ckpt["seq_len"]),
        numeric_cols=list(ckpt["numeric_cols"]),
        cat_cols=list(ckpt["categorical_cols"]),
    )


# =============================================================================
# 3. Preprocessing (identik dengan notebook §7-§10)
# =============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2r - lat1r) / 2.0) ** 2
         + np.cos(lat1r) * np.cos(lat2r) * np.sin((lon2r - lon1r) / 2.0) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def _velocity_24h(times: pd.Series) -> np.ndarray:
    times_ns = times.values.astype("datetime64[ns]").astype(np.int64)
    window_ns = 24 * 3600 * 1_000_000_000
    left_idx = np.searchsorted(times_ns, times_ns - window_ns, side="left")
    return (np.arange(len(times_ns)) - left_idx).astype(np.float64)


def prepare_frame(raw: pd.DataFrame, bundle: Bundle) -> pd.DataFrame:
    """Raw transaksi -> frame terurut per customer lengkap dengan semua fitur.
    Semua kolom asli (mis. merchant, category, amt, is_fraud) tetap dipertahankan."""
    missing = [c for c in REQUIRED_RAW_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada di data: {missing}")

    df = raw.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    # --- fitur statis (§7) ---
    df["log_amt"] = np.log1p(df["amt"])
    df["distance"] = haversine_km(df["lat"].values, df["long"].values,
                                  df["merch_lat"].values, df["merch_long"].values)
    df["hour"] = df["trans_date_trans_time"].dt.hour.astype(np.float32)
    df["dayofweek"] = df["trans_date_trans_time"].dt.dayofweek.astype(np.float32)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.float32)
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25

    # --- encoding vocabulary, 0 = unseen/PAD (§8) ---
    for col in bundle.cat_cols:
        vocab = bundle.vocabularies[col]
        df[col + "_enc"] = df[col].astype(str).map(vocab).fillna(0).astype(np.int64)

    # --- urutkan per customer, lalu fitur temporal kausal (§9b) ---
    df = df.sort_values(["cc_num", "trans_date_trans_time"], kind="mergesort").reset_index(drop=True)
    df["pos_in_group"] = df.groupby("cc_num").cumcount()

    g = df.groupby("cc_num")
    df["time_since_last_txn_hours"] = (
        g["trans_date_trans_time"].diff().dt.total_seconds() / 3600.0
    ).fillna(-1.0)
    df["_amt_shifted"] = df.groupby("cc_num")["amt"].shift(1)
    df["roll_mean_amt_5"] = df.groupby("cc_num")["_amt_shifted"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()).fillna(0.0)
    df["roll_std_amt_5"] = df.groupby("cc_num")["_amt_shifted"].transform(
        lambda s: s.rolling(5, min_periods=1).std()).fillna(0.0)
    df = df.drop(columns=["_amt_shifted"])
    df["txn_count_24h"] = df.groupby("cc_num")["trans_date_trans_time"].transform(_velocity_24h)
    return df


def build_windows(df: pd.DataFrame, row_idx: np.ndarray, bundle: Bundle):
    """Sliding window seq_len transaksi terakhir (termasuk transaksi itu sendiri)
    untuk baris-baris `row_idx`; tidak pernah menyeberang ke customer lain (§10)."""
    n = len(df)
    L = bundle.seq_len
    pos = df["pos_in_group"].values
    offsets = np.arange(L - 1, -1, -1)
    idx = row_idx[:, None] - offsets[None, :]
    valid = (idx >= 0) & (pos[row_idx][:, None] >= offsets[None, :])
    idx_c = np.clip(idx, 0, n - 1)

    num_mat = df[bundle.numeric_cols].values.astype(np.float32)
    cat_mat = df[bundle.cat_enc_cols].values.astype(np.int64)
    num_seq = num_mat[idx_c] * valid[:, :, None]
    cat_seq = cat_mat[idx_c] * valid[:, :, None]

    # standardisasi hanya di posisi valid (padding tetap 0), sama seperti §10-scaler
    shp = num_seq.shape
    flat = num_seq.reshape(-1, shp[-1])
    flat_valid = valid.reshape(-1)
    out = flat.copy()
    if flat_valid.any():
        out[flat_valid] = bundle.scaler.transform(flat[flat_valid])
    out[~flat_valid] = 0.0
    num_seq = out.reshape(shp).astype(np.float32)
    return cat_seq, num_seq, valid


@torch.no_grad()
def predict(bundle: Bundle, cat_seq, num_seq, valid, batch_size: int = 1024):
    probs, attns = [], []
    for i in range(0, len(cat_seq), batch_size):
        c = torch.from_numpy(cat_seq[i:i + batch_size]).long()
        n = torch.from_numpy(num_seq[i:i + batch_size]).float()
        m = torch.from_numpy(valid[i:i + batch_size]).bool()
        logit, _, w = bundle.model(c, n, m)
        probs.append(torch.sigmoid(logit).numpy())
        attns.append(w.numpy() if w is not None else np.zeros(m.shape, dtype=np.float32))
    return np.concatenate(probs), np.concatenate(attns)


def score_rows(bundle: Bundle, df: pd.DataFrame, row_idx: Optional[np.ndarray] = None):
    if row_idx is None:
        row_idx = np.arange(len(df))
    cat_seq, num_seq, valid = build_windows(df, row_idx, bundle)
    probs, attn = predict(bundle, cat_seq, num_seq, valid)
    return probs, attn, valid


def explain_window(df: pd.DataFrame, row: int, valid_row: np.ndarray,
                   attn_row: np.ndarray, bundle: Bundle) -> pd.DataFrame:
    """Tabel transaksi dalam window model + bobot attention-nya."""
    offsets = np.arange(bundle.seq_len - 1, -1, -1)
    rows = row - offsets
    keep = valid_row.astype(bool)
    cols = ["trans_date_trans_time", "merchant", "category", "amt", "distance"]
    w = df.iloc[rows[keep]][cols].copy()
    w.insert(0, "langkah", [f"t-{o}" if o else "t (transaksi ini)" for o in offsets[keep]])
    w["attention"] = attn_row[keep]
    return w.reset_index(drop=True)


# =============================================================================
# 4. UI Streamlit
# =============================================================================
def _mask(cc) -> str:
    return f"****{str(cc)[-4:]}"


def main():
    import datetime as dt

    import streamlit as st
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)

    st.set_page_config(page_title="Demo Fraud Detector — Attention-BiLSTM",
                       page_icon="🛡️", layout="wide")

    @st.cache_resource(show_spinner="Memuat model...")
    def _load_bundle(source):
        return load_bundle(source)

    @st.cache_data(show_spinner="Membaca CSV...")
    def _read_csv(source):
        src = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
        return pd.read_csv(src)

    # ------------------------------ sidebar ---------------------------------
    st.sidebar.header("⚙️ Sumber model & data")
    ckpt_path = st.sidebar.text_input("Path checkpoint (.pth)", "bilstm_fraud_detector.pth")
    ckpt_upload = st.sidebar.file_uploader("…atau upload checkpoint", type=["pth", "pt"])
    csv_path = st.sidebar.text_input("Path CSV transaksi", "fraudTest.csv")
    csv_upload = st.sidebar.file_uploader("…atau upload CSV", type=["csv"])

    try:
        bundle = _load_bundle(ckpt_upload.getvalue() if ckpt_upload else ckpt_path)
    except Exception as e:  # noqa: BLE001
        st.title("🛡️ Demo Fraud Detector")
        st.error(f"Gagal memuat checkpoint: {e}")
        st.info("Isi path yang benar di sidebar atau upload file `bilstm_fraud_detector.pth`.")
        st.stop()

    try:
        raw_all = _read_csv(csv_upload.getvalue() if csv_upload else csv_path)
        missing = [c for c in REQUIRED_RAW_COLS if c not in raw_all.columns]
        if missing:
            raise ValueError(f"CSV tidak punya kolom: {missing}")
    except Exception as e:  # noqa: BLE001
        st.title("🛡️ Demo Fraud Detector")
        st.error(f"Gagal membaca data: {e}")
        st.info("Isi path CSV di sidebar atau upload `fraudTest.csv`.")
        st.stop()

    threshold = st.sidebar.slider(
        "Threshold fraud", 0.0, 1.0, float(np.clip(bundle.threshold, 0.0, 1.0)), 0.001,
        help="Default = threshold optimal dari validasi (tersimpan di checkpoint).")
    st.sidebar.caption(f"Threshold checkpoint: **{bundle.threshold:.4f}** · "
                       f"window: **{bundle.seq_len}** transaksi")
    st.sidebar.caption(f"Data: {len(raw_all):,} transaksi · "
                       f"{raw_all['cc_num'].nunique():,} customer")

    st.title("🛡️ Demo Fraud Detector — Attention-BiLSTM")
    st.caption("Model membaca hingga 20 transaksi terakhir seorang customer (urut waktu), "
               "lalu memberi probabilitas bahwa transaksi terakhir itu fraud.")

    tab1, tab2, tab3 = st.tabs(["🔎 Skor transaksi historis",
                                "🧪 Simulasi transaksi baru",
                                "📊 Batch & evaluasi"])

    customers = raw_all.groupby("cc_num").size().sort_values(ascending=False)

    def show_attention(w: pd.DataFrame):
        st.markdown("**Bobot attention** — transaksi mana di riwayat yang paling "
                    "diperhatikan model:")
        chart = w.set_index("langkah")[["attention"]]
        st.bar_chart(chart)
        st.dataframe(
            w, width="stretch", hide_index=True,
            column_config={
                "attention": st.column_config.ProgressColumn(
                    "attention", min_value=0.0, max_value=1.0, format="%.3f"),
                "amt": st.column_config.NumberColumn("amt", format="$%.2f"),
                "distance": st.column_config.NumberColumn("jarak (km)", format="%.1f"),
            })

    def verdict(prob: float):
        c1, c2 = st.columns([1, 2])
        c1.metric("Probabilitas fraud", f"{prob:.2%}",
                  delta=f"threshold {threshold:.2%}", delta_color="off")
        with c2:
            st.progress(float(min(max(prob, 0.0), 1.0)))
            if prob >= threshold:
                st.error("🚨 **FRAUD** — transaksi ini akan ditandai/diblokir.")
            else:
                st.success("✅ **LEGIT** — transaksi ini dianggap normal.")

    # =========================================================================
    # TAB 1 — transaksi historis
    # =========================================================================
    with tab1:
        st.subheader("Pilih customer & transaksi")
        only_fraud_cust = st.checkbox(
            "Hanya customer yang punya transaksi fraud (label asli)",
            value=("is_fraud" in raw_all.columns),
            disabled=("is_fraud" not in raw_all.columns))
        pool = customers
        if only_fraud_cust and "is_fraud" in raw_all.columns:
            fraud_ccs = raw_all.loc[raw_all["is_fraud"] == 1, "cc_num"].unique()
            pool = customers[customers.index.isin(fraud_ccs)]
            if len(pool) == 0:
                pool = customers
        cc = st.selectbox(
            "Customer (kartu)", pool.index.tolist(),
            format_func=lambda x: f"{_mask(x)} — {pool[x]:,} transaksi", key="t1_cc")

        raw_c = raw_all[raw_all["cc_num"] == cc]
        df_c = prepare_frame(raw_c, bundle)
        probs, attn, valid = score_rows(bundle, df_c)
        df_c["prob"] = probs
        df_c["pred"] = (probs >= threshold).astype(int)

        tbl_cols = ["trans_date_trans_time", "merchant", "category", "amt", "distance", "prob", "pred"]
        if "is_fraud" in df_c.columns:
            tbl_cols.insert(-2, "is_fraud")
        view = df_c[tbl_cols].copy()
        view.insert(0, "row", df_c.index)

        f1, f2 = st.columns(2)
        show_only = f1.selectbox("Filter tabel", ["Semua", "Ditandai fraud oleh model",
                                                   "Fraud asli (label)"], key="t1_filter")
        n_show = f2.slider("Jumlah baris ditampilkan", 10, 500, 50, 10, key="t1_n")
        if show_only == "Ditandai fraud oleh model":
            view = view[view["pred"] == 1]
        elif show_only == "Fraud asli (label)" and "is_fraud" in view.columns:
            view = view[view["is_fraud"] == 1]
        view = view.sort_values("prob", ascending=False).head(n_show)
        st.dataframe(
            view, width="stretch", hide_index=True,
            column_config={
                "prob": st.column_config.ProgressColumn(
                    "skor fraud", min_value=0.0, max_value=1.0, format="%.3f"),
                "amt": st.column_config.NumberColumn("amt", format="$%.2f"),
                "distance": st.column_config.NumberColumn("jarak (km)", format="%.1f"),
            })

        st.divider()
        st.subheader("Detail satu transaksi")
        options = view["row"].tolist() or df_c.index.tolist()[:1]

        def _label(r):
            x = df_c.loc[r]
            extra = ""
            if "is_fraud" in df_c.columns:
                extra = " · label: FRAUD" if x["is_fraud"] == 1 else ""
            return (f"{x['trans_date_trans_time']:%Y-%m-%d %H:%M} · {x['category']} · "
                    f"${x['amt']:.2f} · skor {x['prob']:.3f}{extra}")

        r = st.selectbox("Transaksi", options, format_func=_label, key="t1_row")
        x = df_c.loc[r]
        verdict(float(x["prob"]))
        if "is_fraud" in df_c.columns:
            st.caption(f"Label asli: **{'FRAUD' if x['is_fraud'] == 1 else 'LEGIT'}**")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Jumlah", f"${x['amt']:.2f}")
        d2.metric("Jarak ke merchant", f"{x['distance']:.1f} km")
        d3.metric("Txn 24 jam terakhir", f"{int(x['txn_count_24h'])}")
        d4.metric("Selang dari txn sebelumnya",
                  "—" if x["time_since_last_txn_hours"] < 0
                  else f"{x['time_since_last_txn_hours']:.1f} jam")
        show_attention(explain_window(df_c, int(r), valid[int(r)], attn[int(r)], bundle))

    # =========================================================================
    # TAB 2 — simulasi transaksi baru
    # =========================================================================
    with tab2:
        st.subheader("Tambahkan transaksi hipotetis ke riwayat customer")
        st.caption("Transaksi baru ditambahkan di akhir riwayat asli customer, lalu "
                   "model menilainya seperti transaksi live.")

        cc2 = st.selectbox(
            "Customer (kartu)", customers.index.tolist(),
            format_func=lambda x: f"{_mask(x)} — {customers[x]:,} transaksi", key="t2_cc")
        hist = raw_all[raw_all["cc_num"] == cc2].copy()
        hist["trans_date_trans_time"] = pd.to_datetime(hist["trans_date_trans_time"])
        hist = hist.sort_values("trans_date_trans_time").tail(200).reset_index(drop=True)
        last = hist.iloc[-1]
        last_ts = last["trans_date_trans_time"]

        merchants = sorted(bundle.vocabularies["merchant"].keys())
        categories = sorted(bundle.vocabularies["category"].keys())
        top_merchant = hist["merchant"].mode().iloc[0]
        top_category = hist["category"].mode().iloc[0]

        preset = st.radio("Skenario", ["Normal", "Mencurigakan", "Manual"],
                          horizontal=True, key="t2_preset")
        if preset == "Normal":
            d = dict(merchant=top_merchant, category=top_category,
                     amt=float(round(hist["amt"].median(), 2)),
                     ts=last_ts + pd.Timedelta(hours=3),
                     mlat=float(hist["merch_lat"].median()),
                     mlong=float(hist["merch_long"].median()))
        elif preset == "Mencurigakan":
            d = dict(merchant=top_merchant,
                     category="shopping_net" if "shopping_net" in categories else top_category,
                     amt=float(max(1200.0, round(hist["amt"].quantile(0.99) * 3, 2))),
                     ts=(last_ts.normalize() + pd.Timedelta(days=1, hours=2, minutes=30)),
                     mlat=float(min(last["lat"] + 7.0, 49.0)),
                     mlong=float(last["long"] - 18.0))
        else:
            d = dict(merchant=top_merchant, category=top_category,
                     amt=float(round(hist["amt"].median(), 2)),
                     ts=last_ts + pd.Timedelta(hours=1),
                     mlat=float(last["merch_lat"]), mlong=float(last["merch_long"]))
        k = f"{cc2}_{preset}"  # ganti key -> widget ter-reset saat preset/customer berganti

        c1, c2, c3 = st.columns(3)
        merchant = c1.selectbox("Merchant", merchants, index=merchants.index(d["merchant"])
                                if d["merchant"] in merchants else 0, key=f"m_{k}")
        category = c2.selectbox("Kategori", categories, index=categories.index(d["category"])
                                if d["category"] in categories else 0, key=f"c_{k}")
        amt = c3.number_input("Jumlah ($)", min_value=0.01, value=max(d["amt"], 0.01),
                              step=10.0, key=f"a_{k}")
        c4, c5 = st.columns(2)
        date_v = c4.date_input("Tanggal", value=d["ts"].date(), key=f"d_{k}")
        time_v = c5.time_input("Jam", value=d["ts"].time().replace(microsecond=0), key=f"t_{k}")
        c6, c7 = st.columns(2)
        mlat = c6.number_input("Latitude merchant", value=d["mlat"], format="%.4f", key=f"la_{k}")
        mlong = c7.number_input("Longitude merchant", value=d["mlong"], format="%.4f", key=f"lo_{k}")

        new_ts = pd.Timestamp(dt.datetime.combine(date_v, time_v))
        home_dist = float(haversine_km(last["lat"], last["long"], mlat, mlong))
        st.caption(f"Jarak rumah customer → merchant: **{home_dist:,.1f} km** · "
                   f"transaksi terakhir customer: {last_ts:%Y-%m-%d %H:%M}")

        if st.button("🔍 Skor transaksi ini", type="primary", key="t2_go"):
            new_row = last.copy()
            new_row["trans_date_trans_time"] = new_ts
            new_row["merchant"] = merchant
            new_row["category"] = category
            new_row["amt"] = float(amt)
            new_row["merch_lat"] = float(mlat)
            new_row["merch_long"] = float(mlong)
            if "is_fraud" in new_row.index:
                new_row["is_fraud"] = 0
            hist["_is_new"] = False
            new_df = pd.DataFrame([new_row])
            new_df["_is_new"] = True
            combo = pd.concat([hist, new_df], ignore_index=True)

            df2 = prepare_frame(combo, bundle)
            ridx = int(np.where(df2["_is_new"].values)[0][0])
            p, a, v = score_rows(bundle, df2, np.array([ridx]))
            prob = float(p[0])

            if merchant not in bundle.vocabularies["merchant"]:
                st.warning("Merchant tidak ada di vocabulary training → dianggap 'unseen'.")
            verdict(prob)
            xr = df2.loc[ridx]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Jarak ke merchant", f"{xr['distance']:.1f} km")
            m2.metric("Txn 24 jam terakhir", f"{int(xr['txn_count_24h'])}")
            m3.metric("Rata-rata 5 txn terakhir", f"${xr['roll_mean_amt_5']:.2f}")
            m4.metric("Selang dari txn sebelumnya",
                      "—" if xr["time_since_last_txn_hours"] < 0
                      else f"{xr['time_since_last_txn_hours']:.1f} jam")
            show_attention(explain_window(df2, ridx, v[0], a[0], bundle))

    # =========================================================================
    # TAB 3 — batch & evaluasi
    # =========================================================================
    with tab3:
        st.subheader("Skor banyak customer sekaligus")
        n_total = len(customers)
        n_cust = st.slider("Jumlah customer (sampel acak)", 1, min(n_total, 300),
                           min(30, n_total), key="t3_n")
        seed = st.number_input("Seed sampel", value=42, step=1, key="t3_seed")
        if st.button("▶️ Jalankan batch scoring", type="primary", key="t3_go"):
            rng = np.random.RandomState(int(seed))
            chosen = rng.choice(customers.index.values, size=n_cust, replace=False)
            with st.spinner("Menyiapkan fitur & menjalankan model..."):
                dfb = prepare_frame(raw_all[raw_all["cc_num"].isin(chosen)], bundle)
                pb, _, _ = score_rows(bundle, dfb)
                dfb["prob"] = pb
            st.session_state["batch_df"] = dfb

        dfb = st.session_state.get("batch_df")
        if dfb is not None:
            pred = (dfb["prob"].values >= threshold).astype(int)
            st.write(f"**{len(dfb):,}** transaksi dari **{dfb['cc_num'].nunique()}** customer · "
                     f"ditandai fraud: **{int(pred.sum()):,}**")

            if "is_fraud" in dfb.columns and dfb["is_fraud"].nunique() > 1:
                y = dfb["is_fraud"].astype(int).values
                p_, r_, f_, _ = precision_recall_fscore_support(
                    y, pred, average="binary", zero_division=0)
                m = st.columns(5)
                m[0].metric("Precision", f"{p_:.4f}")
                m[1].metric("Recall", f"{r_:.4f}")
                m[2].metric("F1", f"{f_:.4f}")
                m[3].metric("ROC-AUC", f"{roc_auc_score(y, dfb['prob']):.4f}")
                m[4].metric("PR-AUC", f"{average_precision_score(y, dfb['prob']):.4f}")
                cm = confusion_matrix(y, pred, labels=[0, 1])
                st.markdown("**Confusion matrix**")
                st.dataframe(pd.DataFrame(
                    cm, index=["Aktual: Legit", "Aktual: Fraud"],
                    columns=["Prediksi: Legit", "Prediksi: Fraud"]))
            else:
                st.info("Data tidak punya label `is_fraud` yang bervariasi — metrik dilewati.")

            topn = st.slider("Tampilkan top-N skor tertinggi", 10, 500, 50, 10, key="t3_top")
            cols = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "prob"]
            if "is_fraud" in dfb.columns:
                cols.insert(-1, "is_fraud")
            out = dfb[cols].sort_values("prob", ascending=False).head(topn).copy()
            out["cc_num"] = out["cc_num"].map(_mask)
            st.dataframe(out, width="stretch", hide_index=True,
                         column_config={"prob": st.column_config.ProgressColumn(
                             "skor fraud", min_value=0.0, max_value=1.0, format="%.3f")})
            st.download_button(
                "⬇️ Download hasil (CSV)",
                dfb[cols].assign(cc_num=dfb["cc_num"].map(_mask), pred=pred).to_csv(index=False),
                file_name="hasil_batch_scoring.csv", mime="text/csv")


if __name__ == "__main__":
    main()
