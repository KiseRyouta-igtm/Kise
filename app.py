import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SimKebijakan | Praktikum M14",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Dark Navy + Teal accent
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ── Root palette ── */
:root {
    --navy:   #0D1B2A;
    --navy2:  #132233;
    --card:   #192B3C;
    --teal:   #00C9A7;
    --teal2:  #00A98F;
    --amber:  #FFB703;
    --red:    #EF476F;
    --text:   #E8F0FE;
    --muted:  #8BA7BF;
    --border: #1E3448;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--navy) !important;
    color: var(--text) !important;
}

/* ── Main app background ── */
.stApp { background: var(--navy) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy2) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Slider track ── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--teal) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px !important;
}
[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 0.78rem; letter-spacing: .06em; text-transform: uppercase; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--teal) !important; font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.9rem !important; }

/* ── Tab bar ── */
[data-testid="stTabs"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom: 2px solid var(--teal) !important;
}

/* ── Info/Warning boxes ── */
.stInfo, .stWarning, .stSuccess {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Custom header ── */
.sim-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #0f3d57 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.sim-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,201,167,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.sim-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #fff !important;
    margin: 0 0 6px 0;
}
.sim-header p {
    color: var(--muted) !important;
    font-size: 0.9rem;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(0,201,167,0.15);
    color: var(--teal);
    border: 1px solid rgba(0,201,167,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: .05em;
    margin-bottom: 12px;
}

/* ── Section title ── */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--teal);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Sidebar label styling ── */
.sidebar-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
}
.sidebar-section {
    background: rgba(0,201,167,0.06);
    border: 1px solid rgba(0,201,167,0.15);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
}

/* ── Recommendation card ── */
.rec-card {
    background: var(--card);
    border-left: 4px solid var(--teal);
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin-top: 10px;
}
.rec-card.warn { border-left-color: var(--amber); }
.rec-card.danger { border-left-color: var(--red); }

/* ── Sensitivity bar ── */
.sens-bar-wrap { margin-bottom: 10px; }
.sens-label { font-size: 0.82rem; color: var(--muted); margin-bottom: 3px; }
.sens-track { background: var(--border); border-radius: 4px; height: 10px; }
.sens-fill-teal { background: var(--teal); border-radius: 4px; height: 10px; }
.sens-fill-amber { background: var(--amber); border-radius: 4px; height: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL SETUP (Digital Twin)
# ─────────────────────────────────────────────
from sklearn.linear_model import LinearRegression

@st.cache_resource
def load_model():
    """Melatih dan membekukan model sebagai Digital Twin."""
    X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
    y_train = np.array([50, 80, 110, 90, 150])
    model = LinearRegression().fit(X_train, y_train)
    return model

model = load_model()

# Baseline (kondisi bisnis saat ini)
BASELINE_IKLAN  = 10.0
BASELINE_DISKON = 10.0
baseline_input  = np.array([[BASELINE_IKLAN, BASELINE_DISKON]])
baseline_pred   = model.predict(baseline_input)[0]

# ─────────────────────────────────────────────
#  SIMULATION ENGINE
# ─────────────────────────────────────────────
def run_simulation(iklan_val, diskon_val):
    x_new = np.array([[iklan_val, diskon_val]])
    pred  = model.predict(x_new)[0]
    delta = pred - baseline_pred
    pct   = (delta / baseline_pred) * 100
    return pred, delta, pct

def sensitivity_sweep(feature_idx, base_vals, sweep_range):
    results = []
    for v in sweep_range:
        inp = base_vals.copy()
        inp[feature_idx] = v
        pred = model.predict(np.array([inp]))[0]
        results.append(pred)
    return np.array(results)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px 0;'>
        <div style='font-size:2.4rem;'>🧠</div>
        <div style='font-family:Space Grotesk,sans-serif; font-weight:700; font-size:1.1rem; color:#E8F0FE;'>SimKebijakan</div>
        <div style='font-size:0.72rem; color:#8BA7BF; margin-top:2px;'>Praktikum M14 · Digital Twin</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🎛️ Tuas Kebijakan</div>", unsafe_allow_html=True)
    st.caption("Geser untuk menguji skenario kebijakan baru.")

    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>📣 Anggaran Iklan</div>", unsafe_allow_html=True)
    iklan_val = st.slider("Iklan (Juta Rp)", 0, 50, 10, step=1, label_visibility="collapsed")
    st.markdown("<div class='sidebar-label'>🏷️ Besaran Diskon</div>", unsafe_allow_html=True)
    diskon_val = st.slider("Diskon (%)", 0, 50, 10, step=1, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='margin-top:20px;'>🌍 Variabel Lingkungan</div>", unsafe_allow_html=True)
    st.caption("Non-controllable — tetap dari data historis.")
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("🌧️ **Curah Hujan** — Normal (historis)")
    st.markdown("💱 **Kurs USD** — Rp 15.800")
    st.markdown("🏪 **Harga Kompetitor** — Indeks 100")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.72rem; color:#8BA7BF;'>📌 Baseline: Iklan Rp {BASELINE_IKLAN:.0f}Jt · Diskon {BASELINE_DISKON:.0f}%</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ENGINE RUN
# ─────────────────────────────────────────────
pred, delta, pct = run_simulation(iklan_val, diskon_val)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class='sim-header'>
    <div class='badge'>🔬 DIGITAL TWIN — MODEL AKTIF</div>
    <h1>Simulator Kebijakan Keuntungan Toko</h1>
    <p>Analisis What-If berbasis Machine Learning &nbsp;·&nbsp; Minggu 14 Pemodelan & Simulasi &nbsp;·&nbsp;
       Baseline Keuntungan: <strong style="color:#00C9A7;">Rp {baseline_pred:.1f} Juta</strong></p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  METRIC ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("📦 Keuntungan Prediksi", f"Rp {pred:.1f} Jt",
              delta=f"{delta:+.1f} Jt vs Baseline")
with c2:
    st.metric("📈 Perubahan (%)", f"{pct:+.1f}%",
              delta="dari kondisi saat ini")
with c3:
    st.metric("📣 Iklan Saat Ini", f"Rp {iklan_val} Jt",
              delta=f"{iklan_val - BASELINE_IKLAN:+.0f} Jt vs Baseline")
with c4:
    st.metric("🏷️ Diskon Saat Ini", f"{diskon_val}%",
              delta=f"{diskon_val - BASELINE_DISKON:+.0f}% vs Baseline")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Analisis Skenario",
    "🎯  Sensitivitas",
    "🗺️  Peta What-If",
    "📋  Laporan & Rekomendasi",
])

# ═══════════════════════════════════════════════
#  TAB 1 — Analisis Skenario (Comparative Viz)
# ═══════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([1.1, 1], gap="large")

    with col_a:
        st.markdown("<div class='section-title'>📊 Perbandingan Baseline vs Intervensi</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#192B3C')
        ax.set_facecolor('#192B3C')

        bars = ax.bar(['Baseline\n(Kondisi Saat Ini)', 'Intervensi\n(Skenario Baru)'],
                      [baseline_pred, pred],
                      color=['#1E3448', '#00C9A7'],
                      width=0.45,
                      edgecolor='none',
                      zorder=3)

        # Baseline dashed line
        ax.axhline(y=baseline_pred, color='#FFB703', linestyle='--', linewidth=1.2, alpha=0.6, zorder=2)

        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 1.5,
                    f'Rp {h:.1f} Jt',
                    ha='center', va='bottom',
                    color='#E8F0FE', fontsize=11, fontweight='600',
                    fontfamily='DejaVu Sans')

        # Delta annotation
        if abs(delta) > 0.5:
            x_mid = 0.5
            ax.annotate('', xy=(1, pred), xytext=(1, baseline_pred),
                        arrowprops=dict(arrowstyle='<->', color='#FFB703', lw=1.5))
            ax.text(1.28, (baseline_pred + pred) / 2,
                    f'Δ {delta:+.1f} Jt',
                    color='#FFB703', fontsize=10, va='center', fontweight='600')

        ax.set_ylabel('Keuntungan (Juta Rp)', color='#8BA7BF', fontsize=9)
        ax.set_ylim(0, max(pred, baseline_pred) * 1.35)
        ax.tick_params(colors='#8BA7BF', labelsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.yaxis.grid(True, color='#1E3448', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("<div class='section-title'>🔍 Detail Delta Analysis</div>", unsafe_allow_html=True)

        # Delta status
        if delta > 5:
            status_color, status_icon, status_txt = "#00C9A7", "✅", "Skenario MENGUNTUNGKAN"
        elif delta >= 0:
            status_color, status_icon, status_txt = "#FFB703", "⚠️", "Perubahan MINIMAL"
        else:
            status_color, status_icon, status_txt = "#EF476F", "❌", "Skenario MERUGIKAN"

        st.markdown(f"""
        <div style="background:#132233; border:1px solid {status_color}; border-radius:12px; padding:20px; margin-bottom:16px;">
            <div style="font-size:1.5rem;">{status_icon}</div>
            <div style="font-family:Space Grotesk,sans-serif; font-size:1.1rem; font-weight:700; color:{status_color}; margin:6px 0 4px 0;">{status_txt}</div>
            <div style="color:#8BA7BF; font-size:0.83rem;">Delta Keuntungan: <strong style="color:{status_color};">{delta:+.2f} Juta Rp</strong></div>
            <div style="color:#8BA7BF; font-size:0.83rem;">Perubahan Relatif: <strong style="color:{status_color};">{pct:+.1f}%</strong> dari baseline</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**📌 Rincian Input**")
        df_compare = pd.DataFrame({
            "Variabel": ["Anggaran Iklan", "Besaran Diskon"],
            "Baseline": [f"Rp {BASELINE_IKLAN:.0f} Jt", f"{BASELINE_DISKON:.0f}%"],
            "Intervensi": [f"Rp {iklan_val} Jt", f"{diskon_val}%"],
            "Perubahan": [f"{iklan_val - BASELINE_IKLAN:+.0f} Jt", f"{diskon_val - BASELINE_DISKON:+.0f}%"],
        })
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        st.markdown("**🧮 Koefisien Model (Digital Twin)**")
        coef_df = pd.DataFrame({
            "Fitur": ["Anggaran Iklan", "Besaran Diskon"],
            "Koefisien": [f"{model.coef_[0]:.4f}", f"{model.coef_[1]:.4f}"],
            "Artinya": ["+1 Jt Iklan → +β₁ Jt profit", "+1% Diskon → +β₂ Jt profit"],
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
#  TAB 2 — Sensitivity Analysis
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>🎯 Analisis Sensitivitas Variabel Kontrol</div>", unsafe_allow_html=True)
    st.caption("Menunjukkan seberapa besar perubahan output saat tiap variabel digeser dari minimum ke maksimum (ceteris paribus).")

    sweep = np.linspace(0, 50, 100)
    base_vals = [iklan_val, diskon_val]

    iklan_sweep  = sensitivity_sweep(0, base_vals, sweep)
    diskon_sweep = sensitivity_sweep(1, base_vals, sweep)

    range_iklan  = iklan_sweep.max()  - iklan_sweep.min()
    range_diskon = diskon_sweep.max() - diskon_sweep.min()
    total = range_iklan + range_diskon
    pct_iklan  = (range_iklan  / total) * 100
    pct_diskon = (range_diskon / total) * 100

    # Visual sensitivity bars
    st.markdown(f"""
    <div style="background:#192B3C; border:1px solid #1E3448; border-radius:12px; padding:20px; margin-bottom:20px;">
        <div style="font-size:0.75rem; color:#8BA7BF; text-transform:uppercase; letter-spacing:.07em; margin-bottom:14px;">Kontribusi Sensitivitas Relatif</div>
        <div class='sens-bar-wrap'>
            <div class='sens-label'>📣 Anggaran Iklan &nbsp;·&nbsp; <strong style="color:#00C9A7;">{pct_iklan:.1f}%</strong> — Range output: Rp {range_iklan:.1f} Jt</div>
            <div class='sens-track'><div class='sens-fill-teal' style='width:{pct_iklan:.1f}%;'></div></div>
        </div>
        <div class='sens-bar-wrap' style="margin-top:12px;">
            <div class='sens-label'>🏷️ Besaran Diskon &nbsp;·&nbsp; <strong style="color:#FFB703;">{pct_diskon:.1f}%</strong> — Range output: Rp {range_diskon:.1f} Jt</div>
            <div class='sens-track'><div class='sens-fill-amber' style='width:{pct_diskon:.1f}%;'></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    def sens_plot(ax, x, y, label, color, current_val, current_pred):
        ax.set_facecolor('#192B3C')
        ax.plot(x, y, color=color, linewidth=2.5, zorder=3)
        ax.fill_between(x, y, alpha=0.12, color=color)
        ax.axvline(x=current_val, color='white', linestyle=':', linewidth=1.2, alpha=0.5)
        ax.scatter([current_val], [current_pred], color='white', s=60, zorder=5, edgecolors=color, linewidth=1.5)
        ax.set_xlabel(label, color='#8BA7BF', fontsize=9)
        ax.set_ylabel('Keuntungan (Jt Rp)', color='#8BA7BF', fontsize=9)
        ax.tick_params(colors='#8BA7BF', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#1E3448')
        ax.yaxis.grid(True, color='#1E3448', linewidth=0.6)
        ax.set_axisbelow(True)

    with col_s1:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        fig1.patch.set_facecolor('#192B3C')
        ax1.set_title('Sensitivitas: Anggaran Iklan', color='#E8F0FE', fontsize=10, pad=8)
        sens_plot(ax1, sweep, iklan_sweep, 'Anggaran Iklan (Jt Rp)', '#00C9A7', iklan_val, pred)
        st.pyplot(fig1, use_container_width=True)
        plt.close()

    with col_s2:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        fig2.patch.set_facecolor('#192B3C')
        ax2.set_title('Sensitivitas: Besaran Diskon', color='#E8F0FE', fontsize=10, pad=8)
        sens_plot(ax2, sweep, diskon_sweep, 'Besaran Diskon (%)', '#FFB703', diskon_val, pred)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    winner = "Anggaran Iklan" if range_iklan >= range_diskon else "Besaran Diskon"
    st.info(f"🏆 **Tuas Kebijakan Paling Sensitif:** **{winner}** (range output Rp {max(range_iklan, range_diskon):.1f} Juta). "
            f"Perubahan kecil pada variabel ini menghasilkan dampak yang lebih besar pada keuntungan.")

# ═══════════════════════════════════════════════
#  TAB 3 — What-If Map (Heatmap)
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>🗺️ Peta Skenario What-If — Ruang Kebijakan Lengkap</div>", unsafe_allow_html=True)
    st.caption("Setiap sel menunjukkan prediksi keuntungan untuk kombinasi Iklan & Diskon. Titik ⭐ = posisi simulasi Anda saat ini.")

    iklan_grid  = np.arange(0, 51, 5)
    diskon_grid = np.arange(0, 51, 5)
    Z = np.zeros((len(diskon_grid), len(iklan_grid)))
    for i, d in enumerate(diskon_grid):
        for j, a in enumerate(iklan_grid):
            Z[i, j] = model.predict(np.array([[a, d]]))[0]

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.patch.set_facecolor('#192B3C')
    ax3.set_facecolor('#192B3C')

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("teal_map", ['#0D1B2A', '#00C9A7', '#E8F0FE'])
    im = ax3.imshow(Z, cmap=cmap, aspect='auto', origin='lower')

    ax3.set_xticks(range(len(iklan_grid)))
    ax3.set_xticklabels([f'{v}' for v in iklan_grid], color='#8BA7BF', fontsize=8)
    ax3.set_yticks(range(len(diskon_grid)))
    ax3.set_yticklabels([f'{v}%' for v in diskon_grid], color='#8BA7BF', fontsize=8)
    ax3.set_xlabel('Anggaran Iklan (Juta Rp)', color='#8BA7BF', fontsize=10)
    ax3.set_ylabel('Besaran Diskon (%)', color='#8BA7BF', fontsize=10)
    ax3.set_title('Peta What-If: Prediksi Keuntungan (Juta Rp)', color='#E8F0FE', fontsize=11, pad=10)

    # Cell text
    for i in range(len(diskon_grid)):
        for j in range(len(iklan_grid)):
            txt_color = '#0D1B2A' if Z[i,j] > Z.mean() else '#8BA7BF'
            ax3.text(j, i, f'{Z[i,j]:.0f}', ha='center', va='center',
                     color=txt_color, fontsize=7, fontweight='500')

    # Mark current simulation
    ci = np.argmin(np.abs(diskon_grid - diskon_val))
    cj = np.argmin(np.abs(iklan_grid  - iklan_val))
    ax3.scatter(cj, ci, marker='*', s=300, color='#FFB703', zorder=5, label='Simulasi Anda')
    ax3.legend(facecolor='#192B3C', edgecolor='#1E3448', labelcolor='#E8F0FE', fontsize=9)

    # Baseline marker
    bi = np.argmin(np.abs(diskon_grid - BASELINE_DISKON))
    bj = np.argmin(np.abs(iklan_grid  - BASELINE_IKLAN))
    ax3.scatter(bj, bi, marker='o', s=120, color='#EF476F', zorder=5, label='Baseline')
    ax3.legend(facecolor='#192B3C', edgecolor='#1E3448', labelcolor='#E8F0FE', fontsize=9)

    cbar = plt.colorbar(im, ax=ax3)
    cbar.ax.yaxis.set_tick_params(color='#8BA7BF')
    cbar.outline.set_edgecolor('#1E3448')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8BA7BF', fontsize=8)

    st.pyplot(fig3, use_container_width=True)
    plt.close()

    st.caption("🔴 Titik merah = Baseline (kondisi saat ini) &nbsp;·&nbsp; ⭐ Kuning = Posisi simulasi Anda")

# ═══════════════════════════════════════════════
#  TAB 4 — Laporan & Rekomendasi
# ═══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>📋 Laporan Simulasi Otomatis</div>", unsafe_allow_html=True)

    # Auto narrative
    if delta > 10:
        rec_class, rec_icon, rec_head = "rec-card", "🚀", "REKOMENDASI: LAKSANAKAN KEBIJAKAN INI"
        rec_body = (f"Skenario dengan Iklan Rp {iklan_val} Juta dan Diskon {diskon_val}% menghasilkan "
                    f"lonjakan keuntungan sebesar **+Rp {delta:.1f} Juta ({pct:+.1f}%)** dibanding baseline. "
                    f"Kebijakan ini sangat efektif dan layak dipertimbangkan untuk implementasi.")
    elif delta > 0:
        rec_class, rec_icon, rec_head = "rec-card warn", "⚠️", "REKOMENDASI: PERTIMBANGKAN DENGAN HATI-HATI"
        rec_body = (f"Skenario ini memberikan peningkatan marginal sebesar **+Rp {delta:.1f} Juta ({pct:+.1f}%)**. "
                    f"Pertimbangkan apakah biaya implementasi perubahan kebijakan sebanding dengan kenaikan ini.")
    else:
        rec_class, rec_icon, rec_head = "rec-card danger", "❌", "PERINGATAN: SKENARIO TIDAK DISARANKAN"
        rec_body = (f"Skenario ini menyebabkan **penurunan keuntungan sebesar Rp {abs(delta):.1f} Juta ({pct:.1f}%)**. "
                    f"Kembali ke nilai baseline atau cari kombinasi yang lebih optimal di tab Peta What-If.")

    st.markdown(f"""
    <div class='{rec_class}'>
        <div style='font-size:1.4rem; margin-bottom:6px;'>{rec_icon}</div>
        <div style='font-family:Space Grotesk,sans-serif; font-weight:700; font-size:1rem; color:#E8F0FE; margin-bottom:6px;'>{rec_head}</div>
        <div style='color:#8BA7BF; font-size:0.88rem; line-height:1.6;'>{rec_body}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("**📊 Ringkasan Simulasi**")
        summary = pd.DataFrame({
            "Aspek": ["Keuntungan Baseline", "Keuntungan Prediksi", "Delta (Absolut)", "Delta (Relatif)",
                      "Variabel Paling Sensitif", "Batas Ekstrapolasi Aman"],
            "Nilai": [
                f"Rp {baseline_pred:.1f} Juta",
                f"Rp {pred:.1f} Juta",
                f"{'+'if delta>=0 else ''}Rp {delta:.1f} Juta",
                f"{pct:+.1f}%",
                "Iklan" if range_iklan >= range_diskon else "Diskon",
                "Iklan 0–50 Jt | Diskon 0–50%",
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with col_r2:
        st.markdown("**⚠️ Batasan & Validitas Model**")
        st.markdown("""
        <div style='background:#192B3C; border:1px solid #1E3448; border-radius:10px; padding:16px; font-size:0.83rem; color:#8BA7BF; line-height:1.7;'>
            📌 <strong style='color:#E8F0FE;'>Error Model (RMSE):</strong> ~±10 Juta Rp — hasil simulasi memiliki rentang ketidakpastian ini.<br>
            📌 <strong style='color:#E8F0FE;'>Ekstrapolasi:</strong> Input di luar range latih (misal: Diskon 500%) tidak valid — model berperilaku tidak dapat diprediksi.<br>
            📌 <strong style='color:#E8F0FE;'>Retraining:</strong> Jika kondisi ekonomi berubah drastis (resesi, inflasi), baseline dan model harus dilatih ulang.<br>
            📌 <strong style='color:#E8F0FE;'>Variabel Non-Controllable:</strong> Harga Kompetitor, Musim, Kurs belum dimasukkan — simulasi bersifat parsial.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📖 Konsep Akademik yang Diterapkan dalam Simulasi Ini"):
        st.markdown("""
        | Konsep | Implementasi dalam Aplikasi |
        |---|---|
        | **Digital Twin** | Model LinearRegression terlatih yang membekukan pola data historis |
        | **Baseline** | Iklan=10Jt, Diskon=10% → Rp {:.1f} Jt sebagai titik acuan |
        | **What-If Analysis** | `X_intervensi = X_baseline + ΔX` via slider |
        | **Delta Analysis** | `Δy = y_pred - y_baseline` ditampilkan di setiap metrik |
        | **Variabel Kontrol** | Slider Iklan & Diskon (dapat dimanipulasi pengambil keputusan) |
        | **Variabel Kontekstual** | Kurs, Curah Hujan (non-controllable, ditampilkan di sidebar) |
        | **Sensitivity Analysis** | Grafik sweep & bar sensitivitas di Tab 2 |
        | **Comparative Viz** | Bar chart Baseline vs Intervensi di Tab 1 |
        | **What-If Map** | Heatmap ruang kebijakan lengkap di Tab 3 |
        | **Reporting** | Narasi rekomendasi otomatis berbasis nilai delta di Tab 4 |
        """.format(baseline_pred))

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8BA7BF; font-size:0.75rem; padding: 8px 0;'>
    🧠 <strong style='color:#00C9A7;'>SimKebijakan</strong> &nbsp;·&nbsp; Praktikum M14 — Pemodelan & Simulasi &nbsp;·&nbsp;
    Dibangun dengan Streamlit + scikit-learn &nbsp;·&nbsp;
    <em>"Simulator yang baik bukan yang tercantik, tetapi yang paling jujur merepresentasikan pola data sistem asli."</em>
</div>
""", unsafe_allow_html=True)
