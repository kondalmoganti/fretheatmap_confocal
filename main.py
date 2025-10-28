import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# -------------------------
# App config
# -------------------------
st.set_page_config(
    page_title="FRET Analyzer – FULL",
    layout="wide",
)
st.title("FRET Analyzer – FULL (Auto-fit, Manual/Auto bins, AUC tools)")

# -------------------------
# Helpers
# -------------------------
def split_numeric_blocks_with_headers(text: str):
    """Split a mixed .dat file into numeric blocks (tables/matrices)."""
    text = text.replace(",", ".")  # decimal commas -> dots
    lines = text.splitlines()
    blocks = []
    cur = []
    header_buf = []
    start_idx = None
    isnum = re.compile(r'^\s*[\d\.eE\-\+]+([\s\t,;][\d\.eE\-\+]+)*\s*$').match

    def flush():
        nonlocal cur, header_buf, start_idx
        if not cur:
            return
        s = "\n".join(cur).strip()
        try:
            df = pd.read_csv(io.StringIO(s), sep=r"[\s,;]+", engine="python", header=None)
            blocks.append((df, list(header_buf), start_idx, None))
        except Exception:
            pass
        cur = []
        start_idx = None

    for i, ln in enumerate(lines):
        if isnum(ln):
            if start_idx is None:
                start_idx = i
            cur.append(ln)
        else:
            header_buf.append(ln.strip())
            header_buf = header_buf[-3:]
            if start_idx is not None:
                flush()
    if start_idx is not None:
        flush()
    return blocks


def gaussian(x, y0, mu, sigma, A):
    amp = A / (sigma * np.sqrt(2 * np.pi))
    return y0 + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian2(x, y0, mu1, s1, A1, mu2, s2, A2):
    return gaussian(x, y0, mu1, s1, A1) - y0 + gaussian(x, y0, mu2, s2, A2)


def r2_score(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    ss_res = np.nansum((y - yhat) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def auto_bins(x, rule="Freedman–Diaconis"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 40
    if rule == "Freedman–Diaconis":
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr == 0:
            return int(np.clip(np.sqrt(n), 10, 200))
        bw = 2 * iqr * (n ** (-1 / 3))
    elif rule == "Scott":
        std = np.nanstd(x, ddof=1)
        if std == 0:
            return int(np.clip(np.sqrt(n), 10, 200))
        bw = 3.5 * std * (n ** (-1 / 3))
    else:  # Sturges
        return int(np.clip(np.ceil(np.log2(n) + 1), 10, 200))
    rng = np.nanmax(x) - np.nanmin(x)
    if rng <= 0:
        return int(np.clip(np.sqrt(n), 10, 200))
    return int(np.clip(np.ceil(rng / bw), 10, 200))


def best_autofit(centers, hist, xmin, xmax):
    """Try 1-Gaussian and 2-Gaussian; pick by AICc with separation sanity."""
    z = np.asarray(hist, float)
    x = np.asarray(centers, float)
    m = np.isfinite(z) & np.isfinite(x)
    x, z = x[m], z[m]
    if x.size < 8 or np.count_nonzero(z) < 5:
        return None

    z_s = gaussian_filter1d(z, sigma=max(1, int(len(z) * 0.03)))
    binw = np.median(np.diff(x)) if len(x) > 1 else 0.01
    sig_min = max(0.5 * binw, 1e-4)
    sig_max = max((xmax - xmin) / 1.25, sig_min * 2)
    y0_min = 0.0
    y0_max = max(z.max() * 2, 5.0)

    # 1G init
    mu0 = float(x[int(np.argmax(z_s))])
    y00 = max(1e-6, float(np.quantile(z, 0.1)))
    support = z_s >= 0.1 * z_s.max()
    xw = x[support] if np.any(support) else x
    q25, q75 = np.quantile(xw, [0.25, 0.75])
    sigma0 = max(sig_min, (q75 - q25) / 1.349)
    A0 = float(np.trapz(np.maximum(z - y00, 0), x))
    lower1 = [y0_min, xmin, sig_min, 0.0]
    upper1 = [y0_max, xmax, sig_max, np.inf]
    p1, rss1, perr1 = None, np.inf, None
    try:
        p1_opt, pc1 = curve_fit(
            gaussian,
            x,
            z,
            p0=[y00, mu0, sigma0, max(A0, 1e-6)],
            bounds=(lower1, upper1),
            maxfev=100000,
        )
        yhat1 = gaussian(x, *p1_opt)
        rss1 = np.nansum((z - yhat1) ** 2)
        perr1 = np.sqrt(np.diag(pc1))
        p1 = p1_opt
    except Exception:
        pass

    # 2G init
    peaks, _ = find_peaks(z_s, distance=max(2, int(0.07 / max(np.mean(np.diff(x)), 1e-6))))
    peaks = peaks[np.argsort(z_s[peaks])][::-1]
    if len(peaks) >= 2:
        mu1_0, mu2_0 = x[peaks[:2]]
        if mu1_0 > mu2_0:
            mu1_0, mu2_0 = mu2_0, mu1_0
    else:
        mu1_0 = float(np.clip(mu0 - 0.1 * (xmax - xmin), xmin, xmax))
        mu2_0 = float(np.clip(mu0 + 0.1 * (xmax - xmin), xmin, xmax))
    s1_0 = s2_0 = max(sig_min * 1.5, sigma0 * 0.8)
    A1_0 = A2_0 = max(A0 / 2, 1e-6)
    lower2 = [y0_min, xmin, sig_min, 0.0, xmin, sig_min, 0.0]
    upper2 = [y0_max, xmax, sig_max, np.inf, xmax, sig_max, np.inf]
    p2, rss2, perr2 = None, np.inf, None
    try:
        p2_opt, pc2 = curve_fit(
            gaussian2,
            x,
            z,
            p0=[y00, mu1_0, s1_0, A1_0, mu2_0, s2_0, A2_0],
            bounds=(lower2, upper2),
            maxfev=150000,
        )
        yhat2 = gaussian2(x, *p2_opt)
        rss2 = np.nansum((z - yhat2) ** 2)
        perr2 = np.sqrt(np.diag(pc2))
        p2 = p2_opt
    except Exception:
        pass

    k1, k2 = 4, 7
    n = len(x)
    aicc1 = (
        n * np.log(rss1 / n) + 2 * k1 + (2 * k1 * (k1 + 1)) / (n - k1 - 1)
        if np.isfinite(rss1) and rss1 > 0
        else np.inf
    )
    aicc2 = (
        n * np.log(rss2 / n) + 2 * k2 + (2 * k2 * (k2 + 1)) / (n - k2 - 1)
        if np.isfinite(rss2) and rss2 > 0
        else np.inf
    )

    sep_ok = False
    if p2 is not None:
        _, m1, s1, A1, m2, s2, A2 = p2
        sep_ok = abs(m2 - m1) > 1.2 * max(s1, s2)

    if p2 is not None and sep_ok and aicc2 + 2 < aicc1:
        yhat = gaussian2(x, *p2)
        return {"model": "2G", "params": p2, "perr": perr2, "R2": r2_score(z, yhat), "aicc": aicc2, "yhat": (x, yhat)}
    elif p1 is not None:
        yhat = gaussian(x, *p1)
        return {"model": "1G", "params": p1, "perr": perr1, "R2": r2_score(z, yhat), "aicc": aicc1, "yhat": (x, yhat)}
    else:
        return None


def clean_pair(x, w):
    """Robust numeric casting for arrays/series + mask finite, non-negative weights."""
    x = np.asarray(pd.to_numeric(x, errors="coerce"))
    w = np.asarray(pd.to_numeric(w, errors="coerce"))
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    return x[m], w[m]


def clamp_manual_bins_E(min_label, max_label, bins_label, default_bins=80, key_prefix=""):
    """UI group for manual E binning, clamped to [0,1] with monotonic edges."""
    c1, c2, c3 = st.columns(3)
    with c1:
        nb = st.number_input(
            bins_label, min_value=5, max_value=400, value=default_bins, step=1, key=f"{key_prefix}_nb"
        )
    with c2:
        xmin = st.number_input(
            min_label, value=0.0, min_value=0.0, max_value=1.0, step=0.01, key=f"{key_prefix}_xmin"
        )
    with c3:
        xmax = st.number_input(
            max_label, value=1.0, min_value=0.0, max_value=1.0, step=0.01, key=f"{key_prefix}_xmax"
        )
    xmin = float(xmin)
    xmax = float(xmax)
    if xmax <= xmin:
        xmax = xmin + 1e-6
    edges = np.linspace(xmin, xmax, int(nb) + 1).astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _ensure_edges_from_grid(x_edges_or_centers, nbins=None, rng=(0.0, 1.0)):
    """
    Return valid histogram *edges* from edges or centers; fallback to linspace.
    """
    x = np.asarray(x_edges_or_centers) if x_edges_or_centers is not None else None
    if x is not None and x.ndim == 1 and len(x) >= 2:
        if np.all(np.diff(x) > 0):
            return x  # already edges
    if x is not None and x.ndim == 1 and len(x) >= 3:
        diffs = np.diff(x)
        inner = x[:-1] + diffs / 2.0
        first = x[0] - diffs[0] / 2.0
        last = x[-1] + diffs[-1] / 2.0
        return np.r_[first, inner, last]
    if nbins is None or nbins < 2:
        nbins = 40
    lo, hi = float(rng[0]), float(rng[1])
    return np.linspace(lo, hi, nbins + 1)

# -------------------------
# UI - file upload
# -------------------------
uploaded = st.file_uploader(
    "Upload your .dat file", type=["dat", "txt", "csv"], key="uploader_full_bestall"
)
if uploaded is None:
    st.info("Upload your file to continue.")
    st.stop()

raw = uploaded.getvalue().decode("utf-8", errors="ignore")
blocks = split_numeric_blocks_with_headers(raw)

colorscales = [
    "Viridis","Plasma","Magma","Inferno","Cividis","Turbo","IceFire","YlGnBu","Greys",
]

tabs = st.tabs(
    [
        "Joint (Heatmap + Marginals)",
        "FRET Analysis",
        "AUC Region Analyzer",
        "FRET ↔ Distance",
        "Population ratio (a/b) analysis"
    ]
)

# -------------------------
# TAB 1: Joint view
# -------------------------
with tabs[0]:
    st.subheader("Joint view: S–E heatmap + S/E histograms (from 8-col table)")
    colorscale = st.selectbox("Heatmap colorscale", colorscales, index=0, key="joint_cmap_best")
    mats = [i for i, (df, _, _, _) in enumerate(blocks) if df.shape[0] >= 10 and df.shape[1] >= 10]
    t8   = [i for i, (df, _, _, _) in enumerate(blocks) if df.shape[1] >= 8]
    if not mats or not t8:
        st.info("Need a matrix block and an 8-column block in the file.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            iM = st.selectbox(
                "Matrix block (S×E heatmap)",
                mats, key="joint_hm_block_best",
                format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})",
            )
            smooth = st.slider("Heatmap smoothing σ", 0.0, 6.0, 1.0, 0.1, key="joint_hm_smooth_best")
        with col2:
            iT = st.selectbox(
                "8-col table block (S/E)",
                t8, key="joint_t8_block_best",
                format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})",
            )
            which = st.selectbox("Histogram source", ["PIE", "Classical", "Both"], key="joint_hist_source_best")
            match_bins = st.checkbox("Match histogram bins to heatmap grid", value=True, key="joint_match_bins_best")
            style = st.selectbox("Histogram style", ["Bars", "Lines (smoothed)"], key="joint_style_best")
            smooth_bins = st.slider("Line smoothing (σ in bins)", 0.0, 3.0, 1.0, 0.2, key="joint_smooth_lines_best")

        # compact figure height
        fig_height = st.slider("Figure height (px)", 320, 900, 420, 10, key="joint_fig_height")

        # Heatmap matrix
        M = blocks[iM][0].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()
        Mplot = (
            gaussian_filter1d(gaussian_filter1d(M, sigma=smooth, axis=0), sigma=smooth, axis=1)
            if smooth > 0 else M.copy()
        )
        ny, nx = Mplot.shape  # rows=S, cols=E

        # default uniform edges & centers for the heatmap grid
        e_edges_hm = np.linspace(0, 1, nx + 1); e_centers_hm = 0.5 * (e_edges_hm[:-1] + e_edges_hm[1:])
        s_edges_hm = np.linspace(0, 1, ny + 1); s_centers_hm = 0.5 * (s_edges_hm[:-1] + s_edges_hm[1:])

        # 8-col table (S/E)
        tbl = blocks[iT][0].copy()
        base = [
            "Occur._S_Classical","S_Classical","Occur._S_PIE","S_PIE",
            "E_Classical","Occur._E_Classical","E_PIE","Occur._E_PIE",
        ]
        extra = [f"Extra_{i}" for i in range(max(0, tbl.shape[1] - 8))]
        tbl.columns = base + extra

        def col(name): return pd.to_numeric(tbl[name], errors="coerce").to_numpy()
        S_cl, W_S_cl, S_pie, W_S_pie = col("S_Classical"), col("Occur._S_Classical"), col("S_PIE"), col("Occur._S_PIE")
        E_cl, W_E_cl, E_pie, W_E_pie = col("E_Classical"), col("Occur._E_Classical"), col("E_PIE"), col("Occur._E_PIE")

        # --- helpers ---
        def hist1(x, w, edges):
            m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
            h, _ = np.histogram(x[m], bins=edges, weights=w[m])
            return h

        def _edges_from_centers_or_edges(arr, nbins=None, rng=(0.0, 1.0)):
            """Return valid, strictly increasing histogram *edges*."""
            if arr is None:
                nb = max(2, int(nbins) if nbins else 40)
                return np.linspace(float(rng[0]), float(rng[1]), nb + 1)
            a = np.asarray(arr)
            if a.ndim != 1 or a.size < 2:
                nb = max(2, int(nbins) if nbins else 40)
                return np.linspace(float(rng[0]), float(rng[1]), nb + 1)
            if np.all(np.diff(a) > 0):  # looks like edges already
                return a
            if a.size >= 3:  # centers -> edges
                d = np.diff(a)
                inner = a[:-1] + d/2.0
                first = a[0] - d[0]/2.0
                last  = a[-1] + d[-1]/2.0
                edges = np.r_[first, inner, last]
            else:  # 2 points
                edges = np.linspace(a.min(), a.max(), 3)
            if edges[-1] < edges[0]:
                edges = edges[::-1]
            return edges

        # choose binning for histograms
        if match_bins:
            e_edges = _edges_from_centers_or_edges(e_edges_hm, nbins=nx, rng=(0.0, 1.0))
            s_edges = _edges_from_centers_or_edges(s_edges_hm, nbins=ny, rng=(0.0, 1.0))
            if (len(e_edges) < 2) or (len(s_edges) < 2):
                st.warning("Heatmap grid too small to match histogram bins; using auto bins instead.")
                match_bins = False
            else:
                e_x = 0.5 * (e_edges[:-1] + e_edges[1:])
                s_y = 0.5 * (s_edges[:-1] + s_edges[1:])
        if not match_bins:
            bin_mode = st.radio("Histogram binning (for S/E)", ["Auto (rule)", "Manual"], index=0, key="joint_binmode_best")
            if bin_mode == "Auto (rule)":
                nb_e = auto_bins(np.concatenate([E_cl, E_pie]))
                nb_s = auto_bins(np.concatenate([S_cl, S_pie]))
                e_edges = np.linspace(0, 1, nb_e + 1)
                s_edges = np.linspace(0, 1, nb_s + 1)
            else:
                e_edges, _ = clamp_manual_bins_E("E min", "E max", "E bins (manual)", default_bins=80, key_prefix="joint_E")
                s_edges, _ = clamp_manual_bins_E("S min", "S max", "S bins (manual)", default_bins=80, key_prefix="joint_S")
            e_x = 0.5 * (e_edges[:-1] + e_edges[1:])
            s_y = 0.5 * (s_edges[:-1] + s_edges[1:])

        # histograms
        e_hist_cl = hist1(E_cl, W_E_cl, e_edges)
        e_hist_pie = hist1(E_pie, W_E_pie, e_edges)
        s_hist_cl = hist1(S_cl, W_S_cl, s_edges)
        s_hist_pie = hist1(S_pie, W_S_pie, s_edges)

        # --- figure layout ---
        figj = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "heatmap"}, {"type": "xy"}]],
            column_widths=[0.72, 0.28],
            row_heights=[0.22, 0.78],
            horizontal_spacing=0.03,
            vertical_spacing=0.03,
        )

        def add_hist_top(xc, yc, name):
            if style == "Bars":
                figj.add_trace(go.Bar(x=xc, y=yc, name=name, opacity=0.6, width=np.diff(e_edges)), row=1, col=1)
            else:
                ysm = gaussian_filter1d(yc, sigma=smooth_bins) if smooth_bins > 0 else yc
                figj.add_trace(go.Scatter(x=xc, y=ysm, mode="lines", name=name), row=1, col=1)

        if which in ("Classical", "Both"): add_hist_top(e_x, e_hist_cl, "Classical E")
        if which in ("PIE", "Both"):       add_hist_top(e_x, e_hist_pie, "PIE E")

        figj.add_trace(go.Heatmap(z=Mplot, coloraxis="coloraxis", showscale=True), row=2, col=1)

        def add_hist_right(yc, xc, name):
            if style == "Bars":
                figj.add_trace(go.Bar(y=yc, x=xc, orientation="h", name=name, opacity=0.6, width=np.diff(s_edges)), row=2, col=2)
            else:
                xsm = gaussian_filter1d(xc, sigma=smooth_bins) if smooth_bins > 0 else xc
                figj.add_trace(go.Scatter(y=yc, x=xsm, mode="lines", name=name), row=2, col=2)

        if which in ("Classical", "Both"): add_hist_right(s_y, s_hist_cl, "Classical S")
        if which in ("PIE", "Both"):       add_hist_right(s_y, s_hist_pie, "PIE S")

        # link axes when matching grid
        if match_bins:
            figj.update_xaxes(matches="x", row=1, col=1)
            figj.update_yaxes(matches="y", row=2, col=2)

        figj.update_xaxes(title_text="E (0–1)", row=2, col=1)
        figj.update_yaxes(title_text="S (0–1)", row=2, col=1)
        figj.update_yaxes(title_text="S (0–1)", row=2, col=2)
        figj.update_xaxes(title_text="Counts (E)", row=1, col=1)

        figj.update_layout(
            coloraxis=dict(colorscale=colorscale),
            showlegend=True,
            bargap=0,
            margin=dict(l=40, r=10, t=40, b=40),
            height=fig_height,
        )

        # --- Peak detection for autofill into calculator tab ---
        def _peak_from_hist(x_centers, y_counts):
            if len(y_counts) == 0 or np.all(np.asarray(y_counts) <= 0):
                return None
            i = int(np.nanargmax(y_counts))
            return float(x_centers[i]), i

        peak_E_cl, idx_cl = (None, None)
        peak_E_pie, idx_pie = (None, None)
        if np.any(e_hist_cl): 
            tmp = _peak_from_hist(e_x, e_hist_cl); 
            if tmp: peak_E_cl, idx_cl = tmp
        if np.any(e_hist_pie):
            tmp = _peak_from_hist(e_x, e_hist_pie);
            if tmp: peak_E_pie, idx_pie = tmp

        refine = st.checkbox("Refine peaks with Gaussian (quick fit)", value=False, key="joint_refine_peaks_best")
        if refine and (idx_cl is not None or idx_pie is not None):
            def _refine_gaussian(xc, yc, idx, win=4):
                lo = max(0, idx - win); hi = min(len(xc), idx + win + 1)
                xw = np.asarray(xc[lo:hi], dtype=float)
                yw = np.asarray(yc[lo:hi], dtype=float)
                if xw.size < 3 or np.all(yw <= 0): return float(xc[idx])
                def gfun(x, A, mu, sig, B): 
                    return A * np.exp(-0.5 * ((x - mu) / max(sig, 1e-6))**2) + B
                A0 = float(np.nanmax(yw)); mu0 = float(xc[idx])
                sig0 = 2.0 * (np.median(np.diff(xc)) if len(xc)>1 else 0.01)
                B0 = float(np.nanmin(yw))
                try:
                    popt, _ = curve_fit(gfun, xw, yw, p0=[A0, mu0, sig0, B0], maxfev=20000)
                    _, mu, _, _ = popt
                    return float(np.clip(mu, 0.0, 1.0))
                except Exception:
                    return float(xc[idx])
            if idx_cl is not None:
                peak_E_cl = _refine_gaussian(e_x, e_hist_cl, idx_cl, win=4)
            if idx_pie is not None:
                peak_E_pie = _refine_gaussian(e_x, e_hist_pie, idx_pie, win=4)

        # show peak markers
        if peak_E_cl is not None and which in ("Classical", "Both"):
            figj.add_vline(x=peak_E_cl, line_width=1.5, line_dash="dot", line_color="#2a9d8f", row=1, col=1)
        if peak_E_pie is not None and which in ("PIE", "Both"):
            figj.add_vline(x=peak_E_pie, line_width=1.5, line_dash="dot", line_color="#e76f51", row=1, col=1)

        # stash peaks for calculator tab
        st.session_state["joint_peaks"] = {
            "Classical": peak_E_cl,
            "PIE": peak_E_pie,
            "_meta": {"bin_centers": e_x.tolist()}
        }

        st.plotly_chart(figj, use_container_width=True, key="fig_joint_view")

# -------------------------
# TAB 2: FRET Analysis (single E histogram)
# -------------------------
with tabs[1]:
    st.subheader("FRET – Histogram + Best Auto-fit (PIE or Classical, E only)")
    candidates = [i for i, (df, _, _, _) in enumerate(blocks) if blocks[i][0].shape[1] >= 8]
    if not candidates:
        st.info("No 8+ column table found in this file.")
    else:
        iP = st.selectbox(
            "Pick the 8-column block",
            candidates, key="fret_block_best",
            format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})",
        )
        df = blocks[iP][0].copy()
        df.columns = [
            "Occur_S_Classical","S_Classical","Occur_S_PIE","S_PIE",
            "E_Classical","Occur_E_Classical","E_PIE","Occur_E_PIE",
        ] + [f"Extra_{i}" for i in range(max(0, df.shape[1] - 8))]

        source = st.radio("E source", ["PIE", "Classical"], index=0, horizontal=True, key="fret_source_best")
        bin_mode = st.radio("Binning", ["Auto (rule)", "Manual"], index=0, horizontal=True, key="fret_binmode_best")

        if bin_mode == "Auto (rule)":
            rule = st.selectbox("Auto-binning rule", ["Freedman–Diaconis", "Scott", "Sturges"], index=0, key="fret_rule_best")
            edges = None
        else:
            edges, _ = clamp_manual_bins_E("E min", "E max", "Bins (manual)", default_bins=80, key_prefix="fret")

        style = st.selectbox("Histogram style", ["Bars", "Lines (smoothed)"], index=0, key="fret_style_best")
        smooth_bins = st.slider("Line smoothing (σ in bins)", 0.0, 3.0, 1.0, 0.2, key="fret_smooth_best")
        do_fit = st.checkbox("Run best auto-fit", value=True, key="fret_fit_best")

        if source == "PIE":
            E = pd.to_numeric(df["E_PIE"], errors="coerce").to_numpy()
            W = pd.to_numeric(df["Occur_E_PIE"], errors="coerce").to_numpy()
            label = "PIE FRET (E)"
        else:
            E = pd.to_numeric(df["E_Classical"], errors="coerce").to_numpy()
            W = pd.to_numeric(df["Occur_E_Classical"], errors="coerce").to_numpy()
            label = "Classical FRET (E)"

        m = np.isfinite(E) & np.isfinite(W) & (W >= 0)
        E, W = E[m], W[m]
        if E.size == 0:
            st.warning("No finite E values in this block.")
        else:
            if edges is None:
                nb = auto_bins(E, rule=rule)
                xmin, xmax = float(np.nanmin(E)), float(np.nanmax(E))
                edges = np.linspace(xmin, xmax, nb + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            hist, _ = np.histogram(E, bins=edges, weights=W, density=True)

            fig = go.Figure()
            if style == "Bars":
                fig.add_bar(x=centers, y=hist, width=np.diff(edges), name=f"{label} histogram", opacity=0.55)
            else:
                y = gaussian_filter1d(hist, sigma=smooth_bins) if smooth_bins > 0 else hist
                fig.add_trace(go.Scatter(x=centers, y=y, mode="lines", name=label))

            if do_fit:
                fit = best_autofit(centers, hist, edges[0], edges[-1])
                xs = np.linspace(edges[0], edges[-1], 1000)
                if fit:
                    if fit["model"] == "1G":
                        fig.add_trace(
                            go.Scatter(x=xs, y=gaussian(xs, *fit["params"]),
                                       mode="lines", name=f"Best fit 1G (R²={fit['R2']:.3f})")
                        )
                    else:
                        y0, m1, s1, A1, m2, s2, A2 = fit["params"]
                        fig.add_trace(
                            go.Scatter(x=xs, y=gaussian2(xs, *fit["params"]),
                                       mode="lines", name=f"Best fit 2G (R²={fit['R2']:.3f})")
                        )
                        fig.add_trace(
                            go.Scatter(x=xs, y=gaussian(xs, y0, m1, s1, A1) - y0,
                                       mode="lines", name="Component 1", line=dict(dash="dash"))
                        )
                        fig.add_trace(
                            go.Scatter(x=xs, y=gaussian(xs, y0, m2, s2, A2) - y0,
                                       mode="lines", name="Component 2", line=dict(dash="dash"))
                        )
            fig.update_layout(xaxis_title="FRET efficiency, E", yaxis_title="Density")
            st.plotly_chart(fig, use_container_width=True, key="fig_fret_analysis")

# -------------------------
# -------------------------
# TAB 3: AUC Region Analyzer
# -------------------------
with tabs[2]:
    st.subheader("AUC Region Analyzer – stacked histograms")

    files = st.file_uploader(
        "Upload one or more .dat files",
        type=["dat", "txt", "csv"],
        accept_multiple_files=True,
        key="auc_uploader_best",
    )

    if not files:
        st.info("Upload multiple files to start.")
    else:
        source = st.radio("E source", ["PIE", "Classical"], index=0, horizontal=True, key="auc_source_best")

        # NEW: y-axis control (plotting only)
        y_mode = st.radio(
            "Histogram y-axis (plotting)",
            ["Counts", "Density (area=1)"],
            index=1, horizontal=True, key="auc_y_mode"
        )
        plot_as_density = (y_mode == "Density (area=1)")

        # Keep your preferred 8-col block picker
        preferred_idx = st.number_input(
            "Preferred 8-column block index (applies to each file)",
            min_value=0, value=1, step=1, key="auc_pref_idx"
        )

        bin_mode = st.radio(
            "Binning", ["Auto (rule)", "Manual (fixed)"], index=0, horizontal=True, key="auc_binmode_best"
        )
        if bin_mode == "Auto (rule)":
            rule = st.selectbox(
                "Auto-binning rule (shared)", ["Freedman–Diaconis", "Scott", "Sturges"],
                index=0, key="auc_rule_best"
            )
            manual_edges = None
        else:
            manual_edges, _ = clamp_manual_bins_E(
                "Range min (E)", "Range max (E)", "Number of bins (shared)",
                default_bins=80, key_prefix="auc"
            )

        region = st.slider("AUC region (E-range)", 0.0, 1.0, (0.70, 1.00), 0.01, key="auc_region_best")

        # ---- Load datasets ----
        datasets = []
        for f in files:
            raw2 = f.getvalue().decode("utf-8", errors="ignore")
            blks = split_numeric_blocks_with_headers(raw2)
            cand = [i for i, (df, _, _, _) in enumerate(blks) if blks[i][0].shape[1] >= 8]
            if not cand:
                continue
            pick = int(preferred_idx)
            chosen = pick if pick in cand else cand[0]

            df = blks[chosen][0].copy()
            df.columns = [
                "Occur_S_Classical","S_Classical","Occur_S_PIE","S_PIE",
                "E_Classical","Occur_E_Classical","E_PIE","Occur_E_PIE",
            ] + [f"Extra_{i}" for i in range(max(0, df.shape[1]-8))]

            if source == "PIE":
                E = pd.to_numeric(df["E_PIE"], errors="coerce").to_numpy()
                W = pd.to_numeric(df["Occur_E_PIE"], errors="coerce").to_numpy()
            else:
                E = pd.to_numeric(df["E_Classical"], errors="coerce").to_numpy()
                W = pd.to_numeric(df["Occur_E_Classical"], errors="coerce").to_numpy()

            m = np.isfinite(E) & np.isfinite(W) & (W >= 0)
            if np.any(m):
                datasets.append((f.name, chosen, E[m], W[m]))

        if not datasets:
            st.warning("No valid 8-column blocks found across the files.")
        else:
            # Shared binning
            if manual_edges is None:
                all_E = np.concatenate([E for _, _, E, _ in datasets])
                xmin = float(np.nanmin(all_E)); xmax = float(np.nanmax(all_E))
                nb = auto_bins(all_E, rule=rule)
                edges = np.linspace(xmin, xmax, nb + 1)
            else:
                edges = manual_edges

            centers = 0.5 * (edges[:-1] + edges[1:])
            binw = np.diff(edges)[0]

            rows = []
            for i, (nm, blk_idx, E, W) in enumerate(datasets):
                # Always compute COUNTS for correctness
                counts, _ = np.histogram(E, bins=edges, weights=W, density=False)
                area_counts = float((counts * binw).sum())  # total counts area (counts × bin width)
                # Fraction within region (independent of plotting mode)
                rmin, rmax = region
                mask_bins = (centers >= rmin) & (centers < rmax)
                auc_counts   = float((counts[mask_bins] * binw).sum())
                auc_fraction = (counts[mask_bins].sum() / counts.sum()) if counts.sum() > 0 else np.nan

                # Plot as desired
                if plot_as_density and area_counts > 0:
                    y_plot = counts / area_counts   # probability per bin; area under curve = 1
                    y_label = "Density (area=1)"
                else:
                    y_plot = counts
                    y_label = "Counts per bin"

                fig = go.Figure()
                fig.add_bar(x=centers, y=y_plot, width=np.diff(edges), name=nm, opacity=0.7)
                fig.add_vrect(x0=rmin, x1=rmax, fillcolor="LightSalmon", opacity=0.25,
                              layer="below", line_width=0)
                fig.update_layout(
                    title=f"{nm} — block {blk_idx}",
                    xaxis_title="FRET efficiency, E",
                    yaxis_title=y_label,
                    margin=dict(l=40, r=10, t=40, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"fig_auc_{i}")

                rows.append({
                    "file": nm, "block_used": blk_idx,
                    "bins": len(edges) - 1, "E_min": edges[0], "E_max": edges[-1],
                    "region_min": rmin, "region_max": rmax,
                    "auc_counts": auc_counts,               # absolute area (counts × bin width)
                    "auc_fraction": auc_fraction,           # fraction in region (0–1)
                    "plot_yaxis": y_label,
                })

            df_auc = pd.DataFrame(rows)
            st.subheader("AUC summary for selected region")
            st.dataframe(df_auc, use_container_width=True, key="tbl_auc_summary")
            st.download_button(
                "Download AUC summary (CSV)",
                df_auc.to_csv(index=False).encode(),
                "auc_region_summary.csv",
                "text/csv",
                key="dl_auc_summary",
            )

# -------------------------
# TAB 4: FRET ↔ Distance calculator (with Joint-peak autofill)
# -------------------------
with tabs[3]:
    st.subheader("FRET ↔ Distance Calculator")

    colA, colB = st.columns([1, 1])
    with colA:
        direction = st.radio("Conversion", ["E → r (nm)", "r (nm) → E"], index=0, horizontal=True, key="fr_dir")
        R0 = st.number_input("Förster radius R₀ (nm)", min_value=0.1, max_value=20.0, value=6.4, step=0.1, key="fr_R0")
        st.caption("Tip: For this lab, use R₀ ≈ 6.4 nm (given in the protocol).")
    with colB:
        show_plot = st.checkbox("Show quick plot", value=True, key="fr_plot")

    # Optional autofill from Joint tab peaks
    joint_peaks = st.session_state.get("joint_peaks", {})
    can_autofill = direction.startswith("E") and any(joint_peaks.get(k) is not None for k in ("Classical","PIE"))
    if can_autofill:
        c1, c2 = st.columns(2)
        with c1:
            src_peak = st.selectbox("Use peak E from Joint tab", ["Classical", "PIE"], key="fr_src_peak")
        with c2:
            if joint_peaks.get(src_peak) is not None:
                if st.button("Send peak to calculator", key="fr_send_peak"):
                    val = f"{joint_peaks[src_peak]:.4f}"
                    prev = st.session_state.get("fr_text", "")
                    st.session_state["fr_text"] = (val if not prev.strip() else prev.strip() + ", " + val)
            else:
                st.info(f"No peak detected for {src_peak}. Compute hist on Joint tab.")

    st.markdown("**Input** (single value, list, or CSV):")
    col1, col2 = st.columns(2)
    with col1:
        txt = st.text_area(
            "Paste values",
            value="0.2, 0.4, 0.65, 0.85" if direction.startswith("E") else "3.5 5.0 6.4 8.0",
            height=100,
            key="fr_text"
        )
    with col2:
        up = st.file_uploader("Or upload CSV", type=["csv"], key="fr_csv")
        help_col = "E" if direction.startswith("E") else "r"
        st.caption(f"If you upload CSV, include a column named **{help_col}**.")

    def _parse_numbers(s: str):
        if not s or not s.strip():
            return np.array([])
        parts = [p for p in re.split(r"[,\s;]+", s.strip()) if p]
        vals = []
        for p in parts:
            try: vals.append(float(p))
            except Exception: vals.append(np.nan)
        return np.array(vals, dtype=float)

    # Load inputs
    arr = np.array([], dtype=float)
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            if direction.startswith("E"):
                if "E" in df_in.columns:
                    arr = pd.to_numeric(df_in["E"], errors="coerce").to_numpy()
                else:
                    st.error("CSV must contain a column named 'E'.")
            else:
                if "r" in df_in.columns:
                    arr = pd.to_numeric(df_in["r"], errors="coerce").to_numpy()
                else:
                    st.error("CSV must contain a column named 'r'.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    else:
        arr = _parse_numbers(txt)

    # Conversion functions
    def fret_to_distance(E, R0):
        E = np.asarray(E, dtype=float)
        r = np.full_like(E, np.nan)
        mask = (E > 0.0) & (E < 1.0) & np.isfinite(E)
        r[mask] = R0 * ((1.0 / E[mask] - 1.0) ** (1.0 / 6.0))
        return r

    def distance_to_fret(r, R0):
        r = np.asarray(r, dtype=float)
        E = np.full_like(r, np.nan)
        mask = (r > 0.0) & np.isfinite(r) & (R0 > 0.0)
        E[mask] = 1.0 / (1.0 + (r[mask] / R0) ** 6)
        return E

    # Do conversion
    if direction.startswith("E"):
        E = arr
        r = fret_to_distance(E, R0)
        df_out = pd.DataFrame({"E": E, "r_nm": r})
        valid = np.isfinite(r).sum()
        st.write(f"Converted **{valid}/{len(r)}** values (E must be in (0,1)).")
    else:
        r = arr
        E = distance_to_fret(r, R0)
        df_out = pd.DataFrame({"r_nm": r, "E": E})
        valid = np.isfinite(E).sum()
        st.write(f"Converted **{valid}/{len(E)}** values (r must be > 0).")

    # Show table
    st.dataframe(df_out, use_container_width=True, height=240)

    # Download
    st.download_button(
        "Download results (CSV)",
        df_out.to_csv(index=False).encode(),
        file_name="fret_distance_conversion.csv",
        mime="text/csv",
        key="fr_dl"
    )

    # Quick plot
    if show_plot:
        if direction.startswith("E"):
            xs = np.linspace(0.001, 0.999, 400)
            ys = fret_to_distance(xs, R0)
            fig = go.Figure()
            fig.add_scatter(x=xs, y=ys, mode="lines", name="r(E)")
            if len(arr) > 0:
                fig.add_scatter(x=E, y=r, mode="markers", name="Your points")
            fig.update_layout(
                xaxis_title="FRET efficiency E",
                yaxis_title="Distance r (nm)",
                height=360, margin=dict(l=40, r=20, t=20, b=40)
            )
            st.plotly_chart(fig, use_container_width=True, key="fr_plot1")
        else:
            xs = np.linspace(max(0.1, R0*0.2), R0*3.0, 400)
            ys = distance_to_fret(xs, R0)
            fig = go.Figure()
            fig.add_scatter(x=xs, y=ys, mode="lines", name="E(r)")
            if len(arr) > 0:
                fig.add_scatter(x=r, y=E, mode="markers", name="Your points")
            fig.update_layout(
                xaxis_title="Distance r (nm)",
                yaxis_title="FRET efficiency E",
                height=360, margin=dict(l=40, r=20, t=20, b=40)
            )
            st.plotly_chart(fig, use_container_width=True, key="fr_plot2")

    st.caption("Formulas:  r = R₀ · (1/E − 1)^(1/6)  and  E = 1 / (1 + (r/R₀)^6)")

# -------------------------
# -------------------------
# TAB: Population ratio (a/b) analysis — general titration
# -------------------------
with tabs[4]:
    st.subheader("Population ratio (a/b) and normalized fraction vs condition")

    files = st.file_uploader(
        "Upload all measurement files (.dat) including your reference (e.g. lowest condition)",
        type=["dat", "txt", "csv"],
        accept_multiple_files=True,
        key="ratio_uploader",
    )

    if not files:
        st.info("Upload multiple measurement files (e.g. 0, 0.5, 1.5, ...) to continue.")
        st.stop()

    # User-specified label for x-axis (can be anything: [Urea] (M), Temperature (°C), etc.)
    x_label = st.text_input("X-axis label (condition)", value="[Condition]", key="ratio_xlabel")

    pref_blk = st.number_input("Preferred 8-column block index", min_value=0, value=1, step=1, key="ratio_blk")

    bin_mode = st.radio("Binning for E", ["Auto (rule)", "Manual"], index=0, horizontal=True, key="ratio_binmode")
    if bin_mode == "Auto (rule)":
        rule = st.selectbox("Auto-binning rule", ["Freedman–Diaconis", "Scott", "Sturges"], index=0, key="ratio_rule")
        user_edges = None
    else:
        user_edges, _ = clamp_manual_bins_E("E min", "E max", "Bins", default_bins=80, key_prefix="ratio_bins")

    closed_region = st.slider("Closed-peak fit region (E-range)", 0.0, 1.0, (0.70, 1.00), 0.01, key="ratio_region")
    show_fit_plots = st.checkbox("Show PIE histogram + Gaussian fit for each file", value=True, key="ratio_showfits")

    # ---------- NEW: robust auto-parse + editable 'condition' table ----------
    import re as _re

    def parse_condition_from_name(name: str) -> float | None:
        """
        Try to extract a concentration-like value from the filename and return it in M (mol/L).
        Handles '..._0M', '1.5M_...', '..._750mM', '..._250uM'/'µM', etc.
        If we find multiple numbers, prefer the one closest to substring 'gdmcl' (if present),
        otherwise take the last number in the string.
        """
        s = name.replace(",", ".").lower()
        hits = []
        for m in _re.finditer(r'(\d+(?:\.\d+)?)\s*([munp]?m)?', s):
            val = float(m.group(1))
            unit = (m.group(2) or "m").lower()
            # normalize some common variants
            unit = unit.replace("µ", "u")
            factor = {
                "m": 1.0,   # interpret bare number as 'M' (matches many of your files like "0M", "1.5M")
                "mm": 1e-3,
                "um": 1e-6,
                "nm": 1e-9,
            }.get(unit, 1.0)
            hits.append((m.start(), val * factor))

        if not hits:
            return None

        tag = s.find("gdmcl")  # if present, use the number closest to 'gdmcl'
        if tag != -1:
            idx = min(range(len(hits)), key=lambda i: abs(hits[i][0] - tag))
            return hits[idx][1]
        # otherwise, take the last number (often the suffix)
        return hits[-1][1]

    # Build initial table and let user edit/confirm (in M)
    cfg_rows = [{"file": f.name, "condition": parse_condition_from_name(f.name)} for f in files]
    df_cfg = pd.DataFrame(cfg_rows)
    st.write("**Edit or confirm the condition values (in molar, M):**")
    df_cfg_edit = st.data_editor(
        df_cfg,
        key="ratio_cfg_editor",
        use_container_width=True,
        hide_index=True,
        column_config={
            "condition": st.column_config.NumberColumn(
                "Condition (M)", min_value=0.0, step=0.01, format="%.6f"
            )
        },
    )
    # ------------------------------------------------------------------------

    # --- helpers for E/W extraction and PIE-closed fit ---
    def read_E_and_W_from_8col(df, use_pie: bool):
        df = df.copy()
        df.columns = [
            "Occur_S_Classical", "S_Classical", "Occur_S_PIE", "S_PIE",
            "E_Classical", "Occur_E_Classical", "E_PIE", "Occur_E_PIE",
        ] + [f"Extra_{i}" for i in range(max(0, df.shape[1] - 8))]

        if use_pie:
            E = pd.to_numeric(df["E_PIE"], errors="coerce").to_numpy()
            W = pd.to_numeric(df["Occur_E_PIE"], errors="coerce").to_numpy()
        else:
            E = pd.to_numeric(df["E_Classical"], errors="coerce").to_numpy()
            W = pd.to_numeric(df["Occur_E_Classical"], errors="coerce").to_numpy()

        m = np.isfinite(E) & np.isfinite(W) & (W >= 0)
        return E[m], W[m]

    def fit_closed_peak_area_PIE(E, W, edges, fit_range):
        """
        Fit a 1-Gaussian to the PIE histogram restricted to fit_range (high-E window)
        and return 'a' as the area (counts) under that Gaussian.
        """
        counts, _ = np.histogram(E, bins=edges, weights=W, density=False)
        bw = np.diff(edges)[0]
        x = 0.5 * (edges[:-1] + edges[1:])
        y = counts / bw  # convert to counts per E unit for fitting

        rmin, rmax = fit_range
        m = (x >= rmin) & (x <= rmax) & np.isfinite(y)
        if m.sum() < 5 or y[m].max() <= 0:
            return np.nan, (x, y), None

        xs, ys = x[m], y[m]
        y0_0 = max(1e-6, np.percentile(ys, 10))
        mu_0 = xs[np.argmax(ys)]
        sigma_0 = max(0.5 * bw, (np.percentile(xs, 75) - np.percentile(xs, 25)) / 1.349)
        A_0 = float(np.trapz(np.maximum(ys - y0_0, 0), xs))

        xmin, xmax = xs.min(), xs.max()
        sig_min = max(0.25 * bw, 1e-4)
        sig_max = max(0.5, (xmax - xmin))
        lower = [0.0, xmin, sig_min, 0.0]
        upper = [ys.max() * 2 + 5.0, xmax, sig_max, np.inf]

        try:
            popt, _ = curve_fit(
                gaussian, xs, ys,
                p0=[y0_0, mu_0, sigma_0, max(A_0, 1e-6)],
                bounds=(lower, upper), maxfev=100000
            )
            return float(popt[3]), (x, y), popt  # A is the area in counts (after scaling back by bin width if needed)
        except Exception:
            return np.nan, (x, y), None

    # Explanation panel (kept)
    with st.expander("ℹ️ What do a, b, a/b, and normalized fraction mean?"):
        st.markdown("""
        **a** → *Area of the high-FRET (closed-state) population*  
        • From a Gaussian fit to the **PIE FRET** histogram within the selected high-E window.

        **b** → *Total area under the classical FRET histogram*  
        • Sum of all counts (open + closed).

        **a / b** → *Fraction of closed molecules* for that condition.

        **Normalized fraction** → *(a/b) × (b₀/a₀)*  
        • Scales to a chosen reference (the first/lowest condition). The reference = 1.0.
        """)

    # ---- collect (condition, a, b) ----
    rows = []
    fit_figs = []
    for f in files:
        name = f.name
        # Use the user-edited condition (M) for this file
        try:
            cond_val = float(df_cfg_edit.loc[df_cfg_edit["file"] == name, "condition"].values[0])
        except Exception:
            cond_val = np.nan

        raw2 = f.getvalue().decode("utf-8", errors="ignore")
        blks = split_numeric_blocks_with_headers(raw2)
        cand = [i for i, (df, _, _, _) in enumerate(blks) if blks[i][0].shape[1] >= 8]
        if not cand:
            continue

        chosen = int(pref_blk) if int(pref_blk) in cand else cand[0]
        df8 = blks[chosen][0]

        # PIE (closed peak fit → a)
        E_pie, W_pie = read_E_and_W_from_8col(df8, True)
        if user_edges is None:
            nb = auto_bins(E_pie, rule=rule) if bin_mode == "Auto (rule)" else 80
            edges_pie = np.linspace(float(np.nanmin(E_pie)), float(np.nanmax(E_pie)), nb + 1)
        else:
            edges_pie = user_edges
        a_counts, (xpie, ypie), popt = fit_closed_peak_area_PIE(E_pie, W_pie, edges_pie, closed_region)

        # Classical (total → b)
        E_cl, W_cl = read_E_and_W_from_8col(df8, False)
        if user_edges is None:
            nb2 = auto_bins(E_cl, rule=rule) if bin_mode == "Auto (rule)" else 80
            edges_cl = np.linspace(float(np.nanmin(E_cl)), float(np.nanmax(E_cl)), nb2 + 1)
        else:
            edges_cl = user_edges
        counts_cl, _ = np.histogram(E_cl, bins=edges_cl, weights=W_cl, density=False)
        b_counts = float(np.nansum(counts_cl))

        rows.append(dict(file=name, condition=cond_val, block=chosen, a=a_counts, b=b_counts))

        # Optional per-file figure
        if show_fit_plots:
            fig = go.Figure()
            cts_pie, _ = np.histogram(E_pie, bins=edges_pie, weights=W_pie, density=False)
            cx = 0.5 * (edges_pie[:-1] + edges_pie[1:])
            fig.add_bar(x=cx, y=cts_pie, width=np.diff(edges_pie), name="PIE counts", opacity=0.5)

            if np.isfinite(a_counts) and popt is not None:
                xs = np.linspace(edges_pie[0], edges_pie[-1], 800)
                y_fit_density = gaussian(xs, *popt)      # counts per E
                y_fit_counts = y_fit_density * np.diff(edges_pie)[0]  # convert to counts per bin for plotting
                fig.add_scatter(x=xs, y=y_fit_counts, mode="lines",
                                name=f"Gaussian fit (a≈{a_counts:.0f} counts)")
            fig.add_vrect(x0=closed_region[0], x1=closed_region[1],
                          fillcolor="salmon", opacity=0.25, line_width=0)
            fig.update_layout(title=name, xaxis_title="E", yaxis_title="Counts",
                              margin=dict(l=40, r=10, t=40, b=40))
            fit_figs.append(fig)

    if not rows:
        st.warning("No valid datasets found.")
        st.stop()

    df = pd.DataFrame(rows)

    # Reference is the smallest non-NaN condition value
    if df["condition"].notna().any():
        ref_idx = df["condition"].idxmin()
        a0 = float(df.loc[ref_idx, "a"])
        b0 = float(df.loc[ref_idx, "b"])
    else:
        ref_idx, a0, b0 = df.index[0], df.loc[df.index[0], "a"], df.loc[df.index[0], "b"]

    df = df.sort_values("condition")
    df["a_over_b"] = df["a"] / df["b"]
    df["fraction_norm"] = df["a_over_b"] * (b0 / a0) if (a0 > 0 and b0 > 0) else np.nan

    st.markdown("**Summary of a, b, a/b, and normalized fraction**")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download (CSV)",
        df.to_csv(index=False).encode(),
        "population_ratio_summary.csv",
        "text/csv",
        key="ratio_dl",
    )

    # Plots vs condition (if conditions are present)
    if df["condition"].notna().any():
        fig1 = go.Figure()
        fig1.add_scatter(x=df["condition"], y=df["a_over_b"], mode="lines+markers", name="a/b")
        fig1.update_layout(xaxis_title=x_label, yaxis_title="a/b", title="a/b vs condition")
        st.plotly_chart(fig1, use_container_width=True, key="ratio_plot1")

        fig2 = go.Figure()
        fig2.add_scatter(x=df["condition"], y=df["fraction_norm"], mode="lines+markers",
                         name="(a/b) × (b₀/a₀)")
        fig2.update_layout(xaxis_title=x_label, yaxis_title="Fraction (normalized)",
                           title="Normalized fraction vs condition")
        st.plotly_chart(fig2, use_container_width=True, key="ratio_plot2")

    if show_fit_plots and fit_figs:
        st.markdown("**PIE histograms with Gaussian fits (closed peak)**")
        for i, fig in enumerate(fit_figs):
            st.plotly_chart(fig, use_container_width=True, key=f"ratio_fitfig_{i}")
