import re, io, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

st.set_page_config(page_title="FRET Analyzer – FULL (Best Auto-fit + Manual bins)", layout="wide")
st.title("FRET Analyzer – FULL (Best Auto-fit + Manual bins)")

# =========================
# Parsing
# =========================
def split_numeric_blocks_with_headers(text: str):
    """Split a mixed .dat file into numeric blocks (tables/matrices)."""
    text = text.replace(",", ".")
    lines = text.splitlines()
    blocks = []
    cur = []; header_buf = []; start_idx = None
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
        cur = []; start_idx = None

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

# =========================
# Models & utilities
# =========================
def gaussian(x, y0, mu, sigma, A):
    amp = A / (sigma * np.sqrt(2*np.pi))
    return y0 + amp * np.exp(-0.5 * ((x - mu)/sigma)**2)

def gaussian2(x, y0, mu1, s1, A1, mu2, s2, A2):
    return gaussian(x, y0, mu1, s1, A1) - y0 + gaussian(x, y0, mu2, s2, A2)

def r2_score(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.nansum((y - yhat)**2)
    ss_tot = np.nansum((y - np.nanmean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

def aicc(n, rss, k):
    if n <= k + 1:
        return np.inf
    return n*np.log(rss/n) + 2*k + (2*k*(k+1))/(n - k - 1)

def auto_bins(x, rule="Freedman–Diaconis"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 40
    if rule == "Freedman–Diaconis":
        iqr = np.subtract(*np.percentile(x, [75,25]))
        if iqr == 0:
            return int(np.clip(np.sqrt(n), 10, 200))
        bw = 2 * iqr * (n ** (-1/3))
    elif rule == "Scott":
        std = np.nanstd(x, ddof=1)
        if std == 0:
            return int(np.clip(np.sqrt(n), 10, 200))
        bw = 3.5 * std * (n ** (-1/3))
    else:
        return int(np.clip(np.ceil(np.log2(n)+1), 10, 200))
    rng = np.nanmax(x) - np.nanmin(x)
    if rng <= 0:
        return int(np.clip(np.sqrt(n), 10, 200))
    return int(np.clip(np.ceil(rng/bw), 10, 200))

def best_autofit(centers, hist, xmin, xmax):
    """Try 1-Gaussian and 2-Gaussian; pick by AICc with simple separation sanity."""
    z = np.asarray(hist, float); x = np.asarray(centers, float)
    m = np.isfinite(z) & np.isfinite(x)
    x, z = x[m], z[m]
    if x.size < 8 or np.count_nonzero(z) < 5:
        return None

    z_s = gaussian_filter1d(z, sigma=max(1, int(len(z)*0.03)))
    binw = np.median(np.diff(x))
    sig_min = max(0.5*binw, 1e-4)
    sig_max = max((xmax-xmin)/1.25, sig_min*2)
    y0_min = 0.0
    y0_max = max(z.max()*2, 5.0)

    # --- 1G init
    mu0 = float(x[int(np.argmax(z_s))])
    y00 = max(1e-6, float(np.quantile(z, 0.1)))
    support = z_s >= 0.1*z_s.max()
    xw = x[support] if np.any(support) else x
    q25, q75 = np.quantile(xw, [0.25, 0.75])
    sigma0 = max(sig_min, (q75-q25)/1.349)
    A0 = float(np.trapz(np.maximum(z - y00, 0), x))
    lower1 = [y0_min, xmin, sig_min, 0.0]
    upper1 = [y0_max, xmax, sig_max, np.inf]
    p1, rss1, perr1 = None, np.inf, None
    try:
        p1_opt, pc1 = curve_fit(gaussian, x, z, p0=[y00, mu0, sigma0, max(A0,1e-6)],
                                bounds=(lower1, upper1), maxfev=100000)
        yhat1 = gaussian(x, *p1_opt); rss1 = np.nansum((z - yhat1)**2)
        perr1 = np.sqrt(np.diag(pc1)); p1 = p1_opt
    except Exception:
        pass

    # --- 2G init
    peaks, _ = find_peaks(z_s, distance=max(2, int(0.07/np.mean(np.diff(x)))))
    peaks = peaks[np.argsort(z_s[peaks])][::-1]
    if len(peaks) >= 2:
        mu1_0, mu2_0 = x[peaks[:2]]
        if mu1_0 > mu2_0: mu1_0, mu2_0 = mu2_0, mu1_0
    else:
        mu1_0 = float(np.clip(mu0 - 0.1*(xmax-xmin), xmin, xmax))
        mu2_0 = float(np.clip(mu0 + 0.1*(xmax-xmin), xmin, xmax))
    s1_0 = s2_0 = max(sig_min*1.5, sigma0*0.8)
    A1_0 = A2_0 = max(A0/2, 1e-6)
    lower2 = [y0_min, xmin, sig_min, 0.0, xmin, sig_min, 0.0]
    upper2 = [y0_max, xmax, sig_max, np.inf, xmax, sig_max, np.inf]
    p2, rss2, perr2 = None, np.inf, None
    try:
        p2_opt, pc2 = curve_fit(gaussian2, x, z, p0=[y00, mu1_0, s1_0, A1_0, mu2_0, s2_0, A2_0],
                                bounds=(lower2, upper2), maxfev=150000)
        yhat2 = gaussian2(x, *p2_opt); rss2 = np.nansum((z - yhat2)**2)
        perr2 = np.sqrt(np.diag(pc2)); p2 = p2_opt
    except Exception:
        pass

    k1, k2 = 4, 7
    n = len(x)
    aicc1 = n*np.log(rss1/n) + 2*k1 + (2*k1*(k1+1))/(n - k1 - 1) if np.isfinite(rss1) and rss1>0 else np.inf
    aicc2 = n*np.log(rss2/n) + 2*k2 + (2*k2*(k2+1))/(n - k2 - 1) if np.isfinite(rss2) and rss2>0 else np.inf

    sep_ok = False
    if p2 is not None:
        _, m1, s1, A1, m2, s2, A2 = p2
        sep_ok = abs(m2-m1) > 1.2*max(s1, s2)

    if p2 is not None and sep_ok and aicc2 + 2 < aicc1:
        yhat = gaussian2(x, *p2)
        return {"model":"2G","params":p2,"perr":perr2,"R2":r2_score(z, yhat),"aicc":aicc2,"yhat":(x,yhat)}
    elif p1 is not None:
        yhat = gaussian(x, *p1)
        return {"model":"1G","params":p1,"perr":perr1,"R2":r2_score(z, yhat),"aicc":aicc1,"yhat":(x,yhat)}
    else:
        return None

# =========================
# App
# =========================
uploaded = st.file_uploader("Upload your .dat file (for tabs 1–5)", type=["dat","txt","csv"], key="uploader_full_bestall")
if uploaded is None:
    st.info("Upload your file to continue."); st.stop()

raw = uploaded.getvalue().decode("utf-8", errors="ignore")
blocks = split_numeric_blocks_with_headers(raw)

colorscales = ["Viridis","Plasma","Magma","Inferno","Cividis","Turbo","IceFire","YlGnBu","Greys"]

tabs = st.tabs([
    "Heatmap",
    "Histogram (single)",
    "Overlay",
    "Joint (Heatmap + Marginals)",
    "FRET Analysis",
    "Compare multiple files (E)",
    "AUC Region Analyzer"
])

# ---- Heatmap ----
with tabs[0]:
    st.subheader("Correlogram Heatmap")
    mats = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[0] >= 10 and df.shape[1] >= 10]
    if not mats: mats = list(range(len(blocks)))
    iM = st.selectbox("Choose matrix block", mats, key="hm_block_best",
                      format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
    dfm = blocks[iM][0].astype(float).replace([np.inf, -np.inf], np.nan)
    c1,c2,c3 = st.columns(3)
    with c1: zmin = st.number_input("zmin (0=auto)", value=0.0, key="hm_zmin_best")
    with c2: zmax = st.number_input("zmax (0=auto)", value=0.0, key="hm_zmax_best")
    with c3: cmap = st.selectbox("Colorscale", colorscales, index=0, key="hm_cmap_best")
    smooth = st.slider("Gaussian smoothing (σ)", 0.0, 6.0, 1.0, 0.1, key="hm_smooth_best")
    arr = dfm.to_numpy()
    arrp = gaussian_filter1d(gaussian_filter1d(arr, sigma=smooth, axis=0), sigma=smooth, axis=1) if smooth>0 else arr
    fig = px.imshow(arrp, origin="lower", aspect="auto", color_continuous_scale=cmap,
                    zmin=None if zmin<=0 else zmin, zmax=None if zmax<=0 else zmax,
                    labels=dict(x="E bins (columns)", y="S bins (rows)", color="Counts"))
    st.plotly_chart(fig, use_container_width=True)

# ---- Histogram (single) ----
with tabs[1]:
    st.subheader("Histogram + Best Auto-fit (single dataset)")
    tbls = list(range(len(blocks)))
    iT = st.selectbox("Choose block", tbls, index=tbls[-1] if tbls else 0, key="single_block_best",
                      format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
    dft = blocks[iT][0].copy()
    templ = st.radio("Column template", ["Generic (C0..)", "FRET 8-cols (named)"], index=1, key="single_template_best")
    if templ == "FRET 8-cols (named)" and dft.shape[1] >= 8:
        base = ["Occur._S_Classical","S_Classical","Occur._S_PIE","S_PIE",
                "E_Classical","Occur._E_Classical","E_PIE","Occur._E_PIE"]
        extra = [f"Extra_{i}" for i in range(dft.shape[1]-8)]
        dft.columns = base + extra
    else:
        dft.columns = [f"C{j}" for j in range(dft.shape[1])]

    c1,c2,c3,c4 = st.columns(4)
    with c1: x_col = st.selectbox("Values column (x)", dft.columns, index=0, key="single_xcol_best")
    with c2: w_col = st.selectbox("Weights (optional)", ["(none)"]+list(dft.columns), index=0, key="single_wcol_best")
    with c3: bin_mode = st.radio("Binning", ["Auto (rule)","Manual"], index=0, key="single_binmode_best")
    if bin_mode == "Auto (rule)":
        with c4: rule  = st.selectbox("Auto-binning rule", ["Freedman–Diaconis","Scott","Sturges"], index=0, key="single_rule_best")
    else:
        with c4: nb_manual = st.number_input("Bins (manual)", 5, 400, 80, 1, key="single_bins_best")
        c5,c6 = st.columns(2)
        with c5: xmin_man = st.number_input("Range min", value=0.0, step=0.01, key="single_xmin_best")
        with c6: xmax_man = st.number_input("Range max", value=1.0, step=0.01, key="single_xmax_best")
    style = st.selectbox("Histogram style", ["Bars","Lines (smoothed)"], index=0, key="single_style_best")
    smooth_bins = st.slider("Line smoothing (σ in bins)", 0.0, 3.0, 1.0, 0.2, key="single_smooth_best")
    do_fit = st.checkbox("Run best auto-fit (1G/2G)", value=True, key="single_autofit_best")

    x = pd.to_numeric(dft[x_col], errors="coerce").to_numpy()
    if w_col != "(none)":
        w = pd.to_numeric(dft[w_col], errors="coerce").to_numpy()
        w = np.where(np.isfinite(w) & (w>0), w, 0.0)
    else:
        w = None
    m = np.isfinite(x)
    if not np.any(m):
        st.warning("No finite values in the chosen column.")
    else:
        xmin = float(np.nanmin(x[m])); xmax = float(np.nanmax(x[m]))
        if bin_mode == "Auto (rule)":
            nb = auto_bins(x[m], rule=rule); edges = np.linspace(xmin, xmax, nb+1)
        else:
            x0, x1 = float(xmin_man), float(xmax_man)
            if x1 <= x0: x1 = x0 + 1e-6
            nb = int(nb_manual); edges = np.linspace(x0, x1, nb+1)
        centers = 0.5*(edges[:-1]+edges[1:])
        hist, _ = np.histogram(x[m], bins=edges, weights=w if w is not None else None, density=True)
        figH = go.Figure()
        if style == "Bars":
            figH.add_bar(x=centers, y=hist, width=np.diff(edges), name="Histogram", opacity=0.55)
        else:
            y = gaussian_filter1d(hist, sigma=smooth_bins) if smooth_bins>0 else hist
            figH.add_trace(go.Scatter(x=centers, y=y, mode="lines", name="Histogram"))
        if do_fit:
            fit = best_autofit(centers, hist, edges[0], edges[-1])
            if fit:
                xs = np.linspace(edges[0], edges[-1], 1000)
                if fit["model"] == "1G":
                    figH.add_trace(go.Scatter(x=xs, y=gaussian(xs, *fit["params"]), mode="lines", name=f"Best fit 1G (R²={fit['R2']:.3f})"))
                else:
                    y0,m1,s1,A1,m2,s2,A2 = fit["params"]
                    figH.add_trace(go.Scatter(x=xs, y=gaussian2(xs, *fit["params"]), mode="lines", name=f"Best fit 2G (R²={fit['R2']:.3f})"))
                    figH.add_trace(go.Scatter(x=xs, y=gaussian(xs, y0, m1, s1, A1)-y0, mode="lines", name="Component 1", line=dict(dash="dash")))
                    figH.add_trace(go.Scatter(x=xs, y=gaussian(xs, y0, m2, s2, A2)-y0, mode="lines", name="Component 2", line=dict(dash="dash")))
        figH.update_layout(xaxis_title="Value", yaxis_title="Density")
        st.plotly_chart(figH, use_container_width=True)

# ---- Overlay ----
with tabs[2]:
    st.subheader("Overlay: Classical vs PIE (S or E) + Best Auto-fit per curve")
    cand = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[1] >= 8]
    if not cand:
        st.info("No table with ≥8 columns found.")
    else:
        iX = st.selectbox("Choose the 8+ column block", cand, key="ov_block_best",
                          format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
        dfX = blocks[iX][0].copy()
        base = ["Occur._S_Classical","S_Classical","Occur._S_PIE","S_PIE",
                "E_Classical","Occur._E_Classical","E_PIE","Occur._E_PIE"]
        extra = [f"Extra_{i}" for i in range(max(0, dfX.shape[1]-8))]
        dfX.columns = base + extra

        mode = st.radio("Overlay variable", ["Stoichiometry S","FRET efficiency E"], index=0, key="ov_mode_best")
        bin_mode = st.radio("Binning", ["Auto (rule)","Manual"], index=0, key="ov_binmode_best")
        if bin_mode == "Auto (rule)":
            rule = st.selectbox("Auto-binning rule", ["Freedman–Diaconis","Scott","Sturges"], index=0, key="ov_rule_best")
        else:
            c1,c2,c3 = st.columns(3)
            with c1: nb_manual = st.number_input("Bins (manual)", 5, 400, 80, 1, key="ov_bins_best")
            with c2: xmin_man = st.number_input("Range min", value=0.0, step=0.01, key="ov_xmin_best")
            with c3: xmax_man = st.number_input("Range max", value=1.0, step=0.01, key="ov_xmax_best")
        style = st.selectbox("Histogram style", ["Bars","Lines (smoothed)"], index=0, key="ov_style_best")
        smooth_bins = st.slider("Line smoothing (σ in bins)", 0.0, 3.0, 1.0, 0.2, key="ov_smooth_best")
        do_fit = st.checkbox("Run best auto-fit on each", value=True, key="ov_autofit_best")

        if mode == "Stoichiometry S":
            x_cl, w_cl = dfX["S_Classical"].to_numpy(), dfX["Occur._S_Classical"].to_numpy()
            x_pie, w_pie = dfX["S_PIE"].to_numpy(), dfX["Occur._S_PIE"].to_numpy()
            xlabel = "Stoichiometry [S]"; legends = ("Classical (S)", "PIE (S)")
        else:
            x_cl, w_cl = dfX["E_Classical"].to_numpy(), dfX["Occur._E_Classical"].to_numpy()
            x_pie, w_pie = dfX["E_PIE"].to_numpy(), dfX["Occur._E_PIE"].to_numpy()
            xlabel = "FRET efficiency [E]"; legends = ("Classical (E)", "PIE (E)")

        # ---- PATCHED: robust cleaning for arrays/series
        def clean_pair(x, w):
            x = np.asarray(pd.to_numeric(x, errors="coerce"))
            w = np.asarray(pd.to_numeric(w, errors="coerce"))
            m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
            return x[m], w[m]

        x_cl, w_cl = clean_pair(x_cl, w_cl)
        x_pie, w_pie = clean_pair(x_pie, w_pie)

        if x_cl.size == 0 or x_pie.size == 0:
            st.warning("No finite data found for the selected columns.")
        else:
            if bin_mode == "Auto (rule)":
                xmin = float(np.nanmin([np.nanmin(x_cl), np.nanmin(x_pie)]))
                xmax = float(np.nanmax([np.nanmax(x_cl), np.nanmax(x_pie)]))
                nb = auto_bins(np.concatenate([x_cl, x_pie]), rule=rule); edges = np.linspace(xmin, xmax, nb+1)
            else:
                x0, x1 = float(xmin_man), float(xmax_man)
                if x1 <= x0: x1 = x0 + 1e-6
                nb = int(nb_manual); edges = np.linspace(x0, x1, nb+1)

            centers = 0.5*(edges[:-1]+edges[1:])
            hist_cl, _ = np.histogram(x_cl, bins=edges, weights=w_cl, density=True)
            hist_pie, _ = np.histogram(x_pie, bins=edges, weights=w_pie, density=True)

            fig = go.Figure()
            def add_curve(name, hist):
                if style == "Bars":
                    fig.add_bar(x=centers, y=hist, width=np.diff(edges), name=name, opacity=0.5)
                else:
                    y = gaussian_filter1d(hist, sigma=smooth_bins) if smooth_bins>0 else hist
                    fig.add_trace(go.Scatter(x=centers, y=y, mode="lines", name=name))
            add_curve(legends[0], hist_cl)
            add_curve(legends[1], hist_pie)

            if do_fit:
                xs = np.linspace(edges[0], edges[-1], 1000)
                fit_cl = best_autofit(centers, hist_cl, edges[0], edges[-1])
                fit_pie = best_autofit(centers, hist_pie, edges[0], edges[-1])
                if fit_cl:
                    if fit_cl["model"]=="1G":
                        fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, *fit_cl["params"]), mode="lines", name=f"{legends[0]} fit 1G (R²={fit_cl['R2']:.3f})"))
                    else:
                        y0,m1,s1,A1,m2,s2,A2 = fit_cl["params"]
                        fig.add_trace(go.Scatter(x=xs, y=gaussian2(xs, *fit_cl["params"]), mode="lines", name=f"{legends[0]} fit 2G (R²={fit_cl['R2']:.3f})"))
                        fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, y0, m1, s1, A1)-y0, mode="lines", name=f"{legends[0]} comp1", line=dict(dash="dash")))
                        fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, y0, m2, s2, A2)-y0, mode="lines", name=f"{legends[0]} comp2", line=dict(dash="dash")))
                if fit_pie:
                    if fit_pie["model"]=="1G":
                        fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, *fit_pie["params"]), mode="lines", name=f"{legends[1]} fit 1G (R²={fit_pie['R2']:.3f})"))
                    else:
                        y0,m1,s1,A1,m2,s2,A2 = fit_pie["params"]
                        fig.add_trace(go.Scatter(x=xs, y=gaussian2(xs, *fit_pie["params"]), mode="lines", name=f"{legends[1]} fit 2G (R²={fit_pie['R2']:.3f})"))
                        fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, y0, m1, s1, A1)-y0, mode="lines", name=f"{legends[1]} comp1", line=dict(dash="dash")))
                        fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, y0, m2, s2, A2)-y0, mode="lines", name=f"{legends[1]} comp2", line=dict(dash="dash")))
            fig.update_layout(xaxis_title=xlabel, yaxis_title="Density")
            st.plotly_chart(fig, use_container_width=True)

# ---- Joint ----
with tabs[3]:
    st.subheader("Joint view: S–E heatmap + S/E histograms (from 8-col table)")
    colorscale = st.selectbox("Heatmap colorscale", colorscales, index=0, key="joint_cmap_best")
    mats = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[0] >= 10 and df.shape[1] >= 10]
    t8   = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[1] >= 8]
    if not mats or not t8:
        st.info("Need a matrix block and an 8-column block in the file.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            iM = st.selectbox("Matrix block (S×E heatmap)", mats, key="joint_hm_block_best",
                              format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
            smooth = st.slider("Heatmap smoothing σ", 0.0, 6.0, 1.0, 0.1, key="joint_hm_smooth_best")
        with col2:
            iT = st.selectbox("8-col table block (S/E)", t8, key="joint_t8_block_best",
                              format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
            which = st.selectbox("Histogram source", ["PIE", "Classical", "Both"], key="joint_hist_source_best")
            match_bins = st.checkbox("Match histogram bins to heatmap grid", value=True, key="joint_match_bins_best")
            style = st.selectbox("Histogram style", ["Bars","Lines (smoothed)"], key="joint_style_best")
            smooth_bins = st.slider("Line smoothing (σ in bins)", 0.0, 3.0, 1.0, 0.2, key="joint_smooth_lines_best")
        M = blocks[iM][0].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()
        Mplot = gaussian_filter1d(gaussian_filter1d(M, sigma=smooth, axis=0), sigma=smooth, axis=1) if smooth>0 else M.copy()
        ny, nx = Mplot.shape
        e_edges_hm = np.linspace(0, 1, nx+1); e_centers_hm = 0.5*(e_edges_hm[:-1]+e_edges_hm[1:])
        s_edges_hm = np.linspace(0, 1, ny+1); s_centers_hm = 0.5*(s_edges_hm[:-1]+s_edges_hm[1:])
        tbl = blocks[iT][0].copy()
        base = ["Occur._S_Classical","S_Classical","Occur._S_PIE","S_PIE",
                "E_Classical","Occur._E_Classical","E_PIE","Occur._E_PIE"]
        extra = [f"Extra_{i}" for i in range(max(0, tbl.shape[1]-8))]
        tbl.columns = base + extra
        def col(name): return pd.to_numeric(tbl[name], errors="coerce").to_numpy()
        S_cl, W_S_cl, S_pie, W_S_pie = col("S_Classical"), col("Occur._S_Classical"), col("S_PIE"), col("Occur._S_PIE")
        E_cl, W_E_cl, E_pie, W_E_pie = col("E_Classical"), col("Occur._E_Classical"), col("E_PIE"), col("Occur._E_PIE")
        def hist1(x, w, edges):
            m = np.isfinite(x) & np.isfinite(w) & (w>=0)
            h, _ = np.histogram(x[m], bins=edges, weights=w[m]); return h
        if match_bins:
            e_edges, s_edges = e_edges_hm, s_edges_hm
            e_x, s_y = e_centers_hm, s_centers_hm
        else:
            bin_mode = st.radio("Histogram binning (for S/E)", ["Auto (rule)","Manual"], index=0, key="joint_binmode_best")
            if bin_mode == "Auto (rule)":
                nb_e = auto_bins(np.concatenate([E_cl, E_pie]))
                nb_s = auto_bins(np.concatenate([S_cl, S_pie]))
                e_edges = np.linspace(0,1,nb_e+1); s_edges = np.linspace(0,1,nb_s+1)
            else:
                c1,c2,c3 = st.columns(3)
                with c1: nb_e = st.number_input("E bins (manual)", 5, 400, 80, 1, key="joint_bins_e_best")
                with c2: e_min = st.number_input("E min", value=0.0, step=0.01, key="joint_e_min_best")
                with c3: e_max = st.number_input("E max", value=1.0, step=0.01, key="joint_e_max_best")
                if e_max <= e_min: e_max = e_min + 1e-6
                c4,c5,c6 = st.columns(3)
                with c4: nb_s = st.number_input("S bins (manual)", 5, 400, 80, 1, key="joint_bins_s_best")
                with c5: s_min = st.number_input("S min", value=0.0, step=0.01, key="joint_s_min_best")
                with c6: s_max = st.number_input("S max", value=1.0, step=0.01, key="joint_s_max_best")
                if s_max <= s_min: s_max = s_min + 1e-6
                e_edges = np.linspace(e_min, e_max, int(nb_e)+1); s_edges = np.linspace(s_min, s_max, int(nb_s)+1)
            e_x = 0.5*(e_edges[:-1]+e_edges[1:]); s_y = 0.5*(s_edges[:-1]+s_edges[1:])
        e_hist_cl = hist1(E_cl, W_E_cl, e_edges); e_hist_pie = hist1(E_pie, W_E_pie, e_edges)
        s_hist_cl = hist1(S_cl, W_S_cl, s_edges); s_hist_pie = hist1(S_pie, W_S_pie, s_edges)
        figj = make_subplots(
            rows=2, cols=2,
            specs=[[{"type":"xy"}, {"type":"xy"}],
                   [{"type":"heatmap"}, {"type":"xy"}]],
            column_widths=[0.8, 0.2], row_heights=[0.25, 0.75],
            horizontal_spacing=0.02, vertical_spacing=0.02
        )
        def add_hist_top(xc, yc, name):
            if style == "Bars":
                figj.add_trace(go.Bar(x=xc, y=yc, name=name, opacity=0.6), row=1, col=1)
            else:
                ysm = gaussian_filter1d(yc, sigma=smooth_bins) if smooth_bins>0 else yc
                figj.add_trace(go.Scatter(x=xc, y=ysm, mode="lines", name=name), row=1, col=1)
        if which in ("Classical","Both"): add_hist_top(e_x, e_hist_cl, "Classical E")
        if which in ("PIE","Both"):       add_hist_top(e_x, e_hist_pie, "PIE E")
        figj.add_trace(go.Heatmap(z=Mplot, coloraxis="coloraxis", showscale=True), row=2, col=1)
        def add_hist_right(yc, xc, name):
            if style == "Bars":
                figj.add_trace(go.Bar(y=yc, x=xc, orientation="h", name=name, opacity=0.6), row=2, col=2)
            else:
                xsm = gaussian_filter1d(xc, sigma=smooth_bins) if smooth_bins>0 else xc
                figj.add_trace(go.Scatter(y=yc, x=xsm, mode="lines", name=name), row=2, col=2)
        if which in ("Classical","Both"): add_hist_right(s_y, s_hist_cl, "Classical S")
        if which in ("PIE","Both"):       add_hist_right(s_y, s_hist_pie, "PIE S")
        if match_bins:
            figj.update_xaxes(matches="x", row=1, col=1); figj.update_yaxes(matches="y", row=2, col=2)
        figj.update_xaxes(title_text="E (0–1)", row=2, col=1); figj.update_yaxes(title_text="S (0–1)", row=2, col=1)
        figj.update_yaxes(title_text="S (0–1)", row=2, col=2); figj.update_xaxes(title_text="Counts (E)", row=1, col=1)
        figj.update_layout(coloraxis=dict(colorscale=colorscale), showlegend=True, bargap=0, margin=dict(l=40,r=10,t=40,b=40))
        st.plotly_chart(figj, use_container_width=True)

# ---- FRET Analysis (E only, PIE or Classical) ----
with tabs[4]:
    st.subheader("FRET – Histogram + Best Auto-fit (PIE or Classical, E only)")
    candidates = [i for i,(df,_,_,_) in enumerate(blocks) if blocks[i][0].shape[1] >= 8]
    if not candidates:
        st.info("No 8+ column table found in this file.")
    else:
        iP = st.selectbox("Pick the 8-column block", candidates, key="fret_block_best",
                          format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
        df = blocks[iP][0].copy()
        base = ["Occur_S_Classical","S_Classical","Occur_S_PIE","S_PIE",
                "E_Classical","Occur_E_Classical","E_PIE","Occur_E_PIE"]
        extra = [f"Extra_{i}" for i in range(max(0, df.shape[1]-8))]
        df.columns = base + extra

        source = st.radio("E source", ["PIE", "Classical"], index=0, horizontal=True, key="fret_source_best")
        bin_mode = st.radio("Binning", ["Auto (rule)","Manual"], index=0, horizontal=True, key="fret_binmode_best")
        if bin_mode == "Auto (rule)":
            rule = st.selectbox("Auto-binning rule", ["Freedman–Diaconis","Scott","Sturges"], index=0, key="fret_rule_best")
            edges = None
        else:
            c1,c2,c3 = st.columns(3)
            with c1: nb = st.number_input("Bins (manual)", 5, 400, 80, 1, key="fret_bins_best")
            with c2: xmin = st.number_input("E min", value=0.0, step=0.01, key="fret_xmin_best")
            with c3: xmax = st.number_input("E max", value=1.0, step=0.01, key="fret_xmax_best")
            if xmax <= xmin: xmax = xmin + 1e-6
            edges = np.linspace(float(xmin), float(xmax), int(nb)+1)
        style = st.selectbox("Histogram style", ["Bars","Lines (smoothed)"], index=0, key="fret_style_best")
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

        m = np.isfinite(E) & np.isfinite(W) & (W>=0)
        E, W = E[m], W[m]
        if E.size == 0:
            st.warning("No finite E values in this block.")
        else:
            if edges is None:
                nb = auto_bins(E, rule=rule); xmin, xmax = float(np.nanmin(E)), float(np.nanmax(E))
                edges = np.linspace(xmin, xmax, nb+1)
            centers = 0.5*(edges[:-1]+edges[1:])
            hist, _ = np.histogram(E, bins=edges, weights=W, density=True)

            fig = go.Figure()
            if style == "Bars":
                fig.add_bar(x=centers, y=hist, width=np.diff(edges), name=f"{label} histogram", opacity=0.55)
            else:
                y = gaussian_filter1d(hist, sigma=smooth_bins) if smooth_bins>0 else hist
                fig.add_trace(go.Scatter(x=centers, y=y, mode="lines", name=label))

            if do_fit:
                fit = best_autofit(centers, hist, edges[0], edges[-1])
                if fit:
                    xs = np.linspace(edges[0], edges[-1], 1000)
                    if fit["model"] == "1G":
                        fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, *fit["params"]), mode="lines", name=f"Best fit 1G (R²={fit['R2']:.3f})"))
                    else:
                        y0,m1,s1,A1,m2,s2,A2 = fit["params"]
                        fig.add_trace(go.Scatter(x=xs, y=gaussian2(xs, *fit["params"]), mode="lines", name=f"Best fit 2G (R²={fit['R2']:.3f})"))
                        fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, y0, m1, s1, A1)-y0, mode="lines", name="Component 1", line=dict(dash="dash")))
                        fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, y0, m2, s2, A2)-y0, mode="lines", name="Component 2", line=dict(dash="dash")))
            fig.update_layout(xaxis_title="FRET efficiency, E", yaxis_title="Density")
            st.plotly_chart(fig, use_container_width=True)

# ---- Compare multiple files ----
with tabs[5]:
    st.subheader("Compare multiple files – overlay E histograms in a selected range")
    files = st.file_uploader("Upload one or more .dat files", type=["dat","txt","csv"], accept_multiple_files=True, key="multi_uploader_best")
    if files:
        source = st.radio("E source to compare", ["PIE", "Classical"], index=0, horizontal=True, key="multi_source_best")
        normalize = st.checkbox("Normalize each histogram area to 1", value=True, key="multi_norm_best")
        bin_mode = st.radio("Binning", ["Auto (rule)","Manual"], index=0, horizontal=True, key="multi_binmode_best")
        if bin_mode == "Auto (rule)":
            rule = st.selectbox("Auto-binning rule (shared)", ["Freedman–Diaconis","Scott","Sturges"], index=0, key="multi_rule_best")
            edges = None
        else:
            c1,c2,c3 = st.columns(3)
            with c1: nb = st.number_input("Bins (manual, shared)", 5, 400, 80, 1, key="multi_bins_best")
            with c2: xmin = st.number_input("E min (shared)", value=0.0, step=0.01, key="multi_xmin_best")
            with c3: xmax = st.number_input("E max (shared)", value=1.0, step=0.01, key="multi_xmax_best")
            if xmax <= xmin: xmax = xmin + 1e-6
            edges = np.linspace(float(xmin), float(xmax), int(nb)+1)

        style = st.selectbox("Histogram style", ["Lines (smoothed)","Bars"], index=0, key="multi_style_best")
        smooth_bins = st.slider("Line smoothing (σ in bins)", 0.0, 3.0, 1.0, 0.2, key="multi_smooth_best")
        region = st.slider("Analysis region (E-range)", 0.0, 1.0, (0.7, 1.0), 0.01, key="multi_region_best")

        all_E, all_W, names = [], [], []
        for f in files:
            raw2 = f.getvalue().decode("utf-8", errors="ignore")
            blks = split_numeric_blocks_with_headers(raw2)
            cand = [i for i,(df,_,_,_) in enumerate(blks) if blks[i][0].shape[1] >= 8]
            if not cand:
                continue
            df = blks[cand[0]][0].copy()
            df.columns = ["Occur_S_Classical","S_Classical","Occur_S_PIE","S_PIE","E_Classical","Occur_E_Classical","E_PIE","Occur_E_PIE"] + [f"Extra_{i}" for i in range(max(0, df.shape[1]-8))]
            if source == "PIE":
                E = pd.to_numeric(df["E_PIE"], errors="coerce").to_numpy()
                W = pd.to_numeric(df["Occur_E_PIE"], errors="coerce").to_numpy()
            else:
                E = pd.to_numeric(df["E_Classical"], errors="coerce").to_numpy()
                W = pd.to_numeric(df["Occur_E_Classical"], errors="coerce").to_numpy()
            m = np.isfinite(E) & np.isfinite(W) & (W>=0)
            if np.any(m):
                all_E.append(E[m]); all_W.append(W[m]); names.append(f.name)

        if len(all_EOT
