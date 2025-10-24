
import re, io, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="FRET Analyzer – FULL (with compare)", layout="wide")
st.title("FRET Analyzer – FULL")

# ---------- parsing ----------
def split_numeric_blocks_with_headers(text: str):
    text = text.replace(",", ".")
    lines = text.splitlines()
    blocks = []
    cur = []; header_buf = []; start_idx = None
    isnum = re.compile(r'^\s*[\d\.eE\-\+]+([\s\t,;][\d\.eE\-\+]+)*\s*$').match
    def flush():
        nonlocal cur, header_buf, start_idx
        if not cur: return
        s = "\n".join(cur).strip()
        try:
            df = pd.read_csv(io.StringIO(s), sep=r"[\s,;]+", engine="python", header=None)
            blocks.append((df, list(header_buf), start_idx, None))
        except Exception:
            pass
        cur = []; start_idx = None
    for i, ln in enumerate(lines):
        if isnum(ln):
            if start_idx is None: start_idx = i
            cur.append(ln)
        else:
            header_buf.append(ln.strip()); header_buf = header_buf[-3:]
            if start_idx is not None: flush()
    if start_idx is not None: flush()
    return blocks

# ---------- helpers ----------
def gaussian(x, y0, mu, sigma, A):
    amp = A / (sigma * np.sqrt(2*np.pi))
    return y0 + amp * np.exp(-0.5 * ((x - mu)/sigma)**2)

def r2_score(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.nansum((y - yhat)**2); ss_tot = np.nansum((y - np.nanmean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot>0 else np.nan

def auto_bins(x, rule="Freedman–Diaconis", nb_fallback=80):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2: return nb_fallback
    if rule == "Freedman–Diaconis":
        iqr = np.subtract(*np.percentile(x, [75,25]))
        if iqr == 0: return max(10, min(200, int(np.sqrt(n))))
        bw = 2 * iqr * (n ** (-1/3))
    elif rule == "Scott":
        std = np.nanstd(x, ddof=1)
        if std == 0: return max(10, min(200, int(np.sqrt(n))))
        bw = 3.5 * std * (n ** (-1/3))
    else:
        return max(10, min(200, int(np.ceil(np.log2(n)+1))))
    rng = np.nanmax(x) - np.nanmin(x)
    if rng <= 0: return max(10, min(200, int(np.sqrt(n))))
    return int(np.clip(np.ceil(rng/bw), 10, 200))

def smart_fit(centers, hist, xmin, xmax):
    z = np.asarray(hist, float); x = np.asarray(centers, float)
    m = np.isfinite(z) & np.isfinite(x)
    x, z = x[m], z[m]
    if x.size < 8 or np.count_nonzero(z) < 5: return None
    zs = gaussian_filter1d(z, sigma=max(1, int(len(z)*0.02)))
    mu0 = float(x[int(np.argmax(zs))])
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(zs >= 0.08*zs.max(), iterations=2)
    xw, zw = x[mask], z[mask]
    if xw.size < 8: xw, zw = x, z
    q25, q75 = np.quantile(xw, [0.25, 0.75])
    sigma0 = max(1e-3, (q75-q25)/1.349)
    y00 = max(1e-6, float(np.median(z[(z <= np.percentile(z, 20))])))
    A0 = float(np.trapz(zw - np.minimum(zw, y00), xw))
    binw = np.median(np.diff(x))
    lower = [0.0, xmin, max(binw*0.7, 1e-4), 0.0]
    upper = [max(z.max()*2, 1.0), xmax, (xmax-xmin)/1.5, np.inf]
    for mul in (1.0, 0.7, 1.3, 0.5, 1.8):
        p0 = [y00, mu0, sigma0*mul, max(A0, 1e-4)]
        try:
            popt, pcov = curve_fit(gaussian, xw, zw, p0=p0, bounds=(lower, upper), maxfev=80000)
            perr = np.sqrt(np.diag(pcov)); R2 = r2_score(z, gaussian(x, *popt))
            return popt, perr, R2
        except Exception: pass
    return None

def clean_pair(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    return x[m], w[m]

def fwhm_from_sigma(s):
    return 2*np.sqrt(2*np.log(2))*s

# ---------- single-file uploader for first 5 tabs ----------
uploaded = st.file_uploader("Upload your .dat file (for tabs 1–5)", type=["dat","txt","csv"], key="uploader_full_merge_v2")
if uploaded is None:
    st.info("Upload your file to continue."); st.stop()

raw = uploaded.getvalue().decode("utf-8", errors="ignore")
blocks = split_numeric_blocks_with_headers(raw)

colorscales = ["Viridis","Plasma","Magma","Inferno","Cividis","Turbo","IceFire","YlGnBu","Greys"]

tabs = st.tabs(["Heatmap", "Histogram (single)", "Overlay", "Joint (Heatmap + Marginals)", "FRET Analysis (choose E: PIE/Classical)", "Compare multiple files (E)"])

# ---- Heatmap ----
with tabs[0]:
    st.subheader("Correlogram Heatmap")
    mats = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[0] >= 10 and df.shape[1] >= 10]
    if not mats: mats = list(range(len(blocks)))
    iM = st.selectbox("Choose matrix block", mats, key="hm_block_merge_v2",
                      format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
    dfm = blocks[iM][0].astype(float).replace([np.inf, -np.inf], np.nan)
    c1,c2,c3 = st.columns(3)
    with c1: zmin = st.number_input("zmin (0=auto)", value=0.0, key="hm_zmin_merge_v2")
    with c2: zmax = st.number_input("zmax (0=auto)", value=0.0, key="hm_zmax_merge_v2")
    with c3: cmap = st.selectbox("Colorscale", colorscales, index=0, key="hm_cmap_merge_v2")
    smooth = st.slider("Gaussian smoothing (σ)", 0.0, 6.0, 1.0, 0.1, key="hm_smooth_merge_v2")
    arr = dfm.to_numpy()
    arrp = gaussian_filter1d(gaussian_filter1d(arr, sigma=smooth, axis=0), sigma=smooth, axis=1) if smooth>0 else arr
    fig = px.imshow(arrp, origin="lower", aspect="auto", color_continuous_scale=cmap,
                    zmin=None if zmin<=0 else zmin, zmax=None if zmax<=0 else zmax,
                    labels=dict(x="E bins (columns)", y="S bins (rows)", color="Counts"))
    st.plotly_chart(fig, use_container_width=True)

# ---- Histogram (single) ----
with tabs[1]:
    st.subheader("Histogram + Gaussian fit (single dataset)")
    tbls = list(range(len(blocks)))
    iT = st.selectbox("Choose block", tbls, index=tbls[-1] if tbls else 0, key="single_block_merge_v2",
                      format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
    dft = blocks[iT][0].copy()
    templ = st.radio("Column template", ["Generic (C0..)", "FRET 8-cols (named)"], index=1, key="single_template_merge_v2")
    if templ == "FRET 8-cols (named)" and dft.shape[1] >= 8:
        base = ["Occur._S_Classical","S_Classical","Occur._S_PIE","S_PIE",
                "E_Classical","Occur._E_Classical","E_PIE","Occur._E_PIE"]
        extra = [f"Extra_{i}" for i in range(dft.shape[1]-8)]
        dft.columns = base + extra
    else:
        dft.columns = [f"C{j}" for j in range(dft.shape[1])]
    c1,c2,c3,c4 = st.columns(4)
    with c1: x_col = st.selectbox("Values column (x)", dft.columns, index=0, key="single_xcol_merge_v2")
    with c2: w_col = st.selectbox("Weights (optional)", ["(none)"]+list(dft.columns), index=0, key="single_wcol_merge_v2")
    with c3: rule  = st.selectbox("Auto-binning rule", ["Freedman–Diaconis","Scott","Sturges"], index=0, key="single_rule_merge_v2")
    with c4: auto_fit = st.checkbox("Auto-fit Gaussian", value=True, key="single_autofit_merge_v2")
    x = pd.to_numeric(dft[x_col], errors="coerce").to_numpy()
    w = None
    if w_col != "(none)":
        w = pd.to_numeric(dft[w_col], errors="coerce").to_numpy()
        w = np.where(np.isfinite(w) & (w>0), w, 0.0)
    m = np.isfinite(x); xmin = float(np.nanmin(x[m])); xmax = float(np.nanmax(x[m]))
    nb = auto_bins(x[m], rule=rule); edges = np.linspace(xmin, xmax, nb+1); centers = 0.5*(edges[:-1]+edges[1:])
    hist, _ = np.histogram(x[m], bins=edges, weights=w if w is not None else None, density=True)
    figH = go.Figure(); figH.add_bar(x=centers, y=hist, width=np.diff(edges), name="Histogram", opacity=0.55)
    if auto_fit:
        fit = smart_fit(centers, hist, xmin, xmax)
        if fit:
            p, perr, R2 = fit; xs = np.linspace(xmin, xmax, 800)
            figH.add_trace(go.Scatter(x=xs, y=gaussian(xs, *p), mode="lines", name=f"Gaussian fit (R²={R2:.3f})"))
    figH.update_layout(xaxis_title="Value", yaxis_title="Density"); st.plotly_chart(figH, use_container_width=True)

# ---- Overlay ----
with tabs[2]:
    st.subheader("Overlay: Classical vs PIE (E and S)")
    cand = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[1] >= 8]
    if not cand: st.info("No table with ≥8 columns found.")
    else:
        iX = st.selectbox("Choose the 8+ column block", cand, key="ov_block_merge_v2",
                          format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
        dfX = blocks[iX][0].copy()
        base = ["Occur._S_Classical","S_Classical","Occur._S_PIE","S_PIE",
                "E_Classical","Occur._E_Classical","E_PIE","Occur._E_PIE"]
        extra = [f"Extra_{i}" for i in range(max(0, dfX.shape[1]-8))]
        dfX.columns = base + extra
        mode = st.radio("Overlay variable", ["Stoichiometry S","FRET efficiency E"], index=0, key="ov_mode_merge_v2")
        rule = st.selectbox("Auto-binning rule", ["Freedman–Diaconis","Scott","Sturges"], index=0, key="ov_rule_merge_v2")
        auto_fit = st.checkbox("Auto-fit Gaussian", value=True, key="ov_autofit_merge_v2")
        style = st.selectbox("Histogram style", ["Bars","Lines (smoothed)"], index=0, key="ov_style_merge_v2")
        smooth_bins = st.slider("Line smoothing (σ in bins)", 0.0, 3.0, 1.0, 0.2, key="ov_smooth_merge_v2")
        if mode == "Stoichiometry S":
            x_cl, w_cl = dfX["S_Classical"].to_numpy(), dfX["Occur._S_Classical"].to_numpy()
            x_pie, w_pie = dfX["S_PIE"].to_numpy(), dfX["Occur._S_PIE"].to_numpy()
            xlabel = "Stoichiometry [S]"; legends = ("Classical (S)", "PIE (S)")
        else:
            x_cl, w_cl = dfX["E_Classical"].to_numpy(), dfX["Occur._E_Classical"].to_numpy()
            x_pie, w_pie = dfX["E_PIE"].to_numpy(), dfX["Occur._E_PIE"].to_numpy()
            xlabel = "FRET efficiency [E]"; legends = ("Classical (E)", "PIE (E)")
        x_cl, w_cl = clean_pair(x_cl, w_cl); x_pie, w_pie = clean_pair(x_pie, w_pie)
        xmin = float(np.nanmin([np.nanmin(x_cl), np.nanmin(x_pie)]))
        xmax = float(np.nanmax([np.nanmax(x_cl), np.nanmax(x_pie)]))
        nb = auto_bins(np.concatenate([x_cl, x_pie]), rule=rule); edges = np.linspace(xmin, xmax, nb+1); centers = 0.5*(edges[:-1]+edges[1:])
        hist_cl, _ = np.histogram(x_cl, bins=edges, weights=w_cl, density=True)
        hist_pie, _ = np.histogram(x_pie, bins=edges, weights=w_pie, density=True)
        fig = go.Figure()
        if style == "Bars":
            fig.add_bar(x=centers, y=hist_cl, width=np.diff(edges), name=legends[0], opacity=0.5)
            fig.add_bar(x=centers, y=hist_pie, width=np.diff(edges), name=legends[1], opacity=0.5)
        else:
            y1 = gaussian_filter1d(hist_cl, sigma=smooth_bins) if smooth_bins>0 else hist_cl
            y2 = gaussian_filter1d(hist_pie, sigma=smooth_bins) if smooth_bins>0 else hist_pie
            fig.add_trace(go.Scatter(x=centers, y=y1, mode="lines", name=legends[0]))
            fig.add_trace(go.Scatter(x=centers, y=y2, mode="lines", name=legends[1]))
        if auto_fit:
            fit_cl = smart_fit(centers, hist_cl, xmin, xmax); fit_pie = smart_fit(centers, hist_pie, xmin, xmax)
            xs = np.linspace(xmin, xmax, 1000)
            if fit_cl: p,_,R2 = fit_cl; fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, *p), mode="lines", name=f"{legends[0]} fit (R²={R2:.3f})"))
            if fit_pie: p,_,R2 = fit_pie; fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, *p), mode="lines", name=f"{legends[1]} fit (R²={R2:.3f})"))
        fig.update_layout(xaxis_title=xlabel, yaxis_title="H [Occur.·10^3 Events] (density)"); st.plotly_chart(fig, use_container_width=True)

# ---- Joint (Heatmap + S/E histograms) ----
with tabs[3]:
    st.subheader("Joint view: S–E heatmap + S/E histograms (from 8‑col table)")
    colorscale = st.selectbox("Heatmap colorscale", colorscales, index=0, key="joint_cmap_merge_v2")
    mats = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[0] >= 10 and df.shape[1] >= 10]
    t8   = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[1] >= 8]
    if not mats or not t8:
        st.info("Need a matrix block and an 8‑column block in the file.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            iM = st.selectbox("Matrix block (S×E heatmap)", mats, key="joint_hm_block_merge_v2",
                              format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
            smooth = st.slider("Heatmap smoothing σ", 0.0, 6.0, 1.0, 0.1, key="joint_hm_smooth_merge_v2")
        with col2:
            iT = st.selectbox("8‑col table block (S/E)", t8, key="joint_t8_block_merge_v2",
                              format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
            which = st.selectbox("Histogram source", ["PIE", "Classical", "Both"], key="joint_hist_source_merge_v2")
            match_bins = st.checkbox("Match histogram bins to heatmap grid", value=True, key="joint_match_bins_merge_v2")
            style = st.selectbox("Histogram style", ["Bars","Lines (smoothed)"], key="joint_style_merge_v2")
            smooth_bins = st.slider("Line smoothing (σ in bins)", 0.0, 3.0, 1.0, 0.2, key="joint_smooth_lines_merge_v2")
        M = blocks[iM][0].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()
        Mplot = gaussian_filter1d(gaussian_filter1d(M, sigma=smooth, axis=0), sigma=smooth, axis=1) if smooth>0 else M.copy()
        ny, nx = Mplot.shape
        e_edges = np.linspace(0, 1, nx+1); e_centers = 0.5*(e_edges[:-1]+e_edges[1:])
        s_edges = np.linspace(0, 1, ny+1); s_centers = 0.5*(s_edges[:-1]+s_edges[1:])
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
            e_x, s_y = e_centers, s_centers
            e_hist_cl = hist1(E_cl, W_E_cl, e_edges); e_hist_pie = hist1(E_pie, W_E_pie, e_edges)
            s_hist_cl = hist1(S_cl, W_S_cl, s_edges); s_hist_pie = hist1(S_pie, W_S_pie, s_edges)
        else:
            nb_e = auto_bins(np.concatenate([E_cl, E_pie])); e_edges = np.linspace(0,1,nb_e+1); e_x = 0.5*(e_edges[:-1]+e_edges[1:])
            nb_s = auto_bins(np.concatenate([S_cl, S_pie])); s_edges = np.linspace(0,1,nb_s+1); s_y = 0.5*(s_edges[:-1]+s_edges[1:])
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
        figj.update_layout(coloraxis=dict(colorscale=colorscales[0]), showlegend=True, bargap=0, margin=dict(l=40,r=10,t=40,b=40))
        st.plotly_chart(figj, use_container_width=True)

# ---- FRET Analysis (choose E: PIE or Classical) ----
with tabs[4]:
    st.subheader("FRET – Histogram + Gaussian Fit (PIE or Classical, E only)")
    candidates = [i for i,(df,_,_,_) in enumerate(blocks) if blocks[i][0].shape[1] >= 8]
    if not candidates:
        st.info("No 8+ column table found in this file.")
    else:
        iP = st.selectbox("Pick the 8-column block", candidates, key="fret_block_merge_v2",
                          format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
        df = blocks[iP][0].copy()
        base = ["Occur_S_Classical","S_Classical","Occur_S_PIE","S_PIE","E_Classical","Occur_E_Classical","E_PIE","Occur_E_PIE"]
        extra = [f"Extra_{i}" for i in range(max(0, df.shape[1]-8))]
        df.columns = base + extra

        source = st.radio("E source", ["PIE", "Classical"], index=0, horizontal=True, key="fret_source_merge_v2")
        rule = st.selectbox("Auto-binning rule", ["Freedman–Diaconis","Scott","Sturges"], index=0, key="fret_rule_merge_v2")
        style = st.selectbox("Histogram style", ["Bars","Lines (smoothed)"], index=0, key="fret_style_merge_v2")
        smooth_bins = st.slider("Line smoothing (σ in bins)", 0.0, 3.0, 1.0, 0.2, key="fret_smooth_merge_v2")
        do_fit = st.checkbox("Fit Gaussian", value=True, key="fret_fit_merge_v2")

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
        nb = auto_bins(E, rule=rule); xmin, xmax = float(np.nanmin(E)), float(np.nanmax(E))
        edges = np.linspace(xmin, xmax, nb+1); centers = 0.5*(edges[:-1]+edges[1:])
        hist, _ = np.histogram(E, bins=edges, weights=W, density=True)

        fig = go.Figure()
        if style == "Bars":
            fig.add_bar(x=centers, y=hist, width=np.diff(edges), name=f"{label} histogram", opacity=0.55)
        else:
            y = gaussian_filter1d(hist, sigma=smooth_bins) if smooth_bins>0 else hist
            fig.add_trace(go.Scatter(x=centers, y=y, mode="lines", name=label))

        if do_fit:
            fit = smart_fit(centers, hist, xmin, xmax)
            if fit:
                p, perr, R2 = fit
                xs = np.linspace(xmin, xmax, 1000)
                fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, *p), mode="lines", name=f"Gaussian fit (R²={R2:.3f})"))
                y0, mu, sigma, A = p
                dy0, dmu, dsigma, dA = perr
                fwhm = fwhm_from_sigma(sigma)
                dfres = pd.DataFrame({
                    "parameter": ["baseline y0","mean μ","sigma σ","FWHM","area A","R²"],
                    "value":     [y0, mu, sigma, fwhm, A, R2],
                    "uncertainty":[dy0, dmu, dsigma, fwhm*(dsigma/sigma) if sigma>0 else np.nan, dA, np.nan]
                })
                st.subheader("Fit results")
                st.dataframe(dfres, use_container_width=True)
                st.download_button("Download results (CSV)", dfres.to_csv(index=False).encode(), f"{source.lower()}_fret_fit_results.csv", "text/csv")
            else:
                st.warning("Fit could not converge – try a different binning rule or check data quality.")
        fig.update_layout(xaxis_title="FRET efficiency, E", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

# ---- Compare multiple files ----
with tabs[5]:
    st.subheader("Compare multiple files – overlay E histograms in a selected range")
    files = st.file_uploader("Upload one or more .dat files", type=["dat","txt","csv"], accept_multiple_files=True, key="multi_uploader_v2")
    if files:
        source = st.radio("E source to compare", ["PIE", "Classical"], index=0, horizontal=True, key="multi_source_v2")
        normalize = st.checkbox("Normalize each histogram area to 1", value=True, key="multi_norm_v2")
        rule = st.selectbox("Auto-binning rule (shared)", ["Freedman–Diaconis","Scott","Sturges"], index=0, key="multi_rule_v2")
        style = st.selectbox("Histogram style", ["Lines (smoothed)","Bars"], index=0, key="multi_style_v2")
        smooth_bins = st.slider("Line smoothing (σ in bins)", 0.0, 3.0, 1.0, 0.2, key="multi_smooth_v2")
        region = st.slider("Analysis region (E-range)", 0.0, 1.0, (0.7, 1.0), 0.01, key="multi_region_v2")

        # parse all, extract E and W, determine common bin edges
        all_E = []; all_W = []; names = []
        for f in files:
            raw2 = f.getvalue().decode("utf-8", errors="ignore")
            blks = split_numeric_blocks_with_headers(raw2)
            cand = [i for i,(df,_,_,_) in enumerate(blks) if blks[i][0].shape[1] >= 8]
            if not cand: continue
            df = blks[cand[0]][0].copy()
            df.columns = ["Occur_S_Classical","S_Classical","Occur_S_PIE","S_PIE","E_Classical","Occur_E_Classical","E_PIE","Occur_E_PIE"] + [f"Extra_{i}" for i in range(max(0, df.shape[1]-8))]
            if source == "PIE":
                E = pd.to_numeric(df["E_PIE"], errors="coerce").to_numpy()
                W = pd.to_numeric(df["Occur_E_PIE"], errors="coerce").to_numpy()
            else:
                E = pd.to_numeric(df["E_Classical"], errors="coerce").to_numpy()
                W = pd.to_numeric(df["Occur_E_Classical"], errors="coerce").to_numpy()
            m = np.isfinite(E) & np.isfinite(W) & (W>=0)
            all_E.append(E[m]); all_W.append(W[m]); names.append(f.name)
        if len(all_E) >= 1:
            # common bins over union of E
            xmin = float(np.nanmin([np.nanmin(x) for x in all_E]))
            xmax = float(np.nanmax([np.nanmax(x) for x in all_E]))
            nb = auto_bins(np.concatenate(all_E), rule=rule)
            edges = np.linspace(xmin, xmax, nb+1)
            centers = 0.5*(edges[:-1] + edges[1:])

            fig = go.Figure()
            summary = []
            for E, W, nm in zip(all_E, all_W, names):
                hist, _ = np.histogram(E, bins=edges, weights=W, density=False)
                if normalize:
                    tot = hist.sum()
                    y = hist / tot if tot>0 else hist.astype(float)
                else:
                    y = hist.astype(float)
                if style.startswith("Lines"):
                    yy = gaussian_filter1d(y, sigma=smooth_bins) if smooth_bins>0 else y
                    fig.add_trace(go.Scatter(x=centers, y=yy, mode="lines", name=nm))
                else:
                    fig.add_bar(x=centers, y=y, width=np.diff(edges), name=nm, opacity=0.5)

                # fraction in region
                rmin, rmax = region
                mask_bins = (centers >= rmin) & (centers < rmax)
                frac = y[mask_bins].sum() / y.sum() if y.sum()>0 else np.nan
                summary.append({"file": nm, "fraction_in_region": frac})

            fig.update_layout(xaxis_title="FRET efficiency, E", yaxis_title=("Normalized counts" if normalize else "Counts"))
            st.plotly_chart(fig, use_container_width=True)

            df_sum = pd.DataFrame(summary)
            st.subheader("Population fraction in selected region")
            st.dataframe(df_sum, use_container_width=True)
            st.download_button("Download fractions (CSV)", df_sum.to_csv(index=False).encode(), "region_fractions.csv", "text/csv")
    else:
        st.info("Upload multiple files to compare.")

