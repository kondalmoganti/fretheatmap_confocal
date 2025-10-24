
import re, io, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="FRET Analyzer – FULL", layout="wide")
st.title("FRET Analyzer (FULL build)")

# ---------------- Parsing (with header capture) ----------------
def split_numeric_blocks_with_headers(text: str):
    text = text.replace(",", ".")
    lines = text.splitlines()
    blocks = []
    cur = []
    header_buf = []
    start_idx = None

    def is_num(ln):
        return re.match(r'^\s*[\d\.eE\-\+]+([\s\t,;][\d\.eE\-\+]+)*\s*$', ln) is not None

    def flush(end_idx):
        nonlocal cur, header_buf, start_idx
        if not cur: return
        s = "\n".join(cur).strip()
        try:
            df = pd.read_csv(io.StringIO(s), sep=r"[\s,;]+", engine="python", header=None)
            blocks.append((df, list(header_buf), start_idx, end_idx))
        except Exception:
            pass
        cur = []
        start_idx = None

    for i, ln in enumerate(lines):
        if is_num(ln):
            if start_idx is None:
                start_idx = i
            cur.append(ln)
        else:
            header_buf.append(ln.strip())
            header_buf = header_buf[-3:]
            if start_idx is not None:
                flush(i-1)
    if start_idx is not None:
        flush(len(lines)-1)
    return blocks

# ---------------- Models & helpers ----------------
def gaussian(x, y0, mu, sigma, A):
    amp = A / (sigma * np.sqrt(2*np.pi))
    return y0 + amp * np.exp(-0.5 * ((x - mu)/sigma)**2)

def r2_score(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.nansum((y - yhat)**2); ss_tot = np.nansum((y - np.nanmean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot>0 else np.nan

def auto_bins(x, rule="Freedman–Diaconis", nb_fallback=80):
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2: return nb_fallback
    if rule == "Freedman–Diaconis":
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr == 0: return max(10, min(200, int(np.sqrt(n))))
        bw = 2 * iqr * (n ** (-1/3))
    elif rule == "Scott":
        std = np.nanstd(x, ddof=1)
        if std == 0: return max(10, min(200, int(np.sqrt(n))))
        bw = 3.5 * std * (n ** (-1/3))
    else:
        return max(10, min(200, int(np.ceil(np.log2(n) + 1))))
    rng = np.nanmax(x) - np.nanmin(x)
    if rng <= 0: return max(10, min(200, int(np.sqrt(n))))
    nb = int(np.clip(np.ceil(rng / bw), 10, 200))
    return nb

def smart_fit(centers, hist, xmin, xmax):
    z = np.asarray(hist, float); x = np.asarray(centers, float)
    m = np.isfinite(z) & np.isfinite(x)
    x, z = x[m], z[m]
    if x.size < 8 or np.count_nonzero(z) < 5:
        return None
    # smooth for peak detection
    zs = gaussian_filter1d(z, sigma=max(1, int(len(z)*0.02)))
    argmax = int(np.argmax(zs))
    mu0 = float(x[argmax])
    # choose fit window
    alpha = 0.08
    mask = zs >= alpha * zs.max()
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(mask, iterations=2)
    xw, zw = x[mask], z[mask]
    if xw.size < 8:
        xw, zw = x, z
    q25, q75 = np.quantile(xw, [0.25, 0.75])
    sigma0 = max(1e-3, (q75 - q25) / 1.349)
    y00 = max(1e-6, float(np.median(z[(z <= np.percentile(z, 20))])))
    A0 = float(np.trapz(zw - np.minimum(zw, y00), xw))
    binw = np.median(np.diff(x))
    lower = [0.0, xmin, max(binw*0.7, 1e-4), 0.0]
    upper = [max(z.max()*2, 1.0), xmax, (xmax-xmin)/1.5, np.inf]
    for mul in (1.0, 0.7, 1.3, 0.5, 1.8):
        p0 = [y00, mu0, sigma0*mul, max(A0, 1e-4)]
        try:
            popt, pcov = curve_fit(gaussian, xw, zw, p0=p0, bounds=(lower, upper), maxfev=80000)
            perr = np.sqrt(np.diag(pcov))
            R2 = r2_score(z, gaussian(x, *popt))
            return popt, perr, R2
        except Exception:
            continue
    return None

def clean_pair(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    return x[m], w[m]

# ---------------- UI ----------------
uploaded = st.file_uploader("Upload your .dat file", type=["dat","txt","csv"])

if uploaded is None:
    st.info("Upload your file to continue.")
    st.stop()

raw = uploaded.getvalue().decode("utf-8", errors="ignore")
blocks = split_numeric_blocks_with_headers(raw)

tabs = st.tabs(["Heatmap", "Histogram (single)", "Overlay: Classical vs PIE"])

# -------- Heatmap --------
with tabs[0]:
    st.subheader("Correlogram Heatmap")
    mats = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[0] >= 10 and df.shape[1] >= 10]
    if not mats: mats = list(range(len(blocks)))
    sel = st.selectbox("Choose block", mats, format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
    dfm = blocks[sel][0].astype(float).replace([np.inf, -np.inf], np.nan)
    zmin = st.number_input("zmin (0=auto)", value=0.0); zmax = st.number_input("zmax (0=auto)", value=0.0)
    smooth = st.slider("Gaussian smoothing (σ)", 0.0, 6.0, 1.0, 0.1)
    arr = dfm.to_numpy()
    if smooth>0:
        arrp = gaussian_filter1d(gaussian_filter1d(arr, sigma=smooth, axis=0), sigma=smooth, axis=1)
    else:
        arrp = arr
    fig = px.imshow(arrp, origin="lower", aspect="auto", color_continuous_scale="Viridis",
                    zmin=None if zmin<=0 else zmin, zmax=None if zmax<=0 else zmax,
                    labels=dict(x="E bins (columns)", y="S bins (rows)", color="Counts"))
    st.plotly_chart(fig, use_container_width=True)

# -------- Histogram (single) --------
with tabs[1]:
    st.subheader("Histogram + Gaussian fit (single dataset)")
    tbls = list(range(len(blocks)))
    sel = st.selectbox("Choose block", tbls, index=tbls[-1] if tbls else 0,
                       format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
    dft = blocks[sel][0].copy()
    templ = st.radio("Column template", ["Generic (C0..)", "FRET 8-cols (named)"], index=1)
    if templ == "FRET 8-cols (named)" and dft.shape[1] >= 8:
        base_names = ["Occur._S_Classical","S_Classical","Occur._S_PIE","S_PIE",
                      "E_Classical","Occur._E_Classical","E_PIE","Occur._E_PIE"]
        extra = [f"Extra_{i}" for i in range(dft.shape[1]-8)]
        dft.columns = base_names + extra
    else:
        dft.columns = [f"C{j}" for j in range(dft.shape[1])]
    st.dataframe(dft.head(12), use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: x_col = st.selectbox("Values column (x)", dft.columns, index=0)
    with c2: w_col = st.selectbox("Weights (optional)", ["(none)"]+list(dft.columns), index=0)
    with c3: rule  = st.selectbox("Auto-binning rule", ["Freedman–Diaconis","Scott","Sturges"], index=0)
    with c4: auto_fit = st.checkbox("Auto-fit Gaussian", value=True)

    x = pd.to_numeric(dft[x_col], errors="coerce").to_numpy()
    w = None
    if w_col != "(none)":
        w = pd.to_numeric(dft[w_col], errors="coerce").to_numpy()
        w = np.where(np.isfinite(w) & (w>0), w, 0.0)

    m = np.isfinite(x)
    xmin = float(np.nanmin(x[m])); xmax = float(np.nanmax(x[m]))
    nb = auto_bins(x[m], rule=rule)
    edges = np.linspace(xmin, xmax, nb+1)
    centers = 0.5*(edges[:-1]+edges[1:])
    hist, _ = np.histogram(x[m], bins=edges, weights=w if w is not None else None, density=True)

    figH = go.Figure()
    figH.add_bar(x=centers, y=hist, width=np.diff(edges), name="Histogram", opacity=0.55)

    if auto_fit:
        fit = smart_fit(centers, hist, xmin, xmax)
        if fit:
            p, perr, R2 = fit
            xs = np.linspace(xmin, xmax, 800)
            figH.add_trace(go.Scatter(x=xs, y=gaussian(xs, *p), mode="lines", name=f"Gaussian fit (R²={R2:.3f})"))
            text = (
                "<b>Model</b> Gauss<br>"
                f"y₀ {p[0]:.5g} ± {perr[0]:.2g}<br>"
                f"xc {p[1]:.5g} ± {perr[1]:.2g}<br>"
                f"w  {p[2]:.5g} ± {perr[2]:.2g}<br>"
                f"A  {p[3]:.5g} ± {perr[3]:.2g}<br>"
                f"R² {R2:.5f}"
            )
            figH.add_annotation(xref="paper", yref="paper", x=0.62, y=0.85, align="left",
                                showarrow=False, bordercolor="black", borderwidth=1,
                                bgcolor="rgba(255,255,255,0.85)", text=text)
    figH.update_layout(xaxis_title="Value", yaxis_title="Density")
    st.plotly_chart(figH, use_container_width=True)

# -------- Overlay: Classical vs PIE --------
with tabs[2]:
    st.subheader("Overlay: Classical vs PIE (E and S)")
    cand = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[1] >= 8]
    if not cand:
        st.info("No table with ≥ 8 columns found.")
    else:
        sel = st.selectbox("Choose the 8+ column block", cand,
                           format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
        dfX = blocks[sel][0].copy()
        base_names = ["Occur._S_Classical","S_Classical","Occur._S_PIE","S_PIE",
                      "E_Classical","Occur._E_Classical","E_PIE","Occur._E_PIE"]
        extra = [f"Extra_{i}" for i in range(max(0, dfX.shape[1]-8))]
        dfX.columns = base_names + extra

        mode = st.radio("Overlay variable", ["Stoichiometry S","FRET efficiency E"], index=0)
        rule = st.selectbox("Auto-binning rule", ["Freedman–Diaconis","Scott","Sturges"], index=0)
        auto_fit = st.checkbox("Auto-fit Gaussian", value=True)

        if mode == "Stoichiometry S":
            x_cl, w_cl = dfX["S_Classical"].to_numpy(), dfX["Occur._S_Classical"].to_numpy()
            x_pie, w_pie = dfX["S_PIE"].to_numpy(), dfX["Occur._S_PIE"].to_numpy()
            xlabel = "Stoichiometry [S]"; legends = ("Classical (S)", "PIE (S)")
        else:
            x_cl, w_cl = dfX["E_Classical"].to_numpy(), dfX["Occur._E_Classical"].to_numpy()
            x_pie, w_pie = dfX["E_PIE"].to_numpy(), dfX["Occur._E_PIE"].to_numpy()
            xlabel = "PIE FRET [E]"; legends = ("Classical (E)", "PIE (E)")

        def clean_pair(x, w):
            x = np.asarray(x, float); w = np.asarray(w, float)
            m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
            return x[m], w[m]

        x_cl, w_cl = clean_pair(x_cl, w_cl)
        x_pie, w_pie = clean_pair(x_pie, w_pie)

        xmin = float(np.nanmin([np.nanmin(x_cl), np.nanmin(x_pie)]))
        xmax = float(np.nanmax([np.nanmax(x_cl), np.nanmax(x_pie)]))
        nb = auto_bins(np.concatenate([x_cl, x_pie]), rule=rule)
        st.caption(f"Auto-bins ({rule}) → {nb} bins")
        edges = np.linspace(xmin, xmax, nb+1)
        centers = 0.5*(edges[:-1]+edges[1:])
        hist_cl, _ = np.histogram(x_cl, bins=edges, weights=w_cl, density=True)
        hist_pie, _ = np.histogram(x_pie, bins=edges, weights=w_pie, density=True)

        fig = go.Figure()
        fig.add_bar(x=centers, y=hist_cl, width=np.diff(edges), name=legends[0], opacity=0.5)
        fig.add_bar(x=centers, y=hist_pie, width=np.diff(edges), name=legends[1], opacity=0.5)

        if auto_fit:
            fit_cl = smart_fit(centers, hist_cl, xmin, xmax)
            fit_pie = smart_fit(centers, hist_pie, xmin, xmax)
            xs = np.linspace(xmin, xmax, 1000)
            if fit_cl:
                p, perr, R2 = fit_cl
                fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, *p), mode="lines", name=f"{legends[0]} fit (R²={R2:.3f})"))
            if fit_pie:
                p, perr, R2 = fit_pie
                fig.add_trace(go.Scatter(x=xs, y=gaussian(xs, *p), mode="lines", name=f"{legends[1]} fit (R²={R2:.3f})"))

        fig.update_layout(xaxis_title=xlabel, yaxis_title="H [Occur.·10^3 Events] (density)")
        st.plotly_chart(fig, use_container_width=True)
