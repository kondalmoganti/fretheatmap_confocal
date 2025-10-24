
import re, io, math, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="FRET Analyzer (Heatmap + Origin-like Histogram)", layout="wide")
st.title("FRET Analyzer")

def split_numeric_blocks(text: str):
    text = text.replace(",", ".")
    lines = text.splitlines()
    blocks = []
    cur = []
    num_re = re.compile(r'^\s*[\d\.eE\-\+]+([\s\t,;][\d\.eE\-\+]+)*\s*$')

    def flush(start_i, end_i):
        if not cur: return
        s = "\n".join(cur).strip()
        try:
            df = pd.read_csv(io.StringIO(s), sep=r"[\s,;]+", engine="python", header=None)
        except Exception:
            return
        r, c = df.shape
        if r >= 10 and c >= 10: kind = "matrix"
        elif c == 1: kind = "vector"
        else: kind = "table"
        blocks.append((kind, df, start_i, end_i))

    start = None
    for i, ln in enumerate(lines):
        if num_re.match(ln):
            if start is None: start = i
            cur.append(ln)
        else:
            if start is not None:
                flush(start, i-1); cur = []; start = None
    if start is not None: flush(start, len(lines)-1)
    return blocks

def gaussian(x, y0, mu, sigma, A):
    amp = A / (sigma * np.sqrt(2*np.pi))
    return y0 + amp * np.exp(-0.5 * ((x - mu)/sigma)**2)

def r2_score(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.nansum((y - yhat)**2); ss_tot = np.nansum((y - np.nanmean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot>0 else np.nan

uploaded = st.file_uploader("Upload your .dat file", type=["dat","txt","csv"])

if uploaded is None:
    st.info("Upload your file to continue.")
else:
    raw = uploaded.getvalue().decode("utf-8", errors="ignore")
    blocks = split_numeric_blocks(raw)

    tabs = st.tabs(["Heatmap", "Histogram (Origin style)"])

    with tabs[0]:
        st.subheader("Correlogram Heatmap")
        mats = [i for i,(k,df,_,_) in enumerate(blocks) if k=="matrix"]
        if not mats: mats = list(range(len(blocks)))
        sel = st.selectbox("Choose block", mats, format_func=lambda i: f"Block {i} (shape {blocks[i][1].shape})")
        dfm = blocks[sel][1].astype(float).replace([np.inf,-np.inf], np.nan)
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

    with tabs[1]:
        st.subheader("Histogram + Gaussian fit (Origin style)")
        tbls = [i for i,(k,df,_,_) in enumerate(blocks) if k in ("table","vector","matrix")]
        sel = st.selectbox("Choose block for histogram source", tbls, index=tbls[-1] if tbls else 0,
                           format_func=lambda i: f"Block {i} (shape {blocks[i][1].shape})")
        dft = blocks[sel][1].copy(); dft.columns = [f"C{j}" for j in range(dft.shape[1])]
        st.dataframe(dft.head(12), use_container_width=True)

        c1,c2,c3 = st.columns(3)
        with c1: x_col = st.selectbox("Values column (x)", dft.columns, index=0)
        with c2: w_col = st.selectbox("Weights (optional)", ["(none)"]+list(dft.columns), index=0)
        with c3: nb = st.slider("Bins", 10, 200, 50, 1)

        values = pd.to_numeric(dft[x_col], errors="coerce").to_numpy()
        weights = None
        if w_col != "(none)":
            weights = pd.to_numeric(dft[w_col], errors="coerce").to_numpy()
            weights = np.where(np.isfinite(weights) & (weights>0), weights, 0.0)

        xmin = st.number_input("xmin", value=float(np.nanmin(values)), format="%.6f")
        xmax = st.number_input("xmax", value=float(np.nanmax(values)), format="%.6f")

        hist, edges = np.histogram(values[np.isfinite(values)], bins=nb, range=(xmin, xmax), weights=weights, density=True)
        centers = 0.5*(edges[:-1] + edges[1:])

        # Fit
        y0_0 = max(1e-9, float(np.min(hist))); mu_0 = float(np.average(centers, weights=hist+1e-12))
        sigma_0 = float(np.sqrt(np.average((centers-mu_0)**2, weights=hist+1e-12)))
        A_0 = float(np.trapz(hist, centers))
        popt, pcov = curve_fit(gaussian, centers, hist, p0=[y0_0, mu_0, max(sigma_0,1e-6), A_0], maxfev=20000)
        perr = np.sqrt(np.diag(pcov)); yhat = gaussian(centers, *popt); R2 = r2_score(hist, yhat)
        xfit = np.linspace(edges[0], edges[-1], 800); yfit = gaussian(xfit, *popt)

        xlabel = st.text_input("x label", "PIE FRET [E]")
        ylabel = st.text_input("y label", "H [Occur.·10^{3} Events]")
        show_box = st.checkbox("Show stats box on plot", value=True)

        figH = go.Figure()
        figH.add_trace(go.Bar(x=centers, y=hist, width=np.diff(edges), name="Histogram", opacity=0.55))
        figH.add_trace(go.Scatter(x=xfit, y=yfit, mode="lines", name="Gaussian fit"))
        figH.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)

        if show_box:
            text = (
                "<b>Modell</b>  Gauss<br>"
                "Gleichung  y = y₀ + A/(σ√(2π))·exp(-(x-xc)²/(2σ²))<br>"
                f"y₀  {popt[0]:.5g} ± {perr[0]:.2g}<br>"
                f"xc  {popt[1]:.5g} ± {perr[1]:.2g}<br>"
                f"w   {popt[2]:.5g} ± {perr[2]:.2g}<br>"
                f"A   {popt[3]:.5g} ± {perr[3]:.2g}<br>"
                f"R²  {R2:.5f}"
            )
            figH.add_annotation(xref="paper", yref="paper", x=0.58, y=0.85, align="left",
                                showarrow=False, bordercolor="black", borderwidth=1,
                                bgcolor="rgba(255,255,255,0.85)", text=text)

        st.plotly_chart(figH, use_container_width=True)
