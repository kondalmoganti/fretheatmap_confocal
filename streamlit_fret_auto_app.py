
import re, io, math, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="FRET Analyzer (Multi‑section .dat)", layout="wide")
st.title("FRET Analyzer (Multi‑section .dat)")

st.caption("Parses Origin-like .dat files that may contain multiple numeric sections (matrices, repeated blocks, and labeled tables).")

def split_numeric_blocks(text: str):
    text = text.replace(",", ".")
    lines = text.splitlines()
    blocks = []
    cur = []
    num_re = re.compile(r'^\s*[\d\.eE\-\+]+([\s\t,;][\d\.eE\-\+]+)*\s*$')

    def flush(start_i, end_i):
        if not cur:
            return
        s = "\n".join(cur).strip()
        try:
            df = pd.read_csv(io.StringIO(s), sep=r"[\s,;]+", engine="python", header=None)
        except Exception:
            return
        r, c = df.shape
        if r >= 10 and c >= 10:
            kind = "matrix"
        elif c == 1:
            kind = "vector"
        else:
            kind = "table"
        blocks.append((kind, df, start_i, end_i))

    start = None
    for i, ln in enumerate(lines):
        if num_re.match(ln):
            if start is None:
                start = i
            cur.append(ln)
        else:
            if start is not None:
                flush(start, i-1)
                cur = []
                start = None
    if start is not None:
        flush(start, len(lines)-1)
    return blocks, lines

def gaussian(x, y0, mu, sigma, A):
    amp = A / (sigma * np.sqrt(2*np.pi))
    return y0 + amp * np.exp(-0.5 * ((x - mu)/sigma)**2)

def r2_score(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.nansum((y - yhat)**2)
    ss_tot = np.nansum((y - np.nanmean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot>0 else np.nan

uploaded = st.file_uploader("Upload your .dat file", type=["dat","txt","csv"])

if uploaded is None:
    st.info("Upload the sample .dat you pasted to see all sections parsed. Decimal commas are handled automatically.")
    st.stop()

raw = uploaded.getvalue().decode("utf-8", errors="ignore")
blocks, lines = split_numeric_blocks(raw)

st.subheader("Detected numeric sections")
if len(blocks)==0:
    st.error("No numeric blocks detected."); st.stop()

summary = [{"#":bi, "kind":k, "rows":df.shape[0], "cols":df.shape[1], "start_line":si, "end_line":ei}
           for bi,(k,df,si,ei) in enumerate(blocks)]
st.dataframe(pd.DataFrame(summary), use_container_width=True)

st.divider()
st.subheader("Correlogram (pick a matrix block)")
matrix_indices = [i for i,(k,df,_,_) in enumerate(blocks) if k=="matrix"]
if not matrix_indices:
    matrix_indices = list(range(len(blocks)))
sel_mat = st.selectbox("Choose block for heatmap", matrix_indices, format_func=lambda i: f"Block {i} (shape {blocks[i][1].shape})")
dfm = blocks[sel_mat][1].astype(float).replace([np.inf,-np.inf], np.nan)

left, right = st.columns([0.45, 0.55])
with left:
    zmin = st.number_input("zmin (0=auto)", value=0.0)
    zmax = st.number_input("zmax (0=auto)", value=0.0)
    smooth = st.slider("Gaussian smoothing (σ)", 0.0, 6.0, 1.0, 0.1)

arr = dfm.to_numpy()
if smooth>0:
    arrp = gaussian_filter1d(gaussian_filter1d(arr, sigma=smooth, axis=0), sigma=smooth, axis=1)
else:
    arrp = arr.copy()

with right:
    zmin_use = None if zmin<=0 else zmin
    zmax_use = None if zmax<=0 else zmax
    fig = px.imshow(arrp, origin="lower", aspect="auto", color_continuous_scale="Viridis",
                    zmin=zmin_use, zmax=zmax_use,
                    labels=dict(x="E bins (columns)", y="S bins (rows)", color="Counts"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("**Marginals & ridge**")
c1,c2,c3 = st.columns(3)
with c1:
    cm = np.nansum(arr, axis=0); figc = go.Figure(); figc.add_trace(go.Scatter(y=cm, mode="lines"))
    st.plotly_chart(figc, use_container_width=True)
with c2:
    rm = np.nansum(arr, axis=1); figr = go.Figure(); figr.add_trace(go.Scatter(y=rm, mode="lines"))
    st.plotly_chart(figr, use_container_width=True)
with c3:
    ridge_y = np.nanargmax(arr, axis=0); ridge_x = np.arange(arr.shape[1])
    figrg = go.Figure(); figrg.add_trace(go.Scatter(x=ridge_x, y=ridge_y, mode="markers+lines"))
    st.plotly_chart(figrg, use_container_width=True)

st.divider()
st.subheader("Histogram + Gaussian fit (from any table/vector block)")

table_indices = [i for i,(k,df,_,_) in enumerate(blocks) if k in ("table","vector","matrix")]
sel_tab = st.selectbox("Choose block for histogram source", table_indices, index=table_indices[-1],
                       format_func=lambda i: f"Block {i} (shape {blocks[i][1].shape})")

dft = blocks[sel_tab][1].copy()
dft.columns = [f"C{j}" for j in range(dft.shape[1])]
st.write("Preview of selected block:")
st.dataframe(dft.head(12), use_container_width=True)

cA, cB = st.columns(2)
with cA:
    x_col = st.selectbox("Column = values (x)", dft.columns, index=0)
with cB:
    w_col = st.selectbox("Optional weights column", ["(none)"] + list(dft.columns), index=0)

values = pd.to_numeric(dft[x_col], errors="coerce").to_numpy()
weights = None
if w_col != "(none)":
    weights = pd.to_numeric(dft[w_col], errors="coerce").to_numpy()
    weights = np.where(np.isfinite(weights) & (weights>0), weights, 0.0)

nb = st.slider("Number of histogram bins", 10, 200, 50, 1)
xmin = st.number_input("xmin", value=float(np.nanmin(values)), format="%.6f")
xmax = st.number_input("xmax", value=float(np.nanmax(values)), format="%.6f")

hist, edges = np.histogram(values[np.isfinite(values)], bins=nb, range=(xmin, xmax), weights=weights, density=True)
centers = 0.5*(edges[:-1] + edges[1:])
figH = go.Figure(); figH.add_trace(go.Bar(x=centers, y=hist, width=np.diff(edges), opacity=0.5))
sm_sigma = st.slider("Display smoothing σ", 0.0, 10.0, 1.2, 0.1)
if sm_sigma>0:
    sm = gaussian_filter1d(hist, sigma=sm_sigma)
    figH.add_trace(go.Scatter(x=centers, y=sm, mode="lines", name="Smoothed"))
figH.update_layout(xaxis_title="Value", yaxis_title="Density")
st.plotly_chart(figH, use_container_width=True)

st.markdown("### Gaussian fit to histogram")
y0_0 = max(1e-9, float(np.min(hist)))
mu_0 = float(np.average(centers, weights=hist+1e-12))
sigma_0 = float(np.sqrt(np.average((centers-mu_0)**2, weights=hist+1e-12)))
A_0 = float(np.trapz(hist, centers))
try:
    popt, pcov = curve_fit(gaussian, centers, hist, p0=[y0_0, mu_0, max(sigma_0,1e-6), A_0], maxfev=20000)
    perr = np.sqrt(np.diag(pcov)); yhat = gaussian(centers, *popt)
    R2 = 1 - np.nansum((hist-yhat)**2)/np.nansum((hist-np.nanmean(hist))**2) if np.nansum((hist-np.nanmean(hist))**2)>0 else np.nan
    st.dataframe(pd.DataFrame({"Parameter":["y0","xc (mu)","w (sigma)","A (area)","R^2"],
                               "Value":[popt[0], popt[1], popt[2], popt[3], R2],
                               "Std.Err":[perr[0], perr[1], perr[2], perr[3], np.nan]}), use_container_width=True)
    xfit = np.linspace(edges[0], edges[-1], 800); yfit = gaussian(xfit, *popt)
    figF = go.Figure(); figF.add_trace(go.Bar(x=centers, y=hist, width=np.diff(edges), opacity=0.45))
    figF.add_trace(go.Scatter(x=xfit, y=yfit, mode="lines", name="Gaussian fit"))
    figF.update_layout(xaxis_title="Value", yaxis_title="Density")
    st.plotly_chart(figF, use_container_width=True)
except Exception as e:
    st.warning(f"Gaussian fit failed: {e}")
