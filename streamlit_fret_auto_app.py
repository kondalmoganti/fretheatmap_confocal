
import re, io, math, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="FRET Analyzer (explicit column mapping)", layout="wide")
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

def try_gauss_fit(centers, hist, xmin, xmax):
    nonzero = np.isfinite(hist) & (hist > 0)
    if nonzero.sum() < 5:
        return None
    mu0 = float(np.average(centers[nonzero], weights=hist[nonzero]))
    var0 = float(np.average((centers[nonzero]-mu0)**2, weights=hist[nonzero]))
    sigma0 = max(1e-6, np.sqrt(var0))
    y00 = max(1e-9, float(np.min(hist[nonzero]) * 0.5))
    A0 = float(np.trapz(hist[nonzero], centers[nonzero]))
    lower = [0.0, xmin, 1e-6, 0.0]
    upper = [float(np.max(hist)*2), xmax, (xmax-xmin)*2, np.inf]
    for mul in (1.0, 0.5, 2.0, 0.25, 4.0):
        p0 = [y00, mu0, sigma0*mul, max(A0, 1e-6)]
        try:
            popt, pcov = curve_fit(gaussian, centers[nonzero], hist[nonzero], p0=p0,
                                   bounds=(lower, upper), maxfev=60000)
            perr = np.sqrt(np.diag(pcov))
            return popt, perr
        except Exception:
            continue
    return None

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

    tabs = st.tabs(["Overlay: Classical vs PIE (explicit mapping)"])

    with tabs[0]:
        st.subheader("Overlay: map your columns exactly as headers")
        cand = [i for i,(_,df,_,_) in enumerate(blocks) if df.shape[1] >= 8]
        if not cand:
            st.info("No table with ≥ 8 numeric columns found.")
        else:
            sel = st.selectbox("Choose the 8+ column block", cand,
                               format_func=lambda i: f"Block {i} (shape {blocks[i][1].shape})")
            dfX = blocks[sel][1].copy()
            dfX = dfX.replace(",", ".", regex=True)
            # create named columns
            dfX.columns = [f"C{j}" for j in range(dfX.shape[1])]
            st.dataframe(dfX.head(10), use_container_width=True)

            # Column selectors (defaults match your header order)
            cols = dfX.columns.tolist()
            st.markdown("**Map columns**")
            c1,c2 = st.columns(2)
            with c1:
                idx_cl_s_val = st.selectbox("Classical S values (S[])", cols, index=1)
                idx_cl_s_w   = st.selectbox("Classical S weights (Occur.)", cols, index=0)
                idx_cl_e_val = st.selectbox("Classical E values (E[])", cols, index=4)
                idx_cl_e_w   = st.selectbox("Classical E weights (Occur.)", cols, index=5)
            with c2:
                idx_pie_s_val = st.selectbox("PIE S values (S[])", cols, index=3)
                idx_pie_s_w   = st.selectbox("PIE S weights (Occur.)", cols, index=2)
                idx_pie_e_val = st.selectbox("PIE E values (E[])", cols, index=6)
                idx_pie_e_w   = st.selectbox("PIE E weights (Occur.)", cols, index=7)

            # Convert to numeric arrays
            def get_num(col):
                return pd.to_numeric(dfX[col], errors="coerce").to_numpy()

            S_cl, W_S_cl = get_num(idx_cl_s_val), get_num(idx_cl_s_w)
            S_pie, W_S_pie = get_num(idx_pie_s_val), get_num(idx_pie_s_w)
            E_cl, W_E_cl = get_num(idx_cl_e_val), get_num(idx_cl_e_w)
            E_pie, W_E_pie = get_num(idx_pie_e_val), get_num(idx_pie_e_w)

            # E overlay
            st.markdown("### FRET Efficiency E — Classical vs PIE")
            nb_e = st.slider("Bins (E)", 20, 200, 80, 1)
            xmin_e = st.number_input("E min", value=float(np.nanmin([np.nanmin(E_cl), np.nanmin(E_pie)])), format="%.6f")
            xmax_e = st.number_input("E max", value=float(np.nanmax([np.nanmax(E_cl), np.nanmax(E_pie)])), format="%.6f")
            edges_e = np.linspace(xmin_e, xmax_e, nb_e+1)
            hist_cl_e, _ = np.histogram(E_cl, bins=edges_e, weights=W_E_cl, density=True)
            hist_pie_e, _ = np.histogram(E_pie, bins=edges_e, weights=W_E_pie, density=True)
            centers_e = 0.5*(edges_e[:-1]+edges_e[1:])
            figE = go.Figure()
            figE.add_trace(go.Bar(x=centers_e, y=hist_cl_e, width=np.diff(edges_e), name="Classical (E)", opacity=0.45))
            figE.add_trace(go.Bar(x=centers_e, y=hist_pie_e, width=np.diff(edges_e), name="PIE (E)", opacity=0.45))
            fit_cl_e = try_gauss_fit(centers_e, hist_cl_e, xmin_e, xmax_e)
            fit_pie_e = try_gauss_fit(centers_e, hist_pie_e, xmin_e, xmax_e)
            if fit_cl_e:
                p,_=fit_cl_e; xfit=np.linspace(xmin_e,xmax_e,800); figE.add_trace(go.Scatter(x=xfit,y=gaussian(xfit,*p),name="Classical fit (E)"))
            if fit_pie_e:
                p,_=fit_pie_e; xfit=np.linspace(xmin_e,xmax_e,800); figE.add_trace(go.Scatter(x=xfit,y=gaussian(xfit,*p),name="PIE fit (E)"))
            figE.update_layout(xaxis_title="E", yaxis_title="Density")
            st.plotly_chart(figE, use_container_width=True)

            # S overlay
            st.markdown("### Stoichiometry S — Classical vs PIE")
            nb_s = st.slider("Bins (S)", 20, 200, 80, 1)
            xmin_s = st.number_input("S min", value=float(np.nanmin([np.nanmin(S_cl), np.nanmin(S_pie)])), format="%.6f")
            xmax_s = st.number_input("S max", value=float(np.nanmax([np.nanmax(S_cl), np.nanmax(S_pie)])), format="%.6f")
            edges_s = np.linspace(xmin_s, xmax_s, nb_s+1)
            hist_cl_s, _ = np.histogram(S_cl, bins=edges_s, weights=W_S_cl, density=True)
            hist_pie_s, _ = np.histogram(S_pie, bins=edges_s, weights=W_S_pie, density=True)
            centers_s = 0.5*(edges_s[:-1]+edges_s[1:])
            figS = go.Figure()
            figS.add_trace(go.Bar(x=centers_s, y=hist_cl_s, width=np.diff(edges_s), name="Classical (S)", opacity=0.45))
            figS.add_trace(go.Bar(x=centers_s, y=hist_pie_s, width=np.diff(edges_s), name="PIE (S)", opacity=0.45))
            fit_cl_s = try_gauss_fit(centers_s, hist_cl_s, xmin_s, xmax_s)
            fit_pie_s = try_gauss_fit(centers_s, hist_pie_s, xmin_s, xmax_s)
            if fit_cl_s:
                p,_=fit_cl_s; xfit=np.linspace(xmin_s,xmax_s,800); figS.add_trace(go.Scatter(x=xfit,y=gaussian(xfit,*p),name="Classical fit (S)"))
            if fit_pie_s:
                p,_=fit_pie_s; xfit=np.linspace(xmin_s,xmax_s,800); figS.add_trace(go.Scatter(x=xfit,y=gaussian(xfit,*p),name="PIE fit (S)"))
            figS.update_layout(xaxis_title="S", yaxis_title="Density")
            st.plotly_chart(figS, use_container_width=True)
