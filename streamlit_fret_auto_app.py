
import re, io, math, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="FRET Analyzer (named columns)", layout="wide")
st.title("FRET Analyzer")

def split_numeric_blocks_with_headers(text: str):
    """
    Split the file into blocks of numeric rows.
    Also keep the preceding 1-3 non-numeric 'header' lines for each block.
    """
    text_norm = text.replace(",", ".")
    lines = text_norm.splitlines()
    is_num = lambda ln: re.match(r'^\s*[\d\.eE\-\+]+([\s\t,;][\d\.eE\-\+]+)*\s*$', ln) is not None

    blocks = []
    cur = []
    header_buf = []  # store last few non-numeric lines
    start_idx = None

    def flush(end_idx):
        nonlocal cur, header_buf, start_idx
        if not cur:
            return
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
            # non-numeric line: keep as potential header, and flush if we were in a block
            header_buf.append(ln.strip())
            header_buf = header_buf[-3:]  # keep last 3
            if start_idx is not None:
                flush(i-1)

    # flush at EOF
    if start_idx is not None:
        flush(len(lines)-1)

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
    blocks = split_numeric_blocks_with_headers(raw)

    tabs = st.tabs(["Heatmap", "Histogram (single)", "Overlay: Classical vs PIE"])

    # ---------------- HEATMAP ----------------
    with tabs[0]:
        st.subheader("Correlogram Heatmap")
        mats = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[0] >= 10 and df.shape[1] >= 10]
        if not mats:
            st.info("No clear matrix detected; you can still pick any block below.")
            mats = list(range(len(blocks)))
        sel = st.selectbox("Choose block", mats, format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
        dfm = blocks[sel][0].astype(float).replace([np.inf,-np.inf], np.nan)
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

    # ---------------- SINGLE HISTOGRAM ----------------
    with tabs[1]:
        st.subheader("Histogram + Gaussian fit (single dataset)")
        tbls = list(range(len(blocks)))
        sel = st.selectbox("Choose block", tbls, index=tbls[-1] if tbls else 0,
                           format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
        dft = blocks[sel][0].copy()
        # Allow renaming columns from a quick template
        st.markdown("**Optional: name your columns**")
        templ = st.radio("Template", ["Generic (C0..)", "FRET 8-cols (Occur/S/E mapping)"], index=1)
        if templ == "FRET 8-cols (Occur/S/E mapping)" and dft.shape[1] >= 8:
            base_names = ["Occur._S_Classical","S_Classical","Occur._S_PIE","S_PIE",
                          "E_Classical","Occur._E_Classical","E_PIE","Occur._E_PIE"]
            extra = [f"Extra_{i}" for i in range(dft.shape[1]-8)]
            dft.columns = base_names + extra
        else:
            dft.columns = [f"C{j}" for j in range(dft.shape[1])]
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

        valid = np.isfinite(values)
        hist, edges = np.histogram(values[valid], bins=nb, range=(xmin, xmax), weights=weights if weights is not None else None, density=True)
        centers = 0.5*(edges[:-1] + edges[1:])

        figH = go.Figure()
        figH.add_trace(go.Bar(x=centers, y=hist, width=np.diff(edges), name="Histogram", opacity=0.55))

        fit = try_gauss_fit(centers, hist, xmin, xmax)
        if fit is not None:
            popt, perr = fit
            xfit = np.linspace(edges[0], edges[-1], 800); yfit = gaussian(xfit, *popt)
            R2 = r2_score(hist, gaussian(centers, *popt))
            figH.add_trace(go.Scatter(x=xfit, y=yfit, mode="lines", name="Gaussian fit"))
            text = (
                "<b>Model</b> Gauss<br>"
                f"y₀ {popt[0]:.5g} ± {perr[0]:.2g}<br>"
                f"xc {popt[1]:.5g} ± {perr[1]:.2g}<br>"
                f"w  {popt[2]:.5g} ± {perr[2]:.2g}<br>"
                f"A  {popt[3]:.5g} ± {perr[3]:.2g}<br>"
                f"R² {R2:.5f}"
            )
            figH.add_annotation(xref="paper", yref="paper", x=0.58, y=0.85, align="left",
                                showarrow=False, bordercolor="black", borderwidth=1,
                                bgcolor="rgba(255,255,255,0.85)", text=text)
        st.plotly_chart(figH, use_container_width=True)

    # ---------------- OVERLAY: CLASSICAL vs PIE ----------------
    with tabs[2]:
        st.subheader("Overlay: Classical vs PIE (named columns)")
        # Find candidate blocks with >= 8 columns
        cand = [i for i,(df,_,_,_) in enumerate(blocks) if df.shape[1] >= 8]
        if not cand:
            st.info("No table with ≥ 8 numeric columns found.")
        else:
            sel = st.selectbox("Choose the 8+ column block", cand,
                               format_func=lambda i: f"Block {i} (shape {blocks[i][0].shape})")
            dfX = blocks[sel][0].copy()

            # Apply the friendly names
            base_names = ["Occur._S_Classical","S_Classical","Occur._S_PIE","S_PIE",
                          "E_Classical","Occur._E_Classical","E_PIE","Occur._E_PIE"]
            extra = [f"Extra_{i}" for i in range(dfX.shape[1]-8)]
            dfX.columns = base_names + extra

            st.write("Preview (with names):")
            st.dataframe(dfX.head(12), use_container_width=True)

            # Column selection with pre-selected meaningful names
            c1, c2 = st.columns(2)
            with c1:
                cl_s_val = st.selectbox("Classical S values", dfX.columns, index=dfX.columns.get_loc("S_Classical"))
                cl_s_w   = st.selectbox("Classical S weights", dfX.columns, index=dfX.columns.get_loc("Occur._S_Classical"))
                cl_e_val = st.selectbox("Classical E values", dfX.columns, index=dfX.columns.get_loc("E_Classical"))
                cl_e_w   = st.selectbox("Classical E weights", dfX.columns, index=dfX.columns.get_loc("Occur._E_Classical"))
            with c2:
                pie_s_val = st.selectbox("PIE S values", dfX.columns, index=dfX.columns.get_loc("S_PIE"))
                pie_s_w   = st.selectbox("PIE S weights", dfX.columns, index=dfX.columns.get_loc("Occur._S_PIE"))
                pie_e_val = st.selectbox("PIE E values", dfX.columns, index=dfX.columns.get_loc("E_PIE"))
                pie_e_w   = st.selectbox("PIE E weights", dfX.columns, index=dfX.columns.get_loc("Occur._E_PIE"))

            # Convert
            def num(s): return pd.to_numeric(dfX[s], errors="coerce").to_numpy()
            S_cl, W_S_cl = num(cl_s_val), num(cl_s_w)
            S_pie, W_S_pie = num(pie_s_val), num(pie_s_w)
            E_cl, W_E_cl   = num(cl_e_val), num(cl_e_w)
            E_pie, W_E_pie = num(pie_e_val), num(pie_e_w)

            # Helpers
            def clean_pair(v, w):
                v = np.asarray(v, float); w = np.asarray(w, float)
                m = np.isfinite(v) & np.isfinite(w) & (w >= 0)
                return v[m], w[m]

            S_cl, W_S_cl = clean_pair(S_cl, W_S_cl)
            S_pie, W_S_pie = clean_pair(S_pie, W_S_pie)
            E_cl, W_E_cl = clean_pair(E_cl, W_E_cl)
            E_pie, W_E_pie = clean_pair(E_pie, W_E_pie)

            # E overlay
            st.markdown("### FRET Efficiency (E)")
            nb_e = st.slider("Bins (E)", 20, 200, 80, 1)
            xmin_e = st.number_input("E min", value=float(np.nanmin([np.nanmin(E_cl), np.nanmin(E_pie)])), format="%.6f")
            xmax_e = st.number_input("E max", value=float(np.nanmax([np.nanmax(E_cl), np.nanmax(E_pie)])), format="%.6f")
            edges_e = np.linspace(xmin_e, xmax_e, nb_e+1)
            centers_e = 0.5*(edges_e[:-1]+edges_e[1:])
            hist_cl_e, _ = np.histogram(E_cl, bins=edges_e, weights=W_E_cl, density=True)
            hist_pie_e, _ = np.histogram(E_pie, bins=edges_e, weights=W_E_pie, density=True)

            figE = go.Figure()
            figE.add_trace(go.Bar(x=centers_e, y=hist_cl_e, width=np.diff(edges_e), name="Classical (E)", opacity=0.45))
            figE.add_trace(go.Bar(x=centers_e, y=hist_pie_e, width=np.diff(edges_e), name="PIE (E)", opacity=0.45))
            fit_cl_e = try_gauss_fit(centers_e, hist_cl_e, xmin_e, xmax_e)
            fit_pie_e = try_gauss_fit(centers_e, hist_pie_e, xmin_e, xmax_e)
            if fit_cl_e:
                p,_ = fit_cl_e; xfit = np.linspace(xmin_e, xmax_e, 800)
                figE.add_trace(go.Scatter(x=xfit, y=gaussian(xfit, *p), mode="lines", name="Classical fit (E)"))
            if fit_pie_e:
                p,_ = fit_pie_e; xfit = np.linspace(xmin_e, xmax_e, 800)
                figE.add_trace(go.Scatter(x=xfit, y=gaussian(xfit, *p), mode="lines", name="PIE fit (E)"))
            figE.update_layout(xaxis_title="PIE FRET [E]", yaxis_title="H [Occur.·10^3 Events]")
            st.plotly_chart(figE, use_container_width=True)

            st.markdown("---")

            # S overlay
            st.markdown("### Stoichiometry (S)")
            nb_s = st.slider("Bins (S)", 20, 200, 80, 1)
            xmin_s = st.number_input("S min", value=float(np.nanmin([np.nanmin(S_cl), np.nanmin(S_pie)])), format="%.6f")
            xmax_s = st.number_input("S max", value=float(np.nanmax([np.nanmax(S_cl), np.nanmax(S_pie)])), format="%.6f")
            edges_s = np.linspace(xmin_s, xmax_s, nb_s+1)
            centers_s = 0.5*(edges_s[:-1]+edges_s[1:])
            hist_cl_s, _ = np.histogram(S_cl, bins=edges_s, weights=W_S_cl, density=True)
            hist_pie_s, _ = np.histogram(S_pie, bins=edges_s, weights=W_S_pie, density=True)

            figS = go.Figure()
            figS.add_trace(go.Bar(x=centers_s, y=hist_cl_s, width=np.diff(edges_s), name="Classical (S)", opacity=0.45))
            figS.add_trace(go.Bar(x=centers_s, y=hist_pie_s, width=np.diff(edges_s), name="PIE (S)", opacity=0.45))
            fit_cl_s = try_gauss_fit(centers_s, hist_cl_s, xmin_s, xmax_s)
            fit_pie_s = try_gauss_fit(centers_s, hist_pie_s, xmin_s, xmax_s)
            if fit_cl_s:
                p,_ = fit_cl_s; xfit = np.linspace(xmin_s, xmax_s, 800)
                figS.add_trace(go.Scatter(x=xfit, y=gaussian(xfit, *p), mode="lines", name="Classical fit (S)"))
            if fit_pie_s:
                p,_ = fit_pie_s; xfit = np.linspace(xmin_s, xmax_s, 800)
                figS.add_trace(go.Scatter(x=xfit, y=gaussian(xfit, *p), mode="lines", name="PIE fit (S)"))
            figS.update_layout(xaxis_title="Stoichiometry [S]", yaxis_title="H [Occur.·10^3 Events]")
            st.plotly_chart(figS, use_container_width=True)
