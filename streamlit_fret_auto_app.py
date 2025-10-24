
import re, io, math, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="FRET Analyzer (heatmap + histograms + overlay)", layout="wide")
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

    tabs = st.tabs(["Heatmap", "Histogram (single)", "Overlay: Classical vs PIE"])

    # ---------------- HEATMAP ----------------
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

    # ---------------- SINGLE HISTOGRAM ----------------
    with tabs[1]:
        st.subheader("Histogram + Gaussian fit (single dataset)")
        tbls = [i for i,(k,df,_,_) in enumerate(blocks) if k in ("table","vector","matrix")]
        sel = st.selectbox("Choose block", tbls, index=tbls[-1] if tbls else 0,
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
        st.subheader("Overlay: Classical vs PIE FRET (E and S)")
        # Find candidate blocks with >= 8 columns
        cand = [i for i,(_,df,_,_) in enumerate(blocks) if df.shape[1] >= 8]
        if not cand:
            st.info("No table with >= 8 columns found. Pick the block in the previous tab instead.")
        else:
            sel = st.selectbox("Choose a block that contains Classical+PIE columns", cand,
                               format_func=lambda i: f"Block {i} (shape {blocks[i][1].shape})")
            df8 = blocks[sel][1].copy()
            # by position mapping based on your header example:
            # [0]=Classical Occur (S), [1]=Classical S
            # [2]=PIE Occur (S), [3]=PIE S
            # [4]=Classical E, [5]=Classical Occur (E)
            # [6]=PIE E, [7]=PIE Occur (E)
            # sanitize
            df8 = df8.replace(",", ".", regex=True)
            dfn = pd.DataFrame()
            for c in range(8):
                dfn[c] = pd.to_numeric(df8.iloc[:, c], errors="coerce")
            # --- Overlay for E ---
            st.markdown("### FRET Efficiency (E) — Classical vs PIE")
            nb_e = st.slider("Bins (E)", 20, 200, 80, 1, key="bins_e")
            xmin_e = st.number_input("E min", value=float(np.nanmin([dfn[4].min(), dfn[6].min()])), format="%.6f", key="emin")
            xmax_e = st.number_input("E max", value=float(np.nanmax([dfn[4].max(), dfn[6].max()])), format="%.6f", key="emax")

            # common edges for both so shapes are comparable
            edges_e = np.linspace(xmin_e, xmax_e, nb_e+1)
            hist_cl_e, _ = np.histogram(dfn[4].values, bins=edges_e, weights=dfn[5].values, density=True)
            hist_pie_e, _ = np.histogram(dfn[6].values, bins=edges_e, weights=dfn[7].values, density=True)
            centers_e = 0.5*(edges_e[:-1]+edges_e[1:])

            figE = go.Figure()
            figE.add_trace(go.Bar(x=centers_e, y=hist_cl_e, width=np.diff(edges_e), name="Classical (E)", opacity=0.45))
            figE.add_trace(go.Bar(x=centers_e, y=hist_pie_e, width=np.diff(edges_e), name="PIE (E)", opacity=0.45))

            # fits
            fit_cl_e = try_gauss_fit(centers_e, hist_cl_e, xmin_e, xmax_e)
            fit_pie_e = try_gauss_fit(centers_e, hist_pie_e, xmin_e, xmax_e)
            if fit_cl_e:
                p, s = fit_cl_e
                xfit = np.linspace(xmin_e, xmax_e, 800); yfit = gaussian(xfit, *p)
                figE.add_trace(go.Scatter(x=xfit, y=yfit, mode="lines", name="Classical fit (E)"))
            if fit_pie_e:
                p, s = fit_pie_e
                xfit = np.linspace(xmin_e, xmax_e, 800); yfit = gaussian(xfit, *p)
                figE.add_trace(go.Scatter(x=xfit, y=yfit, mode="lines", name="PIE fit (E)"))
            figE.update_layout(xaxis_title="E", yaxis_title="Density")
            st.plotly_chart(figE, use_container_width=True)

            # stats table
            def tbl_row(tag, fit):
                if not fit: return [tag, np.nan, np.nan, np.nan, np.nan, np.nan]
                p, s = fit
                return [tag, p[1], s[1], p[2], s[2], p[3]]
            tbl = pd.DataFrame([
                ["Dataset", "mu (xc)", "±", "sigma (w)", "±", "Area A"],
                tbl_row("Classical E", fit_cl_e),
                tbl_row("PIE E", fit_pie_e),
            ])
            st.dataframe(tbl, use_container_width=True)

            st.markdown("---")
            # --- Overlay for S ---
            st.markdown("### Stoichiometry (S) — Classical vs PIE")
            nb_s = st.slider("Bins (S)", 20, 200, 80, 1, key="bins_s")
            xmin_s = st.number_input("S min", value=float(np.nanmin([dfn[1].min(), dfn[3].min()])), format="%.6f", key="smin")
            xmax_s = st.number_input("S max", value=float(np.nanmax([dfn[1].max(), dfn[3].max()])), format="%.6f", key="smax")
            edges_s = np.linspace(xmin_s, xmax_s, nb_s+1)
            hist_cl_s, _ = np.histogram(dfn[1].values, bins=edges_s, weights=dfn[0].values, density=True)
            hist_pie_s, _ = np.histogram(dfn[3].values, bins=edges_s, weights=dfn[2].values, density=True)
            centers_s = 0.5*(edges_s[:-1]+edges_s[1:])

            figS = go.Figure()
            figS.add_trace(go.Bar(x=centers_s, y=hist_cl_s, width=np.diff(edges_s), name="Classical (S)", opacity=0.45))
            figS.add_trace(go.Bar(x=centers_s, y=hist_pie_s, width=np.diff(edges_s), name="PIE (S)", opacity=0.45))

            fit_cl_s = try_gauss_fit(centers_s, hist_cl_s, xmin_s, xmax_s)
            fit_pie_s = try_gauss_fit(centers_s, hist_pie_s, xmin_s, xmax_s)
            if fit_cl_s:
                p, s = fit_cl_s
                xfit = np.linspace(xmin_s, xmax_s, 800); yfit = gaussian(xfit, *p)
                figS.add_trace(go.Scatter(x=xfit, y=yfit, mode="lines", name="Classical fit (S)"))
            if fit_pie_s:
                p, s = fit_pie_s
                xfit = np.linspace(xmin_s, xmax_s, 800); yfit = gaussian(xfit, *p)
                figS.add_trace(go.Scatter(x=xfit, y=yfit, mode="lines", name="PIE fit (S)"))
            figS.update_layout(xaxis_title="S", yaxis_title="Density")
            st.plotly_chart(figS, use_container_width=True)
