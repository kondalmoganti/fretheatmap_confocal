
import io
import os
import re
import math
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="FRET Analyzer: Correlogram + Histograms", layout="wide")
st.title("FRET Analyzer")
st.caption("Upload Origin-like `.dat` matrices for S vs. E AND/OR 1D vectors for histogramming & Gaussian fits.")

def parse_origin_like_dat(file_bytes: bytes):
    """Parse an Origin-like .dat correlogram with EU decimal commas & tab/space separation.
    Returns: (df: DataFrame [n_rows x n_cols], metadata_lines: list[str])
    """
    text = file_bytes.decode("ascii", errors="ignore")
    lines = text.strip().splitlines()

    # Keep metadata lines that aren't purely numeric/tab/comma/dot/space
    num_line_re = re.compile(r'^[\d,\t\.\-\s]+$')
    metadata = [ln for ln in lines if not num_line_re.match(ln)]
    numeric_lines = [ln for ln in lines if num_line_re.match(ln)]

    rows = []
    max_len = 0
    for ln in numeric_lines:
        # Convert EU decimal commas to dots
        ln2 = ln.replace(",", ".")
        # Prefer tabs, fallback to any whitespace
        parts = [p for p in ln2.split("\t") if p != ""]
        if len(parts) == 1:
            parts = re.split(r"\s+", ln2.strip())
        try:
            row = [float(p) for p in parts]
        except ValueError:
            continue
        rows.append(row)
        max_len = max(max_len, len(row))

    rows = [r + [math.nan] * (max_len - len(r)) for r in rows] if rows else []
    df = pd.DataFrame(rows)
    return df, metadata

def gaussian(x, y0, mu, sigma, A):
    # A is the AREA under the Gaussian peak (not the amplitude).
    # amplitude = A / (sigma * np.sqrt(2*np.pi))
    amp = A / (sigma * np.sqrt(2*np.pi))
    return y0 + amp * np.exp(-0.5 * ((x - mu)/sigma)**2)

def r2_score(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    ss_res = np.nansum((y - yhat)**2)
    ss_tot = np.nansum((y - np.nanmean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

tabs = st.tabs(["Correlogram (S vs E)", "Histogram + Gaussian Fit"])

# -------------------- Correlogram tab --------------------
with tabs[0]:
    st.subheader("Upload `.dat` correlogram")
    uploaded = st.file_uploader("Upload Matrix `.dat` (Origin-like numeric table)", type=["dat","txt","csv"], key="corr")
    if uploaded is not None:
        df, meta = parse_origin_like_dat(uploaded.getvalue())
        st.write("Matrix shape:", df.shape)
        if meta:
            with st.expander("Metadata (first 5 lines)"):
                st.code("\n".join(meta[:5]))

        left, right = st.columns([0.4, 0.6])
        with left:
            zmin = st.number_input("zmin (0 = auto)", value=0.0)
            zmax = st.number_input("zmax (0 = auto)", value=0.0)
            smoothing = st.slider("Gaussian smoothing (σ)", 0.0, 5.0, 1.0, 0.1)
        df_np = df.to_numpy(float)
        if smoothing > 0:
            df_plot = gaussian_filter1d(df_np, sigma=smoothing, axis=0)
            df_plot = gaussian_filter1d(df_plot, sigma=smoothing, axis=1)
        else:
            df_plot = df_np.copy()
        zmin_use = None if zmin <= 0 else zmin
        zmax_use = None if zmax <= 0 else zmax

        with right:
            fig = px.imshow(df_plot, origin="lower", aspect="auto",
                            color_continuous_scale="Viridis",
                            zmin=zmin_use, zmax=zmax_use,
                            labels=dict(x="E bins (columns)", y="S bins (rows)", color="Counts"))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Marginals & Ridge")
        col1, col2, col3 = st.columns(3)
        with col1:
            col_m = np.nansum(df_np, axis=0)
            fig_col = go.Figure()
            fig_col.add_trace(go.Scatter(y=col_m, mode="lines", name="col marginal"))
            st.plotly_chart(fig_col, use_container_width=True)
        with col2:
            row_m = np.nansum(df_np, axis=1)
            fig_row = go.Figure()
            fig_row.add_trace(go.Scatter(y=row_m, mode="lines", name="row marginal"))
            st.plotly_chart(fig_row, use_container_width=True)
        with col3:
            ridge_y = np.nanargmax(df_np, axis=0)
            ridge_x = np.arange(df_np.shape[1])
            fig_ridge = go.Figure()
            fig_ridge.add_trace(go.Scatter(x=ridge_x, y=ridge_y, mode="markers+lines", name="ridge"))
            st.plotly_chart(fig_ridge, use_container_width=True)

        st.info("Use the next tab to build histograms from any row/column or upload a 1D vector.")

# -------------------- Histogram tab --------------------
with tabs[1]:
    st.subheader("Histogram & Gaussian Fit")
    st.write("Source of data for histogram:")
    source = st.radio("", ["Upload 1D values file", "Use column/row from a matrix file"], horizontal=True)

    values = None
    if source == "Upload 1D values file":
        one_d = st.file_uploader("Upload a 1D list (txt/csv). One number per line or separated by tabs/spaces/commas.", type=["txt","csv","dat"], key="vec")
        if one_d is not None:
            raw = one_d.getvalue().decode("utf-8", errors="ignore").replace(",", ".")
            tokens = re.split(r"[\s,;]+", raw.strip())
            vals = []
            for t in tokens:
                try:
                    vals.append(float(t))
                except:
                    pass
            if len(vals) > 0:
                values = np.array(vals, dtype=float)
    else:
        # from matrix
        uploaded2 = st.file_uploader("Upload Matrix `.dat` (Origin-like numeric table)", type=["dat","txt","csv"], key="corr2")
        if uploaded2 is not None:
            df2, _ = parse_origin_like_dat(uploaded2.getvalue())
            mode = st.radio("Pick trace", ["Column (fix E)", "Row (fix S)", "All values (flatten)"], horizontal=True)
            if mode == "Column (fix E)":
                idx = st.number_input("Column index", 0, df2.shape[1]-1, min(10, df2.shape[1]-1))
                values = df2.iloc[:, int(idx)].to_numpy(dtype=float)
            elif mode == "Row (fix S)":
                idx = st.number_input("Row index", 0, df2.shape[0]-1, min(10, df2.shape[0]-1))
                values = df2.iloc[int(idx), :].to_numpy(dtype=float)
            else:
                values = df2.to_numpy(dtype=float).ravel()
                values = values[np.isfinite(values)]

    if values is None:
        st.info("Upload or pick a source to build the histogram.")
    else:
        st.write(f"Loaded {len(values)} values.")
        c1, c2 = st.columns(2)
        with c1:
            nbins = st.slider("Number of bins", 10, 200, 50, 1)
            xmin = st.number_input("xmin", value=float(np.nanmin(values)), format="%.6f")
            xmax = st.number_input("xmax", value=float(np.nanmax(values)), format="%.6f")
        with c2:
            do_kde = st.checkbox("Show smoothed curve (KDE-like)", value=True)
            smooth_sigma = st.slider("Smoothing σ (for displayed curve)", 0.0, 10.0, 1.5, 0.1)

        # histogram
        hist, edges = np.histogram(values[np.isfinite(values)], bins=nbins, range=(xmin, xmax), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        figh = go.Figure()
        figh.add_trace(go.Bar(x=centers, y=hist, width=np.diff(edges), name="Histogram", opacity=0.5))
        if do_kde:
            sm = gaussian_filter1d(hist, sigma=smooth_sigma) if smooth_sigma > 0 else hist
            figh.add_trace(go.Scatter(x=centers, y=sm, mode="lines", name="Smoothed"))
        figh.update_layout(xaxis_title="Value", yaxis_title="Density")
        st.plotly_chart(figh, use_container_width=True)

        st.markdown("### Gaussian fit")
        # initial guesses
        y0_0 = max(1e-6, float(np.min(hist)))
        mu_0 = float(np.average(centers, weights=hist+1e-12))
        sigma_0 = float(np.sqrt(np.average((centers-mu_0)**2, weights=hist+1e-12)))
        A_0 = float(np.sum(hist) * (edges[-1]-edges[0]))
        try:
            popt, pcov = curve_fit(gaussian, centers, hist, p0=[y0_0, mu_0, max(sigma_0, 1e-6), A_0], maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
            yhat = gaussian(centers, *popt)
            rr = r2_score(hist, yhat)

            table = pd.DataFrame({
                "Parameter": ["y0", "xc (mu)", "w (sigma)", "A (area)", "R^2"],
                "Value": [popt[0], popt[1], popt[2], popt[3], rr],
                "Std.Err": [perr[0], perr[1], perr[2], perr[3], np.nan],
            })
            st.dataframe(table, use_container_width=True)

            # overlay fit
            xfit = np.linspace(edges[0], edges[-1], 800)
            yfit = gaussian(xfit, *popt)
            figh2 = go.Figure()
            figh2.add_trace(go.Bar(x=centers, y=hist, width=np.diff(edges), name="Histogram", opacity=0.45))
            figh2.add_trace(go.Scatter(x=xfit, y=yfit, mode="lines", name="Gaussian fit"))
            figh2.update_layout(xaxis_title="Value", yaxis_title="Density")
            st.plotly_chart(figh2, use_container_width=True)
        except Exception as e:
            st.warning(f"Gaussian fit failed: {e}")
