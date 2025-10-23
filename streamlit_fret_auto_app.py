
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

st.set_page_config(page_title="FRET Correlogram Analyzer", layout="wide")

st.title("FRET Correlogram Analyzer (Origin .dat → plots & fits)")

with st.expander("About this app", expanded=False):
    st.markdown(
        """
        This Streamlit app ingests Origin-like `.dat` files that contain a 2D correlogram
        (e.g., **S vs E** for FRET stoichiometry vs. efficiency). It will:

        - Parse European decimal commas and tab/space separations
        - Display the 2D heatmap (interactive)
        - Compute **marginals** (sum over rows/cols)
        - Extract a **ridge** (per-column maxima) and fit a line or polynomial
        - Optionally fit **1- or 2-exponential** curves to any chosen row/column trace
        - Export results and figures
        """
    )

def parse_origin_like_dat(file_bytes: bytes):
    """Parse an Origin-like .dat correlogram with EU decimal commas & tab separation.
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
            # If any non-numeric sneaks in, skip line
            continue
        rows.append(row)
        max_len = max(max_len, len(row))

    # Pad ragged rows with NaN if needed
    rows = [r + [math.nan] * (max_len - len(r)) for r in rows] if rows else []
    df = pd.DataFrame(rows)
    return df, metadata

uploaded = st.file_uploader("Upload `.dat` correlogram file", type=["dat", "txt", "csv"])

default_sigma = 0.8
default_poly_order = 2
default_smooth = 1.0

left, right = st.columns([0.4, 0.6])

if uploaded is not None:
    with st.status("Parsing file...", expanded=False):
        df, meta = parse_origin_like_dat(uploaded.getvalue())
        st.write("Detected matrix shape:", df.shape)
        if meta:
            st.write("Metadata lines found (first 5 shown):")
            st.code("\n".join(meta[:5]) or "(none)")

    st.divider()

    with left:
        st.subheader("Heatmap controls")
        zmin = st.number_input("zmin (optional, 0 = auto)", value=0.0)
        zmax = st.number_input("zmax (optional, 0 = auto)", value=0.0)
        smoothing = st.slider("Gaussian smoothing (σ)", min_value=0.0, max_value=5.0, value=default_smooth, step=0.1)

        df_np = df.to_numpy(dtype=float)
        if smoothing > 0:
            df_plot = gaussian_filter1d(df_np, sigma=smoothing, axis=0)
            df_plot = gaussian_filter1d(df_plot, sigma=smoothing, axis=1)
        else:
            df_plot = df_np.copy()

        zmin_use = None if zmin <= 0 else zmin
        zmax_use = None if zmax <= 0 else zmax

    with right:
        st.subheader("Correlogram")
        fig = px.imshow(
            df_plot,
            origin="lower",
            aspect="auto",
            color_continuous_scale="Viridis",
            zmin=zmin_use,
            zmax=zmax_use,
            labels=dict(x="E bins (columns)", y="S bins (rows)", color="Counts"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Marginals & Ridge Extraction")

    col1, col2, col3 = st.columns(3)
    with col1:
        col_marg = np.nansum(df_np, axis=0)
        st.markdown("**Column marginal** (sum over rows)")
        fig_col = go.Figure()
        fig_col.add_trace(go.Scatter(y=col_marg, mode="lines", name="col marginal"))
        st.plotly_chart(fig_col, use_container_width=True)
    with col2:
        row_marg = np.nansum(df_np, axis=1)
        st.markdown("**Row marginal** (sum over columns)")
        fig_row = go.Figure()
        fig_row.add_trace(go.Scatter(y=row_marg, mode="lines", name="row marginal"))
        st.plotly_chart(fig_row, use_container_width=True)
    with col3:
        st.markdown("**Ridge (per-column maxima)**")
        # For each column, take the row index of the maximum
        ridge_y = np.nanargmax(df_np, axis=0)
        ridge_x = np.arange(df_np.shape[1])
        fig_ridge = go.Figure()
        fig_ridge.add_trace(go.Scatter(x=ridge_x, y=ridge_y, mode="markers+lines", name="ridge"))
        st.plotly_chart(fig_ridge, use_container_width=True)

    st.info("Tip: If your S/E axes are known (bin edges), you can supply them below to map indices → physical units.")

    with st.expander("Optional: Provide bin edges for S (rows) and E (cols)"):
        s_edges_str = st.text_area("S (rows) bin edges (comma or space-separated)", value="")
        e_edges_str = st.text_area("E (cols) bin edges (comma or space-separated)", value="")

        def parse_edges(s):
            s = s.strip().replace(",", " ")
            if not s:
                return None
            try:
                return np.array([float(x) for x in re.split(r"\s+", s) if x])
            except:
                return None

        s_edges = parse_edges(s_edges_str)
        e_edges = parse_edges(e_edges_str)

        if s_edges is not None and e_edges is not None:
            if len(s_edges) == df_np.shape[0] + 1 and len(e_edges) == df_np.shape[1] + 1:
                st.success("Using provided bin edges to compute physical coordinates.")
                # For ridge, convert indices to bin centers
                s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
                e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])
                ridge_y_phys = s_centers[ridge_y]
                ridge_x_phys = e_centers
            else:
                st.warning("Edge lengths do not match matrix shape (need rows+1 and cols+1). Using indices instead.")
                ridge_y_phys, ridge_x_phys = ridge_y, ridge_x
        else:
            ridge_y_phys, ridge_x_phys = ridge_y, ridge_x

    st.subheader("Ridge Fitting")
    fit_kind = st.selectbox("Fit model", ["Line: y = m*x + b", "Polynomial (order 2)", "Polynomial (order 3)"])

    # Handle NaNs
    mask = ~(np.isnan(ridge_x_phys) | np.isnan(ridge_y_phys))
    X = np.asarray(ridge_x_phys)[mask]
    Y = np.asarray(ridge_y_phys)[mask]

    if len(X) > 3:
        if fit_kind == "Line: y = m*x + b":
            A = np.vstack([X, np.ones_like(X)]).T
            m, b = np.linalg.lstsq(A, Y, rcond=None)[0]
            Y_fit = m * X + b
            coeffs = dict(model="line", m=float(m), b=float(b))
        else:
            order = 2 if "order 2" in fit_kind else 3
            p = np.polyfit(X, Y, order)
            Y_fit = np.polyval(p, X)
            coeffs = dict(model=f"poly{order}", coeffs=[float(c) for c in p.tolist()])

        st.write("Fit coefficients:", coeffs)
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=X, y=Y, mode="markers", name="ridge points"))
        fig_fit.add_trace(go.Scatter(x=X, y=Y_fit, mode="lines", name="fit"))
        st.plotly_chart(fig_fit, use_container_width=True)
    else:
        st.warning("Not enough ridge points to fit.")

    st.divider()
    st.subheader("Trace extraction & decay fitting (optional)")

    st.markdown("Pick a column (E fixed) or row (S fixed), then fit 1- or 2-exponential to its values.")

    trace_mode = st.radio("Trace along", ["Column (vary S, fix E)", "Row (vary E, fix S)"], horizontal=True)
    if trace_mode.startswith("Column"):
        idx = st.number_input("Column index", min_value=0, max_value=df_np.shape[1]-1, value=min(10, df_np.shape[1]-1), step=1)
        x = np.arange(df_np.shape[0])
        y = df_np[:, int(idx)]
        x_label = "S index"
    else:
        idx = st.number_input("Row index", min_value=0, max_value=df_np.shape[0]-1, value=min(10, df_np.shape[0]-1), step=1)
        x = np.arange(df_np.shape[1])
        y = df_np[int(idx), :]
        x_label = "E index"

    y_sm = gaussian_filter1d(y, sigma=0.8) if np.isfinite(y).any() else y

    fit_model = st.selectbox("Decay model", ["None", "Single exponential: A*exp(-x/tau)+C", "Double exponential: A1*exp(-x/tau1)+A2*exp(-x/tau2)+C"])

    def single_exp(x, A, tau, C):
        return A * np.exp(-x / tau) + C

    def double_exp(x, A1, tau1, A2, tau2, C):
        return A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + C

    fig_trace = go.Figure()
    fig_trace.add_trace(go.Scatter(x=x, y=y, mode="markers", name="raw trace"))
    fig_trace.add_trace(go.Scatter(x=x, y=y_sm, mode="lines", name="smoothed"))
    fit_summary = None

    if fit_model != "None" and np.isfinite(y_sm).sum() >= 5:
        try:
            x_fit = x[np.isfinite(y_sm)]
            y_fit = y_sm[np.isfinite(y_sm)]
            if fit_model.startswith("Single"):
                p0 = [float(np.nanmax(y_fit)), max(1.0, len(x_fit)/5.0), float(np.nanmin(y_fit))]
                popt, pcov = curve_fit(single_exp, x_fit, y_fit, p0=p0, maxfev=10000)
                yhat = single_exp(x, *popt)
                perr = np.sqrt(np.diag(pcov))
                fit_summary = {"model": "single_exp", "params": dict(A=popt[0], tau=popt[1], C=popt[2]), "stderr": dict(A=perr[0], tau=perr[1], C=perr[2])}
            else:
                p0 = [float(np.nanmax(y_fit))*0.6, max(1.0, len(x_fit)/8.0), float(np.nanmax(y_fit))*0.4, max(1.0, len(x_fit)/3.0), float(np.nanmin(y_fit))]
                bounds = (-np.inf, np.inf)
                popt, pcov = curve_fit(double_exp, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
                yhat = double_exp(x, *popt)
                perr = np.sqrt(np.diag(pcov))
                fit_summary = {
                    "model": "double_exp",
                    "params": dict(A1=popt[0], tau1=popt[1], A2=popt[2], tau2=popt[3], C=popt[4]),
                    "stderr": dict(A1=perr[0], tau1=perr[1], A2=perr[2], tau2=perr[3], C=perr[4]),
                }

            fig_trace.add_trace(go.Scatter(x=x, y=yhat, mode="lines", name="fit"))
        except Exception as e:
            st.warning(f"Fit failed: {e}")

    st.plotly_chart(fig_trace, use_container_width=True)
    if fit_summary is not None:
        st.json(fit_summary)

    st.divider()
    st.subheader("Export")

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        if st.button("Prepare ridge CSV"):

            ridge_df = pd.DataFrame({"x": X, "y": Y})
            st.download_button("Download ridge.csv", data=ridge_df.to_csv(index=False).encode("utf-8"), file_name="ridge.csv", mime="text/csv")
        if st.button("Prepare marginals CSV"):
            marg = pd.DataFrame({"col_marg": col_marg, "row_marg": row_marg})
            st.download_button("Download marginals.csv", data=marg.to_csv(index=False).encode("utf-8"), file_name="marginals.csv", mime="text/csv")

    with export_col2:
        # Save the heatmap as static image via Plotly JSON export (users can re-open in Plotly)
        heatmap_json = fig.to_json()
        st.download_button("Download heatmap (plotly.json)", data=heatmap_json.encode("utf-8"), file_name="heatmap.plotly.json", mime="application/json")

else:
    st.info("Upload your `.dat` file to get started.")
