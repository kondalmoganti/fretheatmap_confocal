
import re, io, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="FRET Analyzer – Joint View", layout="wide")
st.title("FRET Analyzer (Joint Heatmap + Marginals)")

def split_numeric_blocks(text: str):
    text = text.replace(",", ".")
    lines = text.splitlines()
    blocks = []
    cur = []
    num_re = re.compile(r'^\s*[\d\.eE\-\+]+([\s\t,;][\d\.eE\-\+]+)*\s*$')
    def flush():
        if not cur: return
        s = "\n".join(cur).strip()
        try:
            df = pd.read_csv(io.StringIO(s), sep=r"[\s,;]+", engine="python", header=None)
            blocks.append(df)
        except Exception:
            pass
    for ln in lines:
        if num_re.match(ln):
            cur.append(ln)
        else:
            flush(); cur = []
    flush()
    return blocks

uploaded = st.file_uploader("Upload your .dat file", type=["dat","txt","csv"], key="uploader_joint")
if uploaded is None:
    st.info("Upload a file to continue.")
    st.stop()

raw = uploaded.getvalue().decode("utf-8", errors="ignore")
blocks = split_numeric_blocks(raw)

# pick a heatmap-like block (>= 10x10)
mats = [i for i, df in enumerate(blocks) if df.shape[0] >= 10 and df.shape[1] >= 10]
if not mats:
    st.error("No heatmap-like (matrix) block detected in this file.")
    st.stop()

sel = st.selectbox("Choose matrix block for heatmap", mats, format_func=lambda i: f"Block {i} (shape {blocks[i].shape})", key="joint_block")
M = blocks[sel].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()

smooth = st.slider("Gaussian smoothing (σ, 0=off)", 0.0, 6.0, 1.0, 0.1, key="joint_smooth")
if smooth>0:
    Mplot = gaussian_filter1d(gaussian_filter1d(M, sigma=smooth, axis=0), sigma=smooth, axis=1)
else:
    Mplot = M.copy()

# derive marginals *from the heatmap itself* (ensures strict consistency)
e_marginal = np.nansum(Mplot, axis=0)  # sum over rows -> E (x-axis) marginal
s_marginal = np.nansum(Mplot, axis=1)  # sum over cols -> S (y-axis) marginal

# Bin centers as indices (0..n-1); if you know physical edges you can replace here
nx, ny = Mplot.shape[1], Mplot.shape[0]
e_centers = np.arange(nx)
s_centers = np.arange(ny)

# Build a 2x2 layout: top E marginal, left-bottom heatmap, right S marginal (horizontal), top-right empty
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type":"xy"}, {"type":"xy"}],
           [{"type":"heatmap"}, {"type":"xy"}]],
    column_widths=[0.8, 0.2], row_heights=[0.25, 0.75],
    horizontal_spacing=0.02, vertical_spacing=0.02
)

# Top E marginal (row 1, col 1)
fig.add_trace(go.Bar(x=e_centers, y=e_marginal, name="E marginal", opacity=0.6), row=1, col=1)

# Heatmap (row 2, col 1)
fig.add_trace(go.Heatmap(z=Mplot, coloraxis="coloraxis", showscale=True), row=2, col=1)

# Right S marginal (horizontal) (row 2, col 2)
fig.add_trace(go.Bar(y=s_centers, x=s_marginal, orientation="h", name="S marginal", opacity=0.6), row=2, col=2)

# Match axes so bars align with heatmap bins
fig.update_xaxes(matches="x", row=1, col=1)   # top x matches heatmap x
fig.update_xaxes(matches=None, row=2, col=2)  # independent x for right marginal
fig.update_yaxes(matches="y", row=2, col=2)   # right y matches heatmap y

# Labels (indices as bins); you can relabel to physical S/E later
fig.update_xaxes(title_text="E bins (columns)", row=2, col=1)
fig.update_yaxes(title_text="S bins (rows)", row=2, col=1)
fig.update_yaxes(title_text="S bins (rows)", row=2, col=2)
fig.update_xaxes(title_text="Counts (E marginal)", row=1, col=1)

fig.update_layout(coloraxis=dict(colorscale="Viridis"),
                  showlegend=False,
                  bargap=0,
                  margin=dict(l=40, r=10, t=40, b=40))

st.plotly_chart(fig, use_container_width=True)
