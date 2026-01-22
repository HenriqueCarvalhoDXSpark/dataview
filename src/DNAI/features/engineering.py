import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log2

import prince
import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import masi_distance


def cat_to_numeric(df : pd.DataFrame, cats : list = None) -> pd.DataFrame:

    if not cats:    
        cats = pd.unique(df.values.ravel())

    cats = [c for c in cats if pd.notna(c)]

    cat2id = {c: i for i, c in enumerate(cats)}

    return df.map(lambda x: cat2id.get(x, np.nan)).astype(float)

def get_fitted(df, label1, label2):

    df_clean = df[[label1, label2]].dropna()
    x = df_clean[label1].values.reshape(-1,1)
    y = df_clean[label2].values.reshape(-1,1)

    model = LinearRegression()
    model.fit(x,y)

    r2 = model.score(x,y)

    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)

    return x_range, y_pred, r2

def get_coders(df, supervisor_level='supervisor', rater_level='rater'):

    col_levels = df.columns.names
    
    # Case 1: no supervisor level (2-level MI)
    if supervisor_level not in col_levels:
        # only rater + label
        sup_idx = None
        rater_idx = col_levels.index(rater_level)
        raters = df.columns.get_level_values(rater_idx).unique()
        
        coders = [(None, r) for r in raters]
    
    # Case 2: supervisor + rater + label
    else:
        sup_idx   = col_levels.index(supervisor_level)
        rater_idx = col_levels.index(rater_level)

        # All unique (supervisor, rater) combinations
        coders = sorted(
            set((col[sup_idx], col[rater_idx]) for col in df.columns)
        )

    return col_levels, coders, rater_idx, sup_idx


def build_masi_annotation_data(df, supervisor_level='supervisor', rater_level='rater'):
    """
    df: DataFrame with MultiIndex columns of structure:
         (supervisor, rater, label)   OR
         (rater, label)

    Returns list of (coder_id, unit_id, frozenset(labels)).
    Each unique (supervisor, rater) pair is treated as 1 coder.
    """
    
    col_levels, coders, rater_idx, sup_idx = get_coders(df, supervisor_level, rater_level)

    data = []

    for film_id, row in df.iterrows():
        for supervisor, rater in coders:
            
            # build coder id (string)
            if supervisor is None:
                coder_id = str(rater)
            else:
                coder_id = f"{supervisor}_{rater}"
            
            # select all columns belonging to this coder
            if supervisor is None:
                cols_for_coder = [
                    col for col in df.columns 
                    if col[col_levels.index(rater_level)] == rater
                ]
            else:
                cols_for_coder = [
                    col for col in df.columns 
                    if col[sup_idx] == supervisor and col[rater_idx] == rater
                ]
            
            labels_raw = [row[c] for c in cols_for_coder]
            labels = [lab for lab in labels_raw if pd.notna(lab)]

            if labels:
                data.append((
                    coder_id,
                    str(film_id),
                    frozenset(str(l) for l in labels)
                ))

    return data

def krippendorff_alpha_masi(df):

    data = build_masi_annotation_data(df)

    task = AnnotationTask(data=data, distance=masi_distance)
    return task.alpha()

def entropy(counts):
    """Shannon entropy for a list of category counts."""
    total = sum(counts)
    if total == 0:
        return np.nan
    probs = [c / total for c in counts if c > 0]
    return -sum(p * log2(p) for p in probs)

def per_film_entropy(df):
    rows = []
    
    for film_id, row in df.iterrows():
        # remove NaN (films with missing raters)
        categories = row.dropna().tolist()
        
        # category frequency
        counts = pd.Series(categories).value_counts().tolist()
        
        H = entropy(counts)
        
        rows.append({
            "film_id": film_id,
            "n_raters": len(categories),
            "unique_categories": len(set(categories)),
            "entropy": H
        })
    
    return pd.DataFrame(rows)


def get_binary_matrix(df, cats):

    def get_presence_vector(row):

        obs = set(row.dropna().values)

        return np.array([1 if c in obs else 0 for c in cats], dtype=int)

    b = df.apply(lambda row : get_presence_vector(row), axis = 1)
    b = b.apply(pd.Series)
    b.columns = cats

    return b

def get_mca(df_categories, dimension, df_titles, RESULTS_FIGURES_DIR):

    N_COMPONENTS = 10
    DIM_X, DIM_Y = 0, 1

    # Clustering configuration
    FILM_N_CLUSTERS = 5
    CAT_N_CLUSTERS  = 6
    CLUSTER_METHOD  = "kmeans"  # "kmeans" or "agglomerative"

    # Title column name in df_titles (change if needed)
    TITLE_COL = "title"

    # Output
    HTML_OUT = f"{RESULTS_FIGURES_DIR}/mca_{dimension}.html"
    FIG_OUT = f"{RESULTS_FIGURES_DIR}/mca_{dimension}.png"

    # -------------------------
    # 1) Minimal cleaning for MCA
    # -------------------------
    # Ensure index is film_id
    _df_categories = df_categories.copy()
    
    # drop all-zero categories (important for MCA stability)
    df_cat = _df_categories.loc[:, _df_categories.sum(axis=0) > 0].astype(float)

    # drop all-zero rows (should not happen, but safe)
    df_cat = df_cat.loc[df_cat.sum(axis=1) > 0].copy()

    # Align titles with remaining films
    #df_titles = df_titles.copy()
    common = df_cat.index.intersection(df_titles.index)
    df_titles = df_titles.loc[common]
    df_cat = df_cat.loc[common]

    zero_cols = df_cat.columns[df_cat.sum(axis=0) == 0]
    df_cat = df_cat.drop(columns=zero_cols).astype(float)
    # -------------------------
    # 2) Fit MCA (one_hot=False because df_cat is already indicator-coded)
    # -------------------------
    mca = prince.MCA(
        n_components=N_COMPONENTS,
        n_iter=20,
        copy=True,
        check_input=True,
        engine="sklearn",
        random_state=42,
        one_hot=False
    ).fit(df_cat)

    film_coords = mca.row_coordinates(df_cat)          # index = film_id, columns = [0..]
    cat_coords  = mca.column_coordinates(df_cat)       # index = category, columns = [0..]
    cat_contrib = mca.column_contributions_            # contributions per dimension

    eigvals = mca.eigenvalues_
    expl_var = eigvals / eigvals.sum()
    dim1_pct = round(expl_var[0] * 100, 1)
    dim2_pct = round(expl_var[1] * 100, 1)


    # -------------------------
    # 3) Cluster films and categories in MCA space
    # -------------------------
    def cluster_points(X: pd.DataFrame, n_clusters: int, method: str = "kmeans", random_state: int = 42):
        """
        Cluster points using coordinates (recommended: first few MCA dims).
        Returns labels as a pd.Series aligned to X.index.
        """
        # Use first 5 dims for clustering (typical). Adjust if you prefer.
        Z = X.iloc[:, :min(5, X.shape[1])].copy()

        # Standardize for clustering stability (particularly if dims have different scales)
        Zs = StandardScaler().fit_transform(Z.values)

        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
            labels = model.fit_predict(Zs)
        elif method == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            labels = model.fit_predict(Zs)
        else:
            raise ValueError("method must be 'kmeans' or 'agglomerative'")

        return pd.Series(labels, index=X.index, name="cluster")

    film_cluster = cluster_points(film_coords, FILM_N_CLUSTERS, method=CLUSTER_METHOD)
    cat_cluster  = cluster_points(cat_coords,  CAT_N_CLUSTERS,  method=CLUSTER_METHOD)


    # -------------------------
    # 4) Prepare dataframes for plotting
    # -------------------------
    films_plot = film_coords[[DIM_X, DIM_Y]].copy()
    films_plot.columns = ["Dim1", "Dim2"]
    films_plot["film_id"] = films_plot.index.astype(str)
    films_plot = films_plot.join(df_titles[[TITLE_COL]].rename(columns={TITLE_COL: "title"}), how="left")
    films_plot["film_cluster"] = film_cluster.astype(int).values

    cats_plot = cat_coords[[DIM_X, DIM_Y]].copy()
    cats_plot.columns = ["Dim1", "Dim2"]
    cats_plot["category"] = cats_plot.index.astype(str)
    cats_plot["cat_cluster"] = cat_cluster.astype(int).values

    # Contribution filter for readable category labeling (optional but recommended)
    # Keep categories that contribute above-average to Dim1+Dim2
    contrib_2d = (cat_contrib.iloc[:, DIM_X] + cat_contrib.iloc[:, DIM_Y]).copy()
    contrib_2d.name = "contrib_2d"
    cats_plot = cats_plot.join(contrib_2d, how="left")
    cats_plot["label_me"] = cats_plot["contrib_2d"] > cats_plot["contrib_2d"].mean()


    # -------------------------
    # 5) Static Matplotlib figure with subplots (films + categories)
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True)

    # Films subplot
    ax = axes[0]
    for cl in sorted(films_plot["film_cluster"].unique()):
        sub = films_plot[films_plot["film_cluster"] == cl]
        ax.scatter(sub["Dim1"], sub["Dim2"], alpha=0.7, label=f"Cluster {cl}")
    ax.axhline(0, color = 'k'); ax.axvline(0, color = 'k')
    ax.set_title("Films")
    ax.set_xlabel(f"Dim1 ({dim1_pct:.1f}%)")
    ax.set_ylabel(f"Dim2 ({dim2_pct:.1f}%)")
    ax.legend(loc="best", fontsize=8)

    # Categories subplot
    ax = axes[1]
    for cl in sorted(cats_plot["cat_cluster"].unique()):
        sub = cats_plot[cats_plot["cat_cluster"] == cl]
        ax.scatter(sub["Dim1"], sub["Dim2"], alpha=0.8, label=f"Cluster {cl}")
    # Label only high-contribution categories for readability
    for _, r in cats_plot[cats_plot["label_me"]].iterrows():
        ax.text(r["Dim1"], r["Dim2"], r["category"], fontsize=8)
    ax.axhline(0, color = 'k'); ax.axvline(0, color = 'k')
    ax.set_title("Categories")
    ax.set_xlabel(f"Dim1 ({dim1_pct:.1f}%)")
    ax.set_ylabel(f"Dim2 ({dim2_pct:.1f}%)")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(dimension)    
    fig.savefig(FIG_OUT, dpi=300)

    plt.show()


    # -------------------------
    # 6) Interactive Plotly with box/lasso selection + HTML export
    # -------------------------

    def build_mca_overlay_figure(
        films_plot,
        cats_plot,
        dim1_pct=None,
        dim2_pct=None,
        show_cat_labels=True,
        html_out="mca_overlay.html"
    ):
        fig = go.Figure()

        # --- Add FILM cluster traces (one trace per cluster) ---
        film_clusters = sorted(films_plot["film_cluster"].unique())
        film_trace_idxs = []
        for cl in film_clusters:
            sub = films_plot[films_plot["film_cluster"] == cl]
            fig.add_trace(
                go.Scatter(
                    x=sub["Dim1"],
                    y=sub["Dim2"],
                    mode="markers",
                    marker=dict(symbol="circle",size=7),
                    name=f"Films • Cluster {cl}",
                    legendgroup="films",
                    showlegend=True,
                    customdata=np.stack([sub["film_id"].astype(str), sub["title"].astype(str)], axis=1),
                    hovertemplate=(
                        "film_id=%{customdata[0]}<br>"
                        "title=%{customdata[1]}<br>"
                        "Dim1=%{x:.4f}<br>"
                        "Dim2=%{y:.4f}"
                        "<extra></extra>"
                    ),
                )
            )
            film_trace_idxs.append(len(fig.data) - 1)

        # --- Add CATEGORY cluster traces (one trace per cluster) ---
        cat_clusters = sorted(cats_plot["cat_cluster"].unique())
        cat_trace_idxs = []
        for cl in cat_clusters:
            sub = cats_plot[cats_plot["cat_cluster"] == cl]

            # markers for categories
            fig.add_trace(
                go.Scatter(
                    x=sub["Dim1"],
                    y=sub["Dim2"],
                    mode="markers",
                    marker=dict(symbol="diamond", size=12),
                    name=f"Categories • Cluster {cl}",
                    legendgroup="cats",
                    showlegend=True,
                    customdata=np.stack([sub["category"].astype(str)], axis=1),
                    hovertemplate=(
                        "category=%{customdata[0]}<br>"
                        "Dim1=%{x:.4f}<br>"
                        "Dim2=%{y:.4f}"
                        "<extra></extra>"
                    ),
                )
            )
            cat_trace_idxs.append(len(fig.data) - 1)

            # optional text labels (only for label_me categories)
            if show_cat_labels and "label_me" in sub.columns:
                sublab = sub[sub["label_me"]].copy()
                if len(sublab) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=sublab["Dim1"],
                            y=sublab["Dim2"],
                            mode="text",
                            text=sublab["category"],
                            textposition="top center",
                            name=f"Category labels • Cluster {cl}",
                            legendgroup="cats_labels",
                            showlegend=False,      # keep legend clean
                            hoverinfo="skip",
                            visible=True
                        )
                    )
                    # NOTE: we do not include these in dropdown toggles below by default; we will.
                    # So we will track them as category-related traces too:
                    cat_trace_idxs.append(len(fig.data) - 1)

        # --- Dropdown visibility masks ---
        n = len(fig.data)

        def vis_all_false():
            return [False] * n

        # Films only: show film traces, hide all cat traces (including labels)
        vis_films = vis_all_false()
        for idx in film_trace_idxs:
            vis_films[idx] = True

        # Categories only: show cat traces, hide film traces
        vis_cats = vis_all_false()
        for idx in cat_trace_idxs:
            vis_cats[idx] = True

        # Both
        vis_both = [True] * n

        # --- Layout: dropdown + axes labels + selection mode ---
        xlab = "Dim1" if dim1_pct is None else f"Dim1 ({dim1_pct:.1f}%)"
        ylab = "Dim2" if dim2_pct is None else f"Dim2 ({dim2_pct:.1f}%)"

        fig.update_layout(
            title=f"MCA – {dimension}",
            dragmode="zoom",
            xaxis=dict(title=xlab, zeroline=True),
            yaxis=dict(title=ylab, zeroline=True),
            legend_title_text="Traces (click to toggle clusters)",
            legend=dict(groupclick="toggleitem"),
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.91,
                    y=1.12,
                    xanchor="left",
                    yanchor="top",
                    showactive=True,
                    buttons=[
                        dict(label="Show: Both", method="update", args=[{"visible": vis_both}]),
                        dict(label="Show: Films only", method="update", args=[{"visible": vis_films}]),
                        dict(label="Show: Categories only", method="update", args=[{"visible": vis_cats}]),
                    ],
                )
            ],
            margin=dict(t=120, l=60, r=40, b=50),
            height=750,
        )

        # Default state: Both visible
        fig.update_traces(visible=True)

        # Save HTML
        fig.write_html(html_out, include_plotlyjs="cdn")
        return fig


    # Usage:
    build_mca_overlay_figure(
        films_plot=films_plot,
        cats_plot=cats_plot,
        dim1_pct=dim1_pct,
        dim2_pct=dim2_pct,
        show_cat_labels=True,
        html_out=HTML_OUT
    )

from scipy.stats import chi2_contingency

def perm_chi2_pvalue(y_binary: pd.Series,
                     cluster_labels: pd.Series,
                     n_perm: int = 5000,
                     seed: int = 42):
    """
    Permutation p-value for association between a binary label y and clusters.
    Works under sparse counts.
    """
    rng = np.random.default_rng(seed)

    # contingency table: rows = cluster, cols = {0,1}
    ct = pd.crosstab(cluster_labels, y_binary)

    # Ensure both columns exist (0 and 1)
    if 0 not in ct.columns: ct[0] = 0
    if 1 not in ct.columns: ct[1] = 0
    ct = ct[[0, 1]]

    # If degenerate (all 0s or all 1s), no test possible
    if ct[1].sum() == 0 or ct[0].sum() == 0:
        return np.nan, np.nan

    # Observed chi-square statistic (no Yates correction; multi-cluster)
    chi2_obs, _, _, _ = chi2_contingency(ct.values, correction=False)

    # Permutation distribution
    chi2_perm = np.empty(n_perm, dtype=float)
    labels = cluster_labels.to_numpy()

    for i in range(n_perm):
        perm = rng.permutation(labels)
        ct_perm = pd.crosstab(perm, y_binary).reindex(ct.index, fill_value=0)

        if 0 not in ct_perm.columns: ct_perm[0] = 0
        if 1 not in ct_perm.columns: ct_perm[1] = 0
        ct_perm = ct_perm[[0, 1]]

        chi2_perm[i], _, _, _ = chi2_contingency(ct_perm.values, correction=False)

    # p-value = proportion permuted >= observed
    p = (np.sum(chi2_perm >= chi2_obs) + 1) / (n_perm + 1)
    return chi2_obs, p

from sklearn.linear_model import LogisticRegression

def fit_ridge_logit_stat(X, y, C=1.0):
    """
    Fit ridge logistic regression and return:
      - coefficients (shape: [n_features])
      - omnibus statistic: max absolute coefficient
    """
    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=1000
    )
    model.fit(X, y)
    coefs = model.coef_.ravel()
    stat = np.max(np.abs(coefs))
    return coefs, stat

def perm_omnibus_pvalue(X, y, n_perm=2000, seed=42, C=1.0):
    """
    Permutation p-value for association between factors X and binary outcome y.
    Uses max(|coef|) as statistic. Returns (coefs_obs, stat_obs, pvalue).
    """
    rng = np.random.default_rng(seed)

    coefs_obs, stat_obs = fit_ridge_logit_stat(X, y, C=C)

    perm_stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        _, perm_stats[i] = fit_ridge_logit_stat(X, y_perm, C=C)

    p = (np.sum(perm_stats >= stat_obs) + 1) / (n_perm + 1)
    return coefs_obs, stat_obs, p