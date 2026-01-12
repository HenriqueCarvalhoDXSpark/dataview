import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log2

import prince
import plotly.graph_objects as go
import plotly.express as px

import hdbscan
import torch
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import masi_distance

import spacy
nlp = spacy.load("pt_core_news_lg")

def cat_to_numeric(df : pd.DataFrame, cats : list = None) -> pd.DataFrame:

    if not cats:    
        cats = pd.unique(df.values.ravel())

    cats = [c for c in cats if pd.notna(c)]

    cat2id = {c: i for i, c in enumerate(cats)}

    return df.map(lambda x: cat2id.get(x, np.nan)).astype(float)

def semantic_similarity(row, label_a, label_b, lemma=False):
    
    text1 = row[label_a]
    text2 = row[label_b]

    if lemma:
        text1 = lemmatize_list([text1])
        text2 = lemmatize_list([text2])

    # Encode as tensors
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)

    # Cosine similarity (returns a tensor)
    sim = util.cos_sim(emb1, emb2)

    return sim.item()

def semantic_similarity_intra(row, labels, lemma=False):
    
    #texts = row[labels].fillna('').values
    texts = row[labels].dropna().values

    if lemma:
        texts = lemmatize_list(texts)

    emb = model.encode(texts, convert_to_tensor=True)

    sim_matrix = util.cos_sim(emb, emb)

    mask = ~torch.eye(sim_matrix.size(0), dtype=bool)

    vals = sim_matrix[mask]

    if vals.numel() == 0:
        return np.nan
    
    else:
        return vals.max().item()
    
    
def lemmatize_list(text_list):
    lemmatized = []
    for doc in nlp.pipe(text_list, batch_size=32):   # fast vectorized processing
        lemmas = " ".join([token.lemma_ for token in doc])
        lemmatized.append(lemmas)
    return lemmatized

def semantic_similarity_inter(row, labels_a, labels_b, lemma=False):

    textsA = row.loc[labels_a].fillna('').values
    textsB = row.loc[labels_b].fillna('').values
    
    if lemma:
        textsA = lemmatize_list(textsA)
        textsB = lemmatize_list(textsB)
    
    embA = model.encode(textsA, convert_to_tensor=True)
    embB = model.encode(textsB, convert_to_tensor=True)

    sim_matrix = util.cos_sim(embA, embB)

    return sim_matrix.mean().item()

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

def get_mca(X, dimension, df_titles, n_components = 10):

    mca = prince.MCA(n_components=n_components,
                    n_iter=10,
                    copy=True,
                    check_input=True,
                    engine='sklearn',
                    random_state=42
                    )

    mca.fit(X)

    #X.index = X.index.astype(int)

    # Row coordinates (films)
    X_rows = mca.row_coordinates(X)
    X_rows.columns = [f'Dim{n}' for n in range(1,n_components+1)]
    X_rows['film_id'] = X.index.astype(int)

    df_title = pd.merge(X, df_titles, on='film_id')[['film_id','title']]
    df_title['title'] = df_title['film_id'].astype(str) + ' - ' + df_title['title']
    df_title.set_index('film_id', inplace=True)
    X_rows['title'] = df_title['title']

    # Column coordinates (categories / modalities)
    X_cols = mca.column_coordinates(X)
    X_cols.columns = [f'Dim{n}' for n in range(1,n_components+1)]
    X_cols['modality'] = X_cols.index

    # Optionally, variance explained
    eigvals = mca.eigenvalues_
    expl_var = eigvals / eigvals.sum()
    dim1_pct = round(expl_var[0] * 100, 1)
    dim2_pct = round(expl_var[1] * 100, 1)

    X_for_cluster = X_rows[[f'Dim{n}' for n in range(1,n_components+1)]].values

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,   # tune for your data
        min_samples=None,      # default = min_cluster_size
        metric='euclidean'
    )

    film_labels = clusterer.fit_predict(X_for_cluster)
    X_rows['cluster'] = film_labels

    X_cols_for_cluster = X_cols[[f'Dim{n}' for n in range(1,n_components+1)]].values

    clusterer_cat = hdbscan.HDBSCAN(
        min_cluster_size=2,    # typically smaller, fewer modalities
        metric='euclidean'
    )

    cat_labels = clusterer_cat.fit_predict(X_cols_for_cluster)
    X_cols['cluster'] = cat_labels

    # ---- Build color map for film clusters ----
    film_clusters = X_rows['cluster'].astype(int).unique()
    film_clusters_sorted = sorted(film_clusters)

    # Use a qualitative palette and cycle if needed
    palette = px.colors.qualitative.Safe
    color_map = {cl: palette[i % len(palette)] for i, cl in enumerate(film_clusters_sorted)}

    # Optional: special color for noise (-1)
    if -1 in color_map:
        color_map[-1] = "#272525"  # light grey for noise

    film_colors = X_rows['cluster'].astype(int).map(color_map)

    # ---- Category colors (if you clustered modalities too) ----
    if 'cluster' in X_cols.columns:
        cat_clusters = X_cols['cluster'].astype(int).unique()
        # Reuse the same palette mapping by cluster id
        cat_colors = X_cols['cluster'].astype(int).map(color_map)
    else:
        # If you did not cluster categories, keep them all in one neutral color
        cat_colors = "#FF7F0E"  # orange

    plt.plot(range(1, len(eigvals)+1), expl_var, marker='o')
    plt.xlabel("Dimension")
    plt.ylabel("Explained inertia")
    plt.title("MCA Scree Plot")
    plt.show()

    col_coords = mca.column_coordinates(X)
    col_contrib = (col_coords**2).div(col_coords**2).sum(axis=1)

    # --- Films (individuals) ---
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=X_rows['Dim1'],
            y=X_rows['Dim2'],
            mode='markers',
            name='Films',
            hovertext=X_rows['title'],
            hoverinfo='text+x+y',
            marker=dict(symbol='circle', size=12, opacity=0.8, color=film_colors)
        )
    )

    # --- Modalities (categories) ---
    fig.add_trace(
        go.Scatter(
            x=X_cols['Dim1'],
            y=X_cols['Dim2'],
            mode='markers',
            name='Categories',
            text=X_cols['modality'],
            textposition='top center',
            hovertext=X_cols['modality'],
            hoverinfo='text+x+y',
            marker=dict(symbol='diamond', size=8, opacity=0.6, color = cat_colors)
        )
    )

    fig.update_layout(
        title=f"MCA {dimension}",
        title_x = 0.5,
        xaxis_title=f"Dimension 1 ({dim1_pct}%)",
        yaxis_title=f"Dimension 2 ({dim2_pct}%)",
        template="plotly_white",
        width=900,
        height=700,
        legend=dict(x=0.01, y=0.99)
    )

    return fig
