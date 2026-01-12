import numpy as np
import pandas as pd
from math import log2
from itertools import combinations
import torch

from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import masi_distance

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

import spacy
nlp = spacy.load("pt_core_news_lg")

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import umap


def cat_to_numeric(df, cats = None):

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

def get_similarity_mapping(df):

    cats_1, cats_2 = df.iloc[0].values, df.iloc[1].valus

    # cats_list = list(cats_1)
    # other_list = list(cats_2)

    # # give labels some context to help the embedding model
    # def add_context(label: str) -> str:
    #     return f"Tópico de filme: {label}"

    # cats_with_ctx = [add_context(x) for x in cats_list]
    # other_with_ctx = [add_context(x) for x in other_list]

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # Encode labels (normalized for cosine via dot product)
    emb_cats = model.encode(cats_1, normalize_embeddings=True)
    emb_other = model.encode(cats_2, normalize_embeddings=True)

    # Cosine similarity matrix: shape (len(other) x len(all))
    sim_matrix = emb_other @ emb_cats.T   # because embeddings are normalized

    # For each "other" label, find best match
    best_cat_idx = sim_matrix.argmax(axis=1)
    best_sim = sim_matrix.max(axis=1)

    # Build mapping dict (optionally apply a similarity threshold)
    threshold = 0.50  # tune this
    mapping = {}
    for i, other_label in enumerate(cats_2):
        sim = best_sim[i]
        gold_label = cats_1[best_cat_idx[i]]
        if sim >= threshold:
            mapping[other_label] = (gold_label, float(sim))
        else:
            mapping[other_label] = (None, float(sim))  # no good match

    df_map = pd.DataFrame(
        {
            "cats_2": cats_2,
            "cats_1": [mapping[o][0] for o in cats_2],
            "similarity": [mapping[o][1] for o in cats_2],
        }
    ).sort_values("similarity", ascending=False)

    final_mapping = {k: v for k, (v, s) in mapping.items() if v is not None}

    return df_map, final_mapping


def get_topic_visualizations(df):
        
    topics_per_film = df.apply(
        lambda row: set(row.dropna().values),
        axis=1
    )

    all_topics = sorted({t for s in topics_per_film for t in s})

    # Build binary matrix
    binary_df = pd.DataFrame(
        [
            [1 if t in s else 0 for t in all_topics]
            for s in topics_per_film
        ],
        index=topics_per_film.index,
        columns=all_topics
    )

    topic_matrix = binary_df.T

    sns.clustermap(
    topic_matrix,        # topic × film binary matrix
    metric="jaccard",    # distance
    method="average",    # linkage
    figsize=(12, 10),
    cmap="Blues")       # or "Blues", etc.)

    plt.show()

    # topic_matrix: topics × films
    dist = pdist(topic_matrix.values, metric="jaccard")
    Z = linkage(dist, method="average")

    clusters = fcluster(Z, t=10, criterion='maxclust')

    cluster_df = pd.DataFrame({
        "topic": topic_matrix.index,
        "cluster": clusters
    }).sort_values("cluster")

    print(cluster_df)


    X = topic_matrix.values

    reducer = umap.UMAP(
        n_neighbors=10,      # you can play with this
        min_dist=0.1,        # how tight the clusters are
        n_components=2,
        metric="jaccard",
        random_state=42
    )

    embedding = reducer.fit_transform(X)   # shape: (n_topics, 2)

    plt.figure(figsize=(10, 8))

    # Create a colormap
    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters)

    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=clusters,
        cmap="tab20",   # or any other, e.g. "tab10"
        s=40
    )

    for i, topic in enumerate(topic_matrix.index):
        plt.text(
            embedding[i, 0],
            embedding[i, 1],
            topic,
            fontsize=7,
            ha='center',
            va='center'
        )

    plt.title("UMAP of Topics Colored by Cluster")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    cbar = plt.colorbar(scatter, ticks=unique_clusters)
    cbar.set_label("Cluster")
    plt.tight_layout()
    plt.show()
    



def extract_levels(df, n_levels):

    pattern = r'category_(\d+)_'

    levels = df.columns.str.extract(pattern)[0].astype(float)

    return df.loc[:, levels.between(1, n_levels).values]


def krippendorff_alpha_weighted_level(df, n_levels, weights, missing_value=np.nan):
    """
    Compute a weighted Krippendorff's alpha for multi-level categorical ratings.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows = units (e.g. films)
        Columns = flattened [r1_l1, r1_l2, ..., r1_lK, r2_l1, ..., r2_lK, ...]
        All raters must have the same number of levels per unit.
    n_level : number of levels to examine.
    weights : list or 1D array of float
        Level weights (length = number of levels per rater), e.g. [0.4, 0.3, 0.1].
        Higher weight = disagreement at that level counts more.
    missing_value : value treated as missing (default: np.nan)

    Returns
    -------
    alpha : float
        Weighted Krippendorff's alpha.
    """

    if len(weights) != n_levels:
        raise ValueError(f'Number of levels {n_levels} is different from number of weights {len(weights)}')

    data = df.copy()

    data = extract_levels(data, n_levels)

    # Basic dimensions
    n_cols = data.shape[1]

    if n_cols % n_levels != 0:
        raise ValueError(
            f"Number of columns ({n_cols}) is not a multiple of number of levels ({n_levels}). "
            "Make sure columns follow [r1_l1..lK, r2_l1..lK, ...]."
        )

    n_raters = n_cols // n_levels
    weights = np.asarray(weights, dtype=float)

    # Helper: get rating of (rater j, level ℓ) in a given row
    def get_cell(row, rater_idx, level_idx):
        col_idx = rater_idx * n_levels + level_idx
        return row.iloc[col_idx]

    # ---------- Observed disagreement (Do) ----------
    Do_sum = 0.0
    pair_count = 0

    for _, row in data.iterrows():
        # For each pair of raters
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                dist_ij = 0.0
                valid_levels = 0
                for ℓ in range(n_levels):
                    v1 = get_cell(row, i, ℓ)
                    v2 = get_cell(row, j, ℓ)

                    if (pd.isna(v1) or pd.isna(v2) or
                        v1 == missing_value or v2 == missing_value):
                        continue

                    # nominal disagreement (1 if different, 0 if same), weighted by level
                    dist_ij += weights[ℓ] * (v1 != v2)
                    valid_levels += 1

                # Only count this pair if at least one level was valid
                if valid_levels > 0:
                    Do_sum += dist_ij
                    pair_count += 1

    if pair_count == 0:
        return np.nan  # no comparable pairs

    Do = Do_sum / pair_count

    # ---------- Expected disagreement (Pe) ----------
    Pe = 0.0

    for ℓ in range(n_levels):
        # Take all ratings at level ℓ for all raters, all items
        level_cols = data.iloc[:, ℓ::n_levels].values.flatten()
        level_vals = pd.Series(level_cols)
        level_vals = level_vals[
            (~level_vals.isna()) & (level_vals != missing_value)
        ]

        if level_vals.empty:
            continue

        # Category probabilities at this level
        p = level_vals.value_counts(normalize=True)

        # For nominal distance: E[d] = wℓ * (1 - sum_c p(c)^2)
        Pe_ℓ = weights[ℓ] * (1.0 - np.sum(p.values ** 2))
        Pe += Pe_ℓ

    if Pe == 0:
        # No expected disagreement (e.g. only one category) → alpha undefined or 1
        return np.nan

    alpha = 1.0 - Do / Pe
    #return float(np.round(alpha, decimals=2))
    return alpha



def weight_generator(n_levels, kind, **kwargs):

    def linear(n_levels):
        """
        Linear decay weights for any number of levels.
        Highest weight = level 1.
        """
        raw = np.arange(n_levels, 0, -1)  # e.g. [5,4,3,2,1]
        weights = raw / raw.sum()
        return weights

    def geometric(n_levels, geometric_ratio=0.7):

        """
        Geometric decay weights for any number of levels.
        ratio < 1 produces smooth decay.
        Example ratio values: 0.5 (steeper), 0.7 (smooth), 0.9 (gentle).
        """
        raw = np.array([geometric_ratio**k for k in range(n_levels)])
        weights = raw / raw.sum()
        return weights
    
    def softmax(n_levels, softmax_temperature=1.0):
        """
        Softmax-based decay for any number of levels.
        temperature < 1 makes steeper decay.
        temperature > 1 makes decay gentler.
        """
        scores = np.arange(n_levels, 0, -1)  # e.g. [5,4,3,2,1]
        exp_scores = np.exp(scores / softmax_temperature)
        weights = exp_scores / exp_scores.sum()
        return weights

    functions = {'linear' : linear,
                 'geometric': geometric,
                 'softmax' : softmax
                 }

    if kind not in functions:
        raise ValueError(f'Unknown option {kind}')
    
    return functions[kind](n_levels, **kwargs)

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

def weighted_jaccard_distance_level(a_levels, b_levels, level_weights):
    """
    Distance between two multi-level annotations using OR logic + level weights.
    
    a_levels, b_levels: list-like of categories, ordered by level
                        e.g. ["Drama", "Politics", "War"]
    level_weights: list of floats, same length as number of levels,
                   e.g. [0.4, 0.3, 0.1]
                   
    Returns a distance in [0, 1], where:
      0 = perfect agreement
      1 = complete disagreement
    """
    # Convert to dict: category -> total weight
    w_a = {}
    w_b = {}
    
    # Rater A
    for idx, cat in enumerate(a_levels):
        if cat is None or (isinstance(cat, float) and np.isnan(cat)):
            continue
        if idx >= len(level_weights):
            continue
        w = level_weights[idx]
        w_a[cat] = w_a.get(cat, 0.0) + w

    # Rater B
    for idx, cat in enumerate(b_levels):
        if cat is None or (isinstance(cat, float) and np.isnan(cat)):
            continue
        if idx >= len(level_weights):
            continue
        w = level_weights[idx]
        w_b[cat] = w_b.get(cat, 0.0) + w

    # If both are effectively empty → perfect match
    if not w_a and not w_b:
        return 0.0

    # Weighted Jaccard
    all_cats = set(w_a.keys()).union(w_b.keys())
    inter_w = 0.0
    union_w = 0.0

    for c in all_cats:
        wa = w_a.get(c, 0.0)
        wb = w_b.get(c, 0.0)
        inter_w += min(wa, wb)
        union_w += max(wa, wb)

    if union_w == 0:
        return 0.0

    jacc_w = inter_w / union_w
    dist = 1.0 - jacc_w
    return dist

def krippendorff_alpha_or_levelweighted(df, level_weights):
    """
    Krippendorff-style alpha for multi-level, multi-label annotations with:
      - OR logic (any overlapping category counts as agreement)
      - level weights (primary > secondary > tertiary ...)
    
    Parameters
    ----------
    df : pandas.DataFrame
        Rows = items (e.g. films)
        Columns = raters
        Each cell must be a list/tuple of categories ordered by level,
        e.g. ["Drama", "Politics", "War"].
    
    level_weights : list of floats
        Level importance weights, e.g. [0.4, 0.3, 0.1].
        Length should be >= max number of levels used.
    
    Returns
    -------
    alpha : float
        Reliability coefficient in [-1, 1], or np.nan if undefined.
    """
    n_items, n_raters = df.shape

    # --------------------------
    # Observed disagreement D_o
    # --------------------------
    Do_sum = 0.0
    n_pairs = 0

    for _, row in df.iterrows():
        for r1, r2 in combinations(range(n_raters), 2):
            a = row.iloc[r1]
            b = row.iloc[r2]

            # skip missing annotations
            if a is None or b is None:
                continue
            if (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
                continue

            d = weighted_jaccard_distance_level(a, b, level_weights)
            Do_sum += d
            n_pairs += 1

    if n_pairs == 0:
        return np.nan

    Do = Do_sum / n_pairs

    # --------------------------
    # Expected disagreement D_e
    # --------------------------
    # Pool all annotations across items and raters
    all_ann = []
    for _, row in df.iterrows():
        for r in range(n_raters):
            a = row.iloc[r]
            if a is None:
                continue
            if isinstance(a, float) and np.isnan(a):
                continue
            all_ann.append(a)

    if len(all_ann) < 2:
        return np.nan

    De_sum = 0.0
    De_pairs = 0

    for a, b in combinations(all_ann, 2):
        d = weighted_jaccard_distance_level(a, b, level_weights)
        De_sum += d
        De_pairs += 1

    if De_pairs == 0:
        return np.nan

    De = De_sum / De_pairs

    if De == 0:
        return np.nan  # no expected disagreement → alpha undefined

    alpha = 1.0 - Do / De
    return alpha

def krippendorff_alpha_weighted_rater(df, rater_weights, n_raters = 1, missing_value=np.nan):
    """
    Krippendorff's alpha with rater-importance weighting.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Rows = items (e.g., films)
        Columns = raters. Each cell = assigned category (nominal).
    rater_weights : list or array of floats
        Importance weights per rater. Length must match df.columns.
    missing_value : value treated as missing
    
    Returns
    -------
    alpha : float
        Rater-weighted Krippendorff's alpha.
    """

    data = df.copy()
    n_items = data.shape[0] #, n_raters

    rater_weights = np.asarray(rater_weights, dtype=float)
    if len(rater_weights) != n_raters:
        raise ValueError("Length of rater_weights must match number of raters.")

    # Normalize rater weights (optional but preferred)
    if rater_weights.sum() > 0:
        rater_weights = rater_weights / rater_weights.sum()

    # ---------- OBSERVED DISAGREEMENT Do ----------
    Do_sum = 0.0
    total_pair_weight = 0.0

    for _, row in data.iterrows():
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                v1 = row.iloc[i]
                v2 = row.iloc[j]

                if pd.isna(v1) or pd.isna(v2) or v1 == missing_value or v2 == missing_value:
                    continue

                # pair weight = product of rater importances
                w_ij = rater_weights[i] * rater_weights[j]

                # nominal disagreement (0 if same, 1 if different)
                d_ij = (v1 != v2)

                Do_sum += w_ij * d_ij
                total_pair_weight += w_ij

    if total_pair_weight == 0:
        return np.nan

    Do = Do_sum / total_pair_weight

    # ---------- EXPECTED DISAGREEMENT De ----------
    # Weighted category probabilities across all raters & all items
    cat_weights = {}
    total_weight = 0.0

    for item_idx in range(n_items):
        row = data.iloc[item_idx]
        for r in range(n_raters):
            v = row.iloc[r]
            if pd.isna(v) or v == missing_value:
                continue

            w_r = rater_weights[r]
            cat_weights[v] = cat_weights.get(v, 0.0) + w_r
            total_weight += w_r

    if total_weight == 0:
        return np.nan

    # category probabilities
    probs = np.array([w / total_weight for w in cat_weights.values()])

    # expected nominal disagreement Pe = 1 - sum( p(c)^2 )
    Pe = 1.0 - np.sum(probs ** 2)

    if Pe == 0:
        return np.nan

    # ---------- KRIPPENDORFF ALPHA ----------
    alpha = 1.0 - Do / Pe
    return alpha

def krippendorff_alpha_weighted_level_rater(
    df,
    level_weights,
    rater_weights=None,
    missing_value=np.nan,
):
    """
    Generalized weighted Krippendorff's alpha with:
    - level_weights: importance of levels (e.g. [0.4, 0.3, 0.1])
    - rater_weights: importance of raters (e.g. [1.0, 0.8, 0.5])
    
    Parameters
    ----------
    df : pandas.DataFrame
        Rows = units (e.g., films)
        Columns = flattened by rater & level:
          r1_l1, r1_l2, ..., r1_lK,
          r2_l1, r2_l2, ..., r2_lK,
          ...
        All raters must have the same number of levels.
    level_weights : list or 1D array of float
        Weights for levels (length = number of levels per rater).
    rater_weights : list or 1D array of float, optional
        Weights for raters (length = number of raters).
        If None, all raters are equally weighted.
    missing_value : value treated as missing (default: np.nan)

    Returns
    -------
    alpha : float
        Generalized weighted Krippendorff's alpha.
    """

    data = df.copy()
    n_items, n_cols = data.shape

    level_weights = np.asarray(level_weights, dtype=float)
    n_levels = len(level_weights)

    if n_cols % n_levels != 0:
        raise ValueError(
            f"Number of columns ({n_cols}) is not a multiple of "
            f"number of levels ({n_levels}). "
            "Columns must follow [r1_l1..lK, r2_l1..lK, ...]."
        )

    n_raters = n_cols // n_levels

    if rater_weights is None:
        rater_weights = np.ones(n_raters, dtype=float)
    else:
        rater_weights = np.asarray(rater_weights, dtype=float)
        if len(rater_weights) != n_raters:
            raise ValueError(
                f"Length of rater_weights ({len(rater_weights)}) "
                f"does not match number of raters ({n_raters})."
            )

    # Normalize rater weights (optional but convenient)
    if rater_weights.sum() > 0:
        rater_weights = rater_weights / rater_weights.sum()

    # Helper: fetch rating of (rater j, level ℓ) in a given row (Series)
    def get_cell(row, rater_idx, level_idx):
        col_idx = rater_idx * n_levels + level_idx
        return row.iloc[col_idx]

    # ---------- Observed disagreement (Do), weighted by rater importance ----------
    Do_sum = 0.0
    total_pair_weight = 0.0

    for _, row in data.iterrows():
        # for each pair of raters
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                pair_w = rater_weights[i] * rater_weights[j]

                dist_ij = 0.0
                valid_levels = 0

                for ℓ in range(n_levels):
                    v1 = get_cell(row, i, ℓ)
                    v2 = get_cell(row, j, ℓ)

                    if (
                        pd.isna(v1) or pd.isna(v2) or
                        v1 == missing_value or v2 == missing_value
                    ):
                        continue

                    # nominal disagreement, weighted by level importance
                    dist_ij += level_weights[ℓ] * (v1 != v2)
                    valid_levels += 1

                if valid_levels > 0:
                    Do_sum += pair_w * dist_ij
                    total_pair_weight += pair_w

    if total_pair_weight == 0:
        return np.nan  # no comparable pairs

    Do = Do_sum / total_pair_weight

    # ---------- Expected disagreement (De), using rater-weighted category frequencies ----------
    Pe = 0.0

    for ℓ in range(n_levels):
        # We accumulate category weights: sum over (item, rater) of rater_weight
        # for each observed category at this level.
        category_weight_sum = {}
        total_weight = 0.0

        for item_idx in range(n_items):
            row = data.iloc[item_idx]
            for r in range(n_raters):
                v = get_cell(row, r, ℓ)
                if (
                    pd.isna(v) or
                    v == missing_value
                ):
                    continue

                w_r = rater_weights[r]
                category_weight_sum[v] = category_weight_sum.get(v, 0.0) + w_r
                total_weight += w_r

        if total_weight == 0:
            continue  # no data at this level

        # Weighted probabilities p(c) at this level
        probs = np.array([w / total_weight for w in category_weight_sum.values()])

        # For nominal distance: E[d_level] = w_level * (1 - sum_c p(c)^2)
        Pe_ℓ = level_weights[ℓ] * (1.0 - np.sum(probs ** 2))
        Pe += Pe_ℓ

    if Pe == 0:
        # no expected disagreement (e.g., only one category overall)
        return np.nan

    alpha = 1.0 - Do / Pe
    return alpha



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

from sklearn.linear_model import LinearRegression

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
