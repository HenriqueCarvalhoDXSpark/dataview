import re
import numpy as np
import torch

from nltk.corpus import stopwords

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

import spacy
nlp = spacy.load("pt_core_news_lg")

from sklearn.metrics.pairwise import cosine_distances


pt_stopwords = set(stopwords.words('portuguese'))

def clean_transcript(text):
    # Remove timestamps like \n0:10 or 0:10 or 01:23:45
    text = re.sub(r"\[?\b\d{1,2}:\d{2}(?::\d{2})?\b\]?", " ", text)

    # Remove new lines
    text = text.replace("\n", " ")

    # Remove speaker labels (optional)
    #text = re.sub(r"\b[A-ZÁÀÃÂÉÍÓÕÚÇ]{2,}:", " ", text)
    #text = re.sub(r"\b[A-ZÁÀÃÂÉÍÓÕÚÇ][a-záàãâéíóõúç]+:", " ", text)

    # Lowercase
    text = text.lower()

    return text

# ---------------------------
# 1. LEXICAL RICHNESS: MTLD
# ---------------------------

def mtld(tokens, ttr_threshold=0.72, min_tokens=10):
    """
    Measure of Textual Lexical Diversity (MTLD).
    Fairly robust to text length.
    """
    if len(tokens) < min_tokens:
        return float("nan")

    def _mtld_calc(tok_seq):
        factors = 0
        types = set()
        token_count = 0
        ttr = 1.0

        for tok in tok_seq:
            types.add(tok.lower())
            token_count += 1
            ttr = len(types) / token_count
            if ttr <= ttr_threshold:
                factors += 1
                types = set()
                token_count = 0
        # partial factor
        if token_count > 0:
            factors += (1 - ttr) / (1 - ttr_threshold)
        if factors == 0:
            return float("nan")
        return len(tok_seq) / factors

    forward = _mtld_calc(tokens)
    backward = _mtld_calc(list(reversed(tokens)))
    return (forward + backward) / 2


def lexical_richness(text):
    doc = nlp(text)
    tokens = [t.text for t in doc if t.is_alpha]  # keep only alphabetic tokens
    return mtld(tokens)


# ---------------------------
# 2. SEMANTIC RICHNESS
#    Mean pairwise distance between sentence embeddings
# ---------------------------



def semantic_richness(text):

    def chunk_text(text, chunk_size=10, min_words=3):
        """
        Turn a raw transcript (even without punctuation) into chunks of ~chunk_size words.
        Only keep chunks with at least min_words.
        """
        # get only word tokens (letters/numbers), ignore dots, commas, etc.
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) >= min_words:
                chunks.append(" ".join(chunk_words))
        return chunks

    def semantic_richness_mean_pairwise_cosine(text, chunk_size=10, min_words=3):
        """
        Semantic richness = mean pairwise cosine distance between
        embeddings of word-chunks from the text.
        Works even when there is no punctuation.
        """
        chunks = chunk_text(text, chunk_size=chunk_size, min_words=min_words)

        if len(chunks) < 2:
            raise ValueError(
                f"Text too short: need at least 2 chunks with ≥{min_words} words. "
                f"Try reducing chunk_size or min_words."
            )

        # 2. Embed each chunk
        embeddings = model.encode(chunks, convert_to_numpy=True)

        # 3. Compute all pairwise cosine distances
        dist_matrix = cosine_distances(embeddings)

        # Take upper triangle (exclude diagonal)
        i_upper, j_upper = np.triu_indices_from(dist_matrix, k=1)
        pairwise_dists = dist_matrix[i_upper, j_upper]

        # 4. Mean pairwise distance = semantic richness score
        return pairwise_dists.mean()

    return  semantic_richness_mean_pairwise_cosine(text, chunk_size=10, min_words=3)

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