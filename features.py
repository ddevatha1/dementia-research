# dementia_utils/functions.py

import math
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from fastcoref import FCoref

coref_model = FCoref()
nlp = spacy.load('en_core_web_sm')

# Assume SBERT model is already loaded
from sentence_transformers import SentenceTransformer
_sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 1) Sentence-to-Sentence Embedding Similarity (mean only)
# -----------------------------
def adjacent_cosine_similarity(sentences: list) -> float:
    """
    Compute mean cosine similarity between adjacent sentences.
    """
    if len(sentences) < 2:
        return 0.0
    
    embeddings = [_sbert_model.encode(s, show_progress_bar=False) for s in sentences]
    sims = []
    for i in range(len(embeddings) - 1):
        a = embeddings[i].reshape(1, -1)
        b = embeddings[i + 1].reshape(1, -1)
        sims.append(float(cosine_similarity(a, b)[0, 0]))
    return float(np.mean(sims)) if sims else 0.0


# -----------------------------
# 2) Windowed / Global Embedding Coherence
# -----------------------------
def windowed_embedding_coherence(sentences: list) -> float:
    """
    Compute average pairwise similarity within sliding windows.
    """

    window = max(2, round(len(sentences) / 4))

    if len(sentences) < 2:
        return 0.0
    
    embeddings = [_sbert_model.encode(s, show_progress_bar=False) for s in sentences]
    scores = []
    for i in range(len(embeddings) - window + 1):
        window_emb = embeddings[i:i+window]
        pair_sims = []
        for j in range(window):
            for k in range(j+1, window):
                a = window_emb[j].reshape(1, -1)
                b = window_emb[k].reshape(1, -1)
                pair_sims.append(float(cosine_similarity(a, b)[0, 0]))
        scores.append(np.mean(pair_sims))
    return float(np.mean(scores)) if scores else 0.0


# -----------------------------
# 3) Intra-Sentence Repetition
# -----------------------------
def intra_sentence_repetition(sentences: list) -> float:
    """
    Fraction of repeated content-word lemmas within each sentence.
    Assumes input is list of lists of content-word lemmas.
    """
    if not sentences:
        return 0.0
    
    fractions = []
    for sent in sentences:
        if not sent:
            continue
        counts = Counter(sent)
        repeated = sum(1 for v in counts.values() if v > 1)
        fractions.append(repeated / len(sent))
    return float(np.mean(fractions)) if fractions else 0.0


# -----------------------------
# 4) Lexical / Content-Word Overlap
# -----------------------------
def lexical_overlap(sentences: list) -> float:
    """
    Compute Jaccard similarity between adjacent sentences.
    Assumes input is list of sets of content-word lemmas.
    """
    if len(sentences) < 2:
        return 0.0
    overlaps = []
    for i in range(len(sentences)-1):
        s1, s2 = set(sentences[i]), set(sentences[i+1])
        if not s1 or not s2:
            continue
        overlaps.append(len(s1 & s2) / len(s1 | s2))
    return float(np.mean(overlaps)) if overlaps else 0.0


# -----------------------------
# 5) Disfluency & Filler Rate
# -----------------------------
def filler_rate(sentences: list) -> float:
    """
    sentences: list of strings (participant speech)
    """
    if not sentences:
        return 0.0
    fillers = {"um", "uh", "you know", "like", "well", "so", "hmm", "ah", "er"}
    total_tokens = 0
    total_fillers = 0
    for sentence in sentences:
        tokens = sentence.lower().split()
        total_tokens += len(tokens)
        total_fillers += sum(1 for t in tokens if t in fillers or (t.startswith("&-") and t[2:] in fillers))
    return total_fillers / total_tokens if total_tokens > 0 else 0.0


# 6) Coref Chain Metrics

# -----------------------------
# 6a) Average Chain Length  
# -----------------------------
def avg_chain_len(sentences: list):
    corpus = " ".join(sentences)

    result = coref_model.predict([corpus])[0]
    clusters = result.clusters

    lens = list()
    for cluster in clusters:
        lens.append(len(cluster))
    
    return np.mean(lens) if lens else 0.0


# -----------------------------
# 6b) Coreference Density  
# -----------------------------
def coref_density(sentences: list):
    corpus = " ".join(sentences)

    total_len = len(corpus.split())
    result = coref_model.predict([corpus]
)[0]

    num_mentions = sum(len(chain) for chain in result.clusters)

    return num_mentions / total_len if total_len > 0 else 0.0


# -----------------------------
# 6c) Singleton Ratio
# -----------------------------
def singleton_ratio(sentences: list):
    corpus = " ".join(sentences)

    result = coref_model.predict([corpus]
)[0]
    total_len = len(result.clusters)

    numerator = sum(1 for chain in result.clusters if len(chain) == 1)
    return numerator / total_len if total_len else 0.0


# -----------------------------
# 6d) Entity Transition / Entropy
# -----------------------------
def entity_transition_entropy(sentences: list) -> float:
    """
    Compute entropy of entity transitions.
    transitions: list of strings like 'S', 'O', 'X' per sentence.
    Metric: mean entropy score across all mentioned entities (coref resolved)
    """

    # coreference resolution model
    text = " ".join(sentences)

    result = coref_model.predict(
        texts = [text],
    )[0]

    words_list = [f"{word} " for word in text.split()]

    spans_list = dict()

    for i, entities in enumerate(result.clusters):
        mentions_list = []
        for entity in entities:
            _, span = result.char_map[entity]
            mentions_list.append(span)
        spans_list[i] = mentions_list

    indices_dict = dict()

    for id, spans in spans_list.items():
        indices = list()
        for span in spans:
            start, end = span
            for i, word in enumerate(words_list):
                if end <= 0:
                    indices.append(i-1)
                    break
                else:
                    end -= len(word)
        indices_dict[id] = indices

    text_no_punct = " ".join(t.text for t in nlp(text) if not t.is_punct)
    doc = nlp(text_no_punct)

    entity_grid = list() # list of lists, each ent stores a list of roles

    for id, indices in indices_dict.items():
        roles = list()
        for i in indices:
            token = doc[i]
            if 'subj' in token.dep_:
                roles.append('S')
            elif 'obj' in token.dep_:
                roles.append('O')
            elif 'conj' in token.dep_:
                head_dep = token.head
                while 'conj' in head_dep.dep_:
                    head_dep = head_dep.head
                if 'subj' in head_dep.dep_:
                    roles.append('S')
                elif 'obj' in head_dep.dep_:
                    roles.append('O')
                else:
                    roles.append('X')    
        entity_grid.append(roles)

    transition_scores = list()

    for roles in entity_grid:
        if len(roles) == 0:
            continue
        transitions = list()
        for i in range(len(roles)-1):
            transitions.append(f'{roles[i]}{roles[i+1]}')
        set_transitions = set(transitions)
        entropy_sum = 0
        for item in set_transitions:
            p_t = transitions.count(item) / len(transitions)
            entropy_sum += (p_t * math.log2(p_t))
        transition_scores.append(-entropy_sum)
    
    return np.mean(transition_scores) if transition_scores else 0.0
