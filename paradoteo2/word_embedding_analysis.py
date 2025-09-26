import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec, FastText
from transformers import AutoTokenizer, AutoModel
import torch

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


# Δεδομένα από Παραδοτέο 1

original_sentences = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
    "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again."
]

paraphrased_sentences = [
    "Today we celebrate the Dragon Boat Festival in Chinese culture, wishing safety and happiness in our lives.",
    "Please remind me if the doctor still plans to edit the acknowledgments section before sending it again."
]


# Word2Vec Embeddings

corpus = [s.split() for s in original_sentences + paraphrased_sentences]
w2v_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=2)

def get_w2v_embedding(sentence):
    tokens = sentence.split()
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vecs, axis=0)


# FastText Embeddings

ft_model = FastText(sentences=corpus, vector_size=100, window=5, min_count=1, workers=2)

def get_ft_embedding(sentence):
    tokens = sentence.split()
    vecs = [ft_model.wv[w] for w in tokens if w in ft_model.wv]
    return np.mean(vecs, axis=0)


# BERT Embeddings (Sentence-BERT style)

bert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
bert_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_bert_embedding(sentence):
    inputs = bert_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# Similarity Analysis
#
methods = {
    "Word2Vec": get_w2v_embedding,
    "FastText": get_ft_embedding,
    "BERT": get_bert_embedding
}


for method, func in methods.items():
    print(f"\n--- {method} ---")
    for i in range(len(original_sentences)):
        orig_text = original_sentences[i]
        para_text = paraphrased_sentences[i]
        orig_vec = func(orig_text)
        para_vec = func(para_text)
        sim = cosine_similarity([orig_vec], [para_vec])[0][0]

        print(f"\nPair {i+1}:")
        print(f"Original   : {orig_text}")
        print(f"Paraphrased: {para_text}")
        print(f"Similarity : {sim:.4f}")


# Visualization (PCA + t-SNE)

all_sentences = original_sentences + paraphrased_sentences
labels = ["Original 1", "Original 2", "Paraphrased 1", "Paraphrased 2"]

for method, func in methods.items():
    embeddings = np.array([func(s) for s in all_sentences])

    # PCA
    pca = PCA(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(6,5))
    for i, label in enumerate(labels):
        plt.scatter(pca[i,0], pca[i,1], label=label)
    plt.title(f"PCA projection ({method})")
    plt.legend()
    plt.savefig(f"pca_{method}.png")
    plt.close()

    # t-SNE (με perplexity=2 επειδή έχουμε λίγα δείγματα)
    tsne = TSNE(n_components=2, perplexity=2, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(6,5))
    for i, label in enumerate(labels):
        plt.scatter(tsne[i,0], tsne[i,1], label=label)
    plt.title(f"t-SNE projection ({method})")
    plt.legend()
    plt.savefig(f"tsne_{method}.png")
    plt.close()


