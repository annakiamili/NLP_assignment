from new_paraphrase_sentences import get_paraphrases, sentences as sentences_A
from paraphrase_full_texts import paraphrase, models, text1, text2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Τρέχουμε (Α) - Custom paraphrase για 2 προτάσεις

custom_A = [get_paraphrases(s, num_return_sequences=1) for s in sentences_A]


# Τρέχουμε (Β) - Paraphrase ολόκληρων κειμένων με 3 pipelines

texts_B = [text1, text2]
reconstructions_B = {m: [] for m in models}
for text in texts_B:
    for model_name in models:
        reconstructions_B[model_name].append(paraphrase(text, model_name))


# Υπολογισμός Similarity

embedder = SentenceTransformer("all-MiniLM-L6-v2")


for i, s in enumerate(sentences_A):
    orig_vec = embedder.encode([s])
    para_vec = embedder.encode([custom_A[i]])
    sim = cosine_similarity(orig_vec, para_vec)[0][0]
    print(f"Sentence {i+1}: {sim:.4f}")


for i, text in enumerate(texts_B, start=1):
    orig_vec = embedder.encode([text])
    print(f"\nΚείμενο {i}:")
    for model_name, texts in reconstructions_B.items():
        para_vec = embedder.encode([texts[i-1]])
        sim = cosine_similarity(orig_vec, para_vec)[0][0]
        print(f"- {model_name}: {sim:.4f}")


# PCA Visualization

labels, all_texts = [], []

# Προτάσεις (Α)
for i, s in enumerate(sentences_A):
    all_texts.append(s)
    labels.append(f"Original Sentence {i+1}")
    all_texts.append(custom_A[i])
    labels.append(f"Custom_A {i+1}")

# Κείμενα (Β)
for i, text in enumerate(texts_B, start=1):
    all_texts.append(text)
    labels.append(f"Original Text {i}")
    for model_name, texts in reconstructions_B.items():
        all_texts.append(texts[i-1])
        labels.append(f"{model_name} Text {i}")

vecs = embedder.encode(all_texts)
p2 = PCA(n_components=2).fit_transform(vecs)

plt.figure(figsize=(10,7))
for i, label in enumerate(labels):
    marker = "o" if "Original" in label else "x"
    plt.scatter(p2[i,0], p2[i,1], marker=marker, label=label)

plt.title("PCA of Original vs Paraphrased (Α & Β)")
plt.legend()
plt.savefig("pca_comparison_AB.png")
print("\nSaved PCA plot to pca_comparison_AB.png")
