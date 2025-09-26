from transformers import pipeline

# Φορτώνουμε το Greek BERT fill-mask μοντέλο
unmasker = pipeline("fill-mask", model="nlpaueb/bert-base-greek-uncased-v1")

# Νομικά κείμενα με μάσκες
masked_sentences = [
    "Αν η κυριότητα του [MASK] ανήκει σε περισσότερους εξ αδιαιρέτου συγκυρίους...",
    "Η σύμβαση θεωρείται άκυρη εάν δεν πληρούνται οι [MASK] που ορίζει ο νόμος."
]

# Για κάθε πρόταση με μάσκα
for sent in masked_sentences:
    print("\n--- Νέα πρόταση με μάσκα ---")
    print("Masked:", sent)

    # Παίρνουμε top-5 προβλέψεις
    results = unmasker(sent, top_k=5)

    for r in results:
        token = r["token_str"]
        score = r["score"]
        filled = sent.replace("[MASK]", token)
        print(f"→ {token} (score={score:.4f})")
        print(f"   Συμπληρωμένη πρόταση: {filled}")
