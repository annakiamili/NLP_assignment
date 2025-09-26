from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
import logging
import warnings

# Καθαρή έξοδος - αγνόησε warnings και info μηνύματα
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


# Επιλογή μοντέλου
MODEL_NAME = "tuner007/pegasus_paraphrase"

# Χρήση PegasusTokenizer (slow tokenizer)
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)

def get_paraphrases(sentence,num_return_sequences=1, num_beams=5 ):
    sentence = sentence.strip().replace("\n", " ")

    batch = tokenizer(
        [f"paraphrase: {sentence}"],
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )

    translated = model.generate(
        **batch,
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )

    if num_return_sequences == 1:
        return tokenizer.decode(translated[0], skip_special_tokens=True).strip()
    else:
        return (tokenizer.decode(t, skip_special_tokens=True).strip() for t in translated)


# Παράδειγμα εισόδου
sentences = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
    "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again."
]

# Εκτέλεση
for i, s in enumerate(sentences, start=1):
    print(f"\n---Sentence {i}---")
    print(f"Original: {s}")
    paraphrased = get_paraphrases(s, num_return_sequences=1)
    print()
    print(f"Rephrased:", paraphrased)
