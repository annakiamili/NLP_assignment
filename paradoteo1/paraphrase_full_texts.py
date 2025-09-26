import logging
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Απενεργοποίηση warnings/infos για καθαρή έξοδο
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ορισμός μοντέλων
models = {
    "Pegasus": "tuner007/pegasus_paraphrase",
    "T5": "Vamsi/T5_Paraphrase_Paws",
    "BART": "eugenesiow/bart-paraphrase"
}

# Φορτώνουμε tokenizer + model για το καθένα
tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in models.items()}
nlp_models = {name: AutoModelForSeq2SeqLM.from_pretrained(path) for name, path in models.items()}

def paraphrase(text, model_name, num_beams=5):
    """Ανακατασκευή κειμένου με το αντίστοιχο μοντέλο"""
    tokenizer = tokenizers[model_name]
    model = nlp_models[model_name]

    # Προετοιμασία input
    inputs = tokenizer(
        [f"paraphrase: {text}"],
        return_tensors="pt",
        truncation=True,
        padding="longest"
    )

    # Παραγωγή ανακατασκευής
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_beams=num_beams,
        num_return_sequences=1
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Τα δύο κείμενα από την εκφώνηση
text1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. 
Hope you too, to enjoy it as my deepest wishes. 
Thank your message to show our words to the doctor, as his next contract checking, to all of us. 
I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago.  
I am very appreciated the full support of the professor, for our Springer proceedings publication."""

text2 = """During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. 
We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. 
Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."""

# Εκτέλεση για κάθε κείμενο
for i, text in enumerate([text1, text2], start=1):
    print(f"--- Text {i} ---")
    print(f"Original:\n{text}\n")
    print()

    for model_name in models.keys():
        paraphrased = paraphrase(text, model_name)
        print(f"\n[{model_name}] Rephrased:\n{paraphrased}\n")
