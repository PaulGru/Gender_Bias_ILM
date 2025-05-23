import torch
import math
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM, DistilBertTokenizer, DistilBertForMaskedLM
from invariant_distilbert import InvariantDistilBertForMaskedLM
from tqdm import tqdm

# === CONFIG ===
MODEL_TYPE = "elm"  # "ilm"
TOKENIZER_PATH = "output/Wikitext-0.1_eLM"
MODEL_PATH = "output/Wikitext-0.1_eLM/best_model" # if MODEL_TYPE == "ilm" else "./distilbert-mlm-finetuned"
TEST_PATH = "Wikitext-0.1/val_env/test.txt"

USE_CUDA = torch.cuda.is_available()

# === GENRE PAIRS ===
GENDER_PAIRS = [
    ("he", "she"), ("him", "her"), ("his", "hers"),
    ("man", "woman"), ("men", "women"),
    ("boy", "girl"), ("boys", "girls"),
    ("father", "mother"), ("fathers", "mothers"),
    ("son", "daughter"), ("sons", "daughters"),
    ("brother", "sister"), ("brothers", "sisters"),
    ("uncle", "aunt"), ("uncles", "aunts"),
    ("husband", "wife"), ("husbands", "wives"),
    ("actor", "actress"), ("actors", "actresses"),
    ("king", "queen"), ("kings", "queens"),
    ("waiter", "waitress"), ("waiters", "waitresses"),
    ("prince", "princess"), ("princes", "princesses"),
    ("mr.", "mrs."), ("mr", "mrs"),
    ("male", "female"), ("males", "females"),
    ("gentleman", "lady"), ("gentlemen", "ladies"),
    ("businessman", "businesswoman"), ("businessmen", "businesswomen"),
    ("boyfriend", "girlfriend"), ("boyfriends", "girlfriends"),
    ("stepfather", "stepmother"), ("stepfathers", "stepmothers"),
    ("spokesman", "spokeswoman"), ("spokesmen", "spokeswomen"),
    ("hero", "heroine"), ("heroes", "heroines"),
    ("grandson", "granddaughter"), ("grandsons", "granddaughters"),
]
GENDER_MAP = {w1: w2 for w1, w2 in GENDER_PAIRS}
GENDER_MAP.update({w2: w1 for w1, w2 in GENDER_PAIRS})

# === CHARGER MODELE ===
if MODEL_TYPE == "ilm":
    #AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = InvariantDistilBertForMaskedLM.from_pretrained(MODEL_PATH)
else:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = DistilBertForMaskedLM.from_pretrained(MODEL_PATH)

model.eval()
if USE_CUDA:
    model.to("cuda")

# === CHARGER TEST ===
with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_lines = [line.strip() for line in f if line.strip()]

# === ÉVALUATION DU BIAIS ===
bias_scores = []
for line in tqdm(test_lines, desc="Évaluation du biais"):
    words = re.findall(r"\b\w+\b", line.lower())
    found = [w for w in words if w in GENDER_MAP]

    if len(found) != 1:
        continue  # sauter si pas exactement un mot genré

    target_word = found[0]
    opposite_word = GENDER_MAP[target_word]

    # Remplacer le mot genré par [MASK]
    masked_line = re.sub(rf"\b{target_word}\b", tokenizer.mask_token, line, flags=re.IGNORECASE)
    inputs = tokenizer(masked_line, return_tensors="pt")
    if USE_CUDA:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Localiser l’index du [MASK]
    mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if len(mask_index[0]) == 0:
        continue
    logits = outputs.logits[mask_index][0]
    probs = logits.softmax(dim=-1)

    try:
        id1 = tokenizer.convert_tokens_to_ids(target_word)
        id2 = tokenizer.convert_tokens_to_ids(opposite_word)
        p1 = probs[id1].item()
        p2 = probs[id2].item()
    except:
        continue

    if p1 + p2 < 1e-8:
        continue

    # Normaliser et calculer entropie
    p = p1 / (p1 + p2)
    if p in [0, 1]:
        H = 0.0
    else:
        H = - (p * math.log2(p) + (1 - p) * math.log2(1 - p))
    B_H = 1 - H
    bias_scores.append(B_H)

# === MOYENNE ===
if bias_scores:
    avg_bias = sum(bias_scores) / len(bias_scores)
    print(f"\n✅ Biais moyen d'entropie sur {len(bias_scores)} phrases : {avg_bias:.4f}")
else:
    print("⚠️ Aucune phrase admissible trouvée pour l'évaluation.")
