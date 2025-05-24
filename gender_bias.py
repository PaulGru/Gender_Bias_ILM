#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from invariant_distilbert import InvariantDistilBertForMaskedLM
import torch
import math
import re
from tqdm import tqdm

# === PARAMÈTRES ===
generated_root = "generated_splits"
output_root = "trained_models_2500"
bias_output_root = "bias_outputs_2500"
os.makedirs(bias_output_root, exist_ok=True)

learning_rates = [1e-5, 5e-5]
seeds = [0, 1, 2, 3, 4]
p_values = [0.1, 0.25, 0.5, 0.7, 0.9]
methods = ["eLM", "iLM"]
model_type = "distilbert-base-uncased"
use_cuda = torch.cuda.is_available()
gpu_device = "1"

# GENDER MAP
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

# === BIAIS ===
def compute_bias(model, tokenizer, test_lines, model_type):
    bias_scores = []
    model.eval()
    if use_cuda:
        model.to("cuda")

    for line in tqdm(test_lines, desc=f"Biais {model_type}"):
        words = re.findall(r"\b\w+\b", line.lower())
        found = [w for w in words if w in GENDER_MAP]
        if len(found) != 1:
            continue

        target = found[0]
        opposite = GENDER_MAP[target]
        masked_line = re.sub(rf"\b{target}\b", tokenizer.mask_token, line, flags=re.IGNORECASE)
        inputs = tokenizer(masked_line, return_tensors="pt")
        if use_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_idx[0]) == 0:
            continue
        logits = outputs.logits[mask_idx][0]
        probs = logits.softmax(dim=-1)

        try:
            id1 = tokenizer.convert_tokens_to_ids(target)
            id2 = tokenizer.convert_tokens_to_ids(opposite)
            p1 = probs[id1].item()
            p2 = probs[id2].item()
        except:
            continue

        if p1 + p2 < 1e-8:
            continue

        p = p1 / (p1 + p2)
        if p in [0, 1]:
            H = 0
        else:
            H = - (p * math.log2(p) + (1 - p) * math.log2(1 - p))
        B_H = 1 - H
        bias_scores.append(B_H)

    return bias_scores

# === TRAINING + ÉVALUATION ===
for lr in learning_rates:
    for seed in seeds:
        for p in p_values:
            tag = f"split{seed}_p{str(p).replace('.', '')}"
            tag_with_lr = f"{tag}_lr{str(lr).replace('.', '')}"
            split_dir = os.path.join(generated_root, tag)
            test_file = os.path.join(split_dir, "val_env", "test.txt")

            if not os.path.exists(test_file):
                print(f"❌ Fichier de test manquant pour {tag}, on passe.")
                continue

            with open(test_file, "r") as f:
                test_lines = [line.strip() for line in f if line.strip()]

            for method in methods:
                model_path = os.path.join(output_root, tag_with_lr, method)
                if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                    if method == "iLM":
                        train_file = split_dir
                        mode_arg = "iLM"
                    else:
                        train_file = os.path.join(split_dir, "train_env", "all_train.txt")
                        mode_arg = "eLM"

                    # python3 -m torch.distributed.run --nproc_per_node=2 run invariant_mlm.py \
                    train_cmd = f"""
                    CUDA_VISIBLE_DEVICES={gpu_device} python3 run_invariant_mlm.py \
                        --model_type distilbert \
                        --model_name_or_path {model_type} \
                        --train_file {train_file} \
                        --validation_file {split_dir}/val_env \
                        --do_train --do_eval \
                        --nb_steps 2500 --learning_rate {lr} \
                        --output_dir {model_path} \
                        --seed {seed} --per_device_train_batch_size 16 \
                        --preprocessing_num_workers 8 \
                        --gradient_accumulation_steps 3 \
                        --fp16 --overwrite_output_dir \
                        --mode {mode_arg}
                    """
                    print(f"\n[+] Entraînement {method} sur {tag_with_lr}")
                    subprocess.run(train_cmd, shell=True, check=True)

                try:
                    if method == "iLM":
                        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
                        model = InvariantDistilBertForMaskedLM.from_pretrained(model_path)
                    else:
                        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
                        model = DistilBertForMaskedLM.from_pretrained(model_path)

                    scores = compute_bias(model, tokenizer, test_lines, model_type=method)
                    df = pd.DataFrame({"bias": scores})
                    out_path = os.path.join(bias_output_root, f"{tag_with_lr}_{method}.csv")
                    df.to_csv(out_path, index=False)
                    print(f"✅ Biais sauvegardé dans {out_path} ({len(scores)} phrases)")

                except Exception as e:
                    print(f"❌ Erreur pour {tag_with_lr} / {method} : {e}")
