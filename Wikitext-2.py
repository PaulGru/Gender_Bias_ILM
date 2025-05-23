import os
import random
from tqdm import tqdm
from datasets import load_dataset

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
GENDER_DICT = {w1.lower(): w2.lower() for w1, w2 in GENDER_PAIRS}
GENDER_DICT.update({w2.lower(): w1.lower() for w1, w2 in GENDER_PAIRS})

def swap_gender_terms(sentence, gender_dict):
    tokens = sentence.split()
    swapped_tokens = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token in gender_dict:
            swapped = gender_dict[lower_token]
            if token[0].isupper():
                swapped = swapped.capitalize()
            swapped_tokens.append(swapped)
        else:
            swapped_tokens.append(token)
    return ' '.join(swapped_tokens)

def prepare_split(dataset, seed, p_env_a, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_lines = [l.strip() for l in dataset['train']['text'] if l.strip()]
    val_lines = [l.strip() for l in dataset['validation']['text'] if l.strip()]
    test_lines = [l.strip() for l in dataset['test']['text'] if l.strip()]

    random.seed(seed)
    env_a_lines, env_b_lines = [], []
    for line in tqdm(train_lines, desc=f"Split {seed} - p_env_a={p_env_a}"):
        if random.random() < p_env_a:
            env_a_lines.append(line)
        else:
            env_b_lines.append(swap_gender_terms(line, GENDER_DICT))

    os.makedirs(os.path.join(output_dir, "train_env"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val_env"), exist_ok=True)

    with open(os.path.join(output_dir, "env_A.txt"), "w") as f:
        f.write("\n".join(env_a_lines))
    with open(os.path.join(output_dir, "env_B.txt"), "w") as f:
        f.write("\n".join(env_b_lines))

    with open(os.path.join(output_dir, "train_env", "all_train.txt"), "w") as f:
        f.write("\n".join(env_a_lines + env_b_lines))

    with open(os.path.join(output_dir, "val_env", "val_ind.txt"), "w") as f:
        f.write("\n".join(val_lines))
    with open(os.path.join(output_dir, "val_env", "test.txt"), "w") as f:
        f.write("\n".join(test_lines))

if __name__ == "__main__":
    from datasets import load_dataset

    print("Chargement du dataset Wikitext-2.")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    seeds = [0, 1, 2, 3, 4]
    p_values = [0.1, 0.25, 0.5, 0.7, 0.9]

    for seed in seeds:
        for p in p_values:
            tag = f"split{seed}_p{str(p).replace('.', '')}"
            output_dir = os.path.join("generated_splits", tag)
            prepare_split(dataset, seed=seed, p_env_a=p, output_dir=output_dir)

    print("\n✅ Tous les splits ont été générés.")
