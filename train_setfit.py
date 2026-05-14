import json
import random

import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold

SEED = 42
BASE_MODEL = "sdadas/st-polish-paraphrase-from-mpnet"
N_SPLITS = 5

random.seed(SEED)
np.random.seed(SEED)

rows = [json.loads(line) for line in open("data/prompts.jsonl") if line.strip()]
random.shuffle(rows)

label_map = {"safe": 0, "unsafe": 1}


def to_ds(rs):
    return Dataset.from_list(
        [{"text": r["prompt"], "label": label_map[r["label"]]} for r in rs]
    )


def train(train_rs):
    model = SetFitModel.from_pretrained(BASE_MODEL)
    args = TrainingArguments(
        batch_size=16, num_epochs=2, num_iterations=20,
        output_dir="/tmp/setfit-out", report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=to_ds(train_rs))
    trainer.train()
    return model


print(f"=== {N_SPLITS}-fold cross-validation on {len(rows)} prompts ===")
labels = np.array([label_map[r["label"]] for r in rows])
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

cv_results = []
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(rows, labels)):
    train_rs = [rows[i] for i in train_idx]
    test_rs = [rows[i] for i in test_idx]
    print(f"\nfold {fold_idx + 1}/{N_SPLITS}: {len(train_rs)} train / {len(test_rs)} test")
    model = train(train_rs)
    probs = model.predict_proba([r["prompt"] for r in test_rs])[:, 1].tolist()
    for r, p in zip(test_rs, probs, strict=True):
        cv_results.append({"id": r["id"], "label": r["label"],
                           "prob_unsafe": float(p), "fold": fold_idx})

with open("models/cv_results.jsonl", "w") as f:
    for entry in cv_results:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"\nsaved {len(cv_results)} CV predictions to models/cv_results.jsonl")

# Production model on full 100%
print(f"\n=== Production: train on full {len(rows)} ===")
prod_model = train(rows)
prod_model.save_pretrained("models/setfit")
print("saved production model to models/setfit/")
