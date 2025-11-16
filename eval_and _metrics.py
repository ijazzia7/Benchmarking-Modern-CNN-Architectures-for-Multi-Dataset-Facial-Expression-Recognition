device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 2️⃣  Evaluation helper
# ------------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm




# ------------------------------
# 3️⃣  Run all models
# ------------------------------
results = {}
for key, (name, model, weight_file) in model_weights_mapping.items():
    weight_path = os.path.join(model_dir, weight_file)
    print(f"\n[INFO] Loading {name} from {weight_file}")
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)

    acc, prec, rec, f1, cm = evaluate_model(model, test_loader, device)
    results[key] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm}

    print(f"✅ {name} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")




# ------------------------------
# 4️⃣  Visualization
# ------------------------------
# Bar chart comparing all models
metric_names = ["acc", "prec", "rec", "f1"]
for metric in metric_names:
    plt.figure(figsize=(10,5))
    values = [results[m][metric] for m in results]
    plt.bar(range(len(results)), values)
    plt.xticks(range(len(results)), results.keys(), rotation=45, ha='right')
    plt.ylabel(metric.upper())
    plt.title(f"Model Comparison by {metric.upper()}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ------------------------------
# 5️⃣  Show confusion matrices (optional)
# ------------------------------
for key, r in results.items():
    plt.figure(figsize=(5,5))
    sns.heatmap(r["cm"], annot=False, cmap="Blues")
    plt.title(f"Confusion Matrix — {key}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# ------------------------------
# 6️⃣  Summary table
# ------------------------------
import pandas as pd

summary_df = pd.DataFrame.from_dict(results, orient='index')[["acc","prec","rec","f1"]]
display(summary_df.sort_values("f1", ascending=False))
