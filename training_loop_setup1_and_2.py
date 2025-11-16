from sklearn.metrics import f1_score


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        # Print intermediate metrics every 5 batches
        if (batch_idx + 1) % 20 == 0:
            current_loss = running_loss / total
            current_acc = 100.0 * correct / total
            print(f"[Batch {batch_idx + 1}/{len(dataloader)}] Loss: {current_loss:.4f} | Acc: {current_acc:.2f}%")

    return running_loss / total, 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            # Print intermediate metrics every 5 batches
            if (batch_idx + 1) % 20 == 0:
                current_loss = running_loss / total
                current_acc = 100.0 * correct / total
                print(f"[Val Batch {batch_idx + 1}/{len(dataloader)}] Loss: {current_loss:.4f} | Acc: {current_acc:.2f}%")

    return running_loss / total, 100.0 * correct / total


import torch

def train_model(model, train_loader, test_loader, history, epochs=10, model_saving_path='best_model.pth', patience=5):
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = validate(model, test_loader, criterion, device, return_f1=True)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")

        # ---- early stopping based on macro-F1 ----
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_saving_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered (no F1 improvement in {patience} epochs).")
                break

    print(f"\n[INFO] Training Complete. Best Val F1: {best_f1:.4f}")
    return history

def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # ---- Loss ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'o-', label='Train Loss')
    plt.plot(epochs, history["val_loss"], 'o-', label='Val Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # ---- Accuracy ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], 'o-', label='Train Acc')
    plt.plot(epochs, history["val_acc"], 'o-', label='Val Acc')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


