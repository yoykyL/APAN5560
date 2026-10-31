"""Training loop with checkpoint support."""
from tqdm import tqdm

from .checkpoints import save_checkpoint
from .evaluator import evaluate_model


def train_model(model, train_loader, val_loader, criterion, optimizer,
                device="cpu", epochs=10, checkpoint_dir="checkpoints", start_epoch=0):
    """Train the model, validating and saving a checkpoint after every epoch.

    - Saves checkpoints/model_epoch_XXX.pth after each epoch
    - Saves checkpoints/best_model.pth whenever validation accuracy improves
    - To resume, load a checkpoint first and pass start_epoch=<its epoch>
    - Returns the trained model; per-epoch metrics are in model.history
    """
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=running_loss / total)

        train_loss, train_acc = running_loss / total, 100 * correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        print(f"Epoch {epoch}: train loss {train_loss:.4f}, acc {train_acc:.2f}% | "
              f"val loss {val_loss:.4f}, acc {val_acc:.2f}%")

        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_dir)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc,
                            checkpoint_dir, filename="best_model.pth")
            print(f"  New best model (val acc {val_acc:.2f}%) saved")

    model.history = history
    return model
