
import torch
import torch.nn as nn
from metrics import *
import json

model=None
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
##### use :scaler = torch.cuda.amp.GradScaler()

# Train with mixed precision
###########    use : with torch.cuda.amp.autocast():
def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, 
    num_epochs, device, checkpoint_path='/kaggle/working/model_checkpoint.pth'
):
    best_val_loss = float("inf")  # Initialize best metric to a very high value
    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_ious = []
    val_ious = []
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        iou_list = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            outputs = model(inputs)
            targets = targets.squeeze(1)
            targets = targets.long()
            loss = criterion(outputs, targets)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy = compute_accuracy(outputs, targets)
            running_corrects += accuracy * targets.size(0)
            total_samples += targets.size(0)
            # Compute IoU for the current batch
            ious = compute_mean_iou(outputs.argmax(dim=1), targets)
            iou_list.append(ious)
        # Calculate epoch metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = running_corrects / total_samples
        avg_train_iou = torch.mean(torch.tensor(iou_list)).item()

        # Adjust learning rate
        scheduler.step()

        # Validation step
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        val_iou_list = []
        model.eval()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze(1)
                targets = targets.long()
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_corrects += compute_accuracy(outputs, targets) * targets.size(0)
                val_samples += targets.size(0)

                # Compute IoU for validation batch
                val_ious = compute_mean_iou(outputs.argmax(dim=1), targets)
                val_iou_list.append(val_ious)

        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_accuracy = val_corrects / val_samples
        avg_val_iou = torch.mean(torch.tensor(val_iou_list)).item()

        # Save model if validation loss improves
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)

        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Train IoU: {avg_train_iou:.4f}, "
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
            f"Validation IoU: {avg_val_iou:.4f}"
        )
        if train_accuracy>val_accuracy+0.07 and epoch >10:
          print("the training process will be stoped : it's start overtfit to the validation , exiting...")
          #break

        # Store metrics at the end of each epoch
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_ious.append(avg_train_iou)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_ious.append(avg_val_iou)

    # After the training loop finishes, collect all metrics in a dictionary
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_ious': train_ious,
        'val_ious': val_ious,
    }

    # Convert the dictionary to JSON
    metrics_json = json.dumps(metrics)

    # Optionally save the metrics to a JSON file
    with open('training_metrics.json', 'w') as f:
        f.write(metrics_json)

    # Return the metrics dictionary
    return metrics
