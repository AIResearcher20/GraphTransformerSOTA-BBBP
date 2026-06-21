import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_epoch(model, loader, optimizer, device, criterion, clip=1.0):
    model.train()
    total_loss = 0
    preds, labels, probs = [], [], []
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        total_loss += loss.item()
        preds.extend(out.argmax(dim=1).cpu().numpy())
        labels.extend(batch.y.cpu().numpy())
        probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    return total_loss / len(loader), acc, f1, auc

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    preds, labels, probs = [], [], []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item()
        preds.extend(out.argmax(dim=1).cpu().numpy())
        labels.extend(batch.y.cpu().numpy())
        probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    return total_loss / len(loader), acc, f1, auc

def train(model, train_loader, val_loader, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience']
    )
    
    epochs = config['training']['epochs']
    patience = config['training']['early_stopping_patience']
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1, train_auc = train_epoch(
            model, train_loader, optimizer, device, criterion,
            config['training']['gradient_clip']
        )
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, device, criterion)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"{config['experiment']['output_dir']}/best_model.pt")
            print("  ✅ Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️ Early stopping at epoch {epoch}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history
