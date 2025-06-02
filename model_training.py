
import argparse, os, json, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from model import Simple1DCNN

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
    return correct / len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold_dir', default='folds')
    ap.add_argument('--model_dir', default='models')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.fold_dir, 'label_map.json')) as f:
        label_map = json.load(f)
    n_classes = len(label_map)

    for k in range(1, 6):
        train_npz = os.path.join(args.fold_dir, f'fold_{k}_train.npz')
        test_npz  = os.path.join(args.fold_dir, f'fold_{k}_test.npz')
        if not (os.path.exists(train_npz) and os.path.exists(test_npz)):
            print(f'[Skip] fold {k}: missing files.')
            continue

        train_data = np.load(train_npz)
        X_train, y_train = train_data['X'], train_data['y']
        test_data  = np.load(test_npz)
        X_test, y_test   = test_data['X'],  test_data['y']

        # Torch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        y_test_t  = torch.tensor(y_test, dtype=torch.long)

        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds  = TensorDataset(X_test_t,  y_test_t)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

        model = Simple1DCNN(input_len=X_train.shape[1], n_classes=n_classes).to(args.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
            acc  = evaluate(model, test_loader, args.device)
            print(f'Fold {k} | Epoch {epoch:02d} | Loss {loss:.4f} | Test Acc {acc*100:.2f}%')

        model_path = os.path.join(args.model_dir, f'model_fold_{k}.pt')
        torch.save(model.state_dict(), model_path)
        print(f'Saved model to {model_path}\n')

if __name__ == '__main__':
    main()
