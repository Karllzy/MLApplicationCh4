
import argparse, os, json, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from model import Simple1DCNN

@torch.no_grad()
def evaluate_fold(model, loader, device, n_classes):
    model.eval()
    correct = 0
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        for t, p in zip(yb, preds):
            cm[t, p] += 1
    acc = correct / len(loader.dataset)
    return acc, cm.cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold_dir', default='folds')
    ap.add_argument('--model_dir', default='models')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    with open(os.path.join(args.fold_dir, 'label_map.json')) as f:
        label_map = json.load(f)
    n_classes = len(label_map)
    idx_to_label = {v:k for k,v in label_map.items()}

    all_acc = []
    total_cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    for k in range(1, 6):
        test_npz = os.path.join(args.fold_dir, f'fold_{k}_test.npz')
        model_path = os.path.join(args.model_dir, f'model_fold_{k}.pt')
        if not (os.path.exists(test_npz) and os.path.exists(model_path)):
            print(f'[Skip] fold {k}: missing files.')
            continue

        data = np.load(test_npz)
        X_test = torch.tensor(data['X'], dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(data['y'], dtype=torch.long)
        loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

        model = Simple1DCNN(input_len=X_test.shape[-1], n_classes=n_classes).to(args.device)
        model.load_state_dict(torch.load(model_path, map_location=args.device))

        acc, cm = evaluate_fold(model, loader, args.device, n_classes)
        all_acc.append(acc)
        total_cm += cm
        print(f'Fold {k} Accuracy: {acc*100:.2f}%')

    if all_acc:
        print("\n==== Overall ====")
        print(f"Mean Accuracy: {np.mean(all_acc)*100:.2f}%")

        # ---------- 友好格式化混淆矩阵 ----------
        labels = [idx_to_label[i] for i in range(n_classes)]
        # 列宽 = max(类别名长度, 数字宽度) + 2
        num_width  = max(len(str(total_cm.max().item())), 5)
        name_width = max(max(len(l) for l in labels), 8)
        col_width  = max(num_width, name_width) + 2

        # 打印表头
        header = " " * (name_width + 2)  # 左上角空格
        header += "".join(f"{lbl:>{col_width}}" for lbl in labels)
        print("Confusion Matrix (sum over folds):")
        print(header)

        # 打印每一行
        for i, lbl in enumerate(labels):
            row_vals = " ".join(f"{total_cm[i, j]:>{col_width}d}" for j in range(n_classes))
            print(f"{lbl:<{name_width}}  {row_vals}")


if __name__ == '__main__':
    main()
