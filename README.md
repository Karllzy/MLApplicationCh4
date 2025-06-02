
# Cotton‑Impurity 1‑D CNN Example

A lightweight, end‑to‑end pipeline for classifying four cotton‑impurity classes  
(`background`, `cotton`, `film_on_background`, `film_on_cotton`) from 288‑band
reflectance spectra.

| Stage | Script | Purpose |
|-------|--------|---------|
| **1. Data prep** | `data_preprocessing.py` | ■ Load CSV & clean non‑numeric columns<br>■ Visualise random spectra per class (optional)<br>■ Make reproducible **5‑fold** splits & save as NPZ |
| **2. Model** | `model.py` | Compact 1‑D CNN<br>  • 2 × Conv→ReLU→MaxPool<br>  • FC 64 → 4 classes |
| **3. Training** | `model_training.py` | Train each fold with PyTorch (GPU‑aware), save weights `models/model_fold_k.pt` |
| **4. Evaluation** | `model_evaluation.py` | Load weights, report per‑fold accuracy & nicely formatted confusion matrix |

---

## 1. Installation

```bash
# (optional) create a clean environment
conda create -n cotton_cnn python=3.10 -y
conda activate cotton_cnn

pip install torch torchvision  # CPU or your CUDA build
pip install pandas matplotlib numpy
````

## 2. Dataset

A single CSV where

* **Columns b1‑b288** = float reflectance values
* **Label column** = `label` (or treated as last column)
* Optional `id/x/y` columns are ignored.

```
id,x,y,b1,b2,...,b288,label
1,12,34,0.04,0.09,...,0.33,background
...
```

## 3. Quick start

```bash
# 3‑a  Data split (adds --vis to save spectra plots)
python data_preprocessing.py \
       --csv cotton_impurity_dataset.csv \
       --out_dir folds \
       --vis

# 3‑b  Train (20 epochs, Adam, automatic GPU/CPU detection)
python model_training.py \
       --fold_dir folds \
       --model_dir models \
       --epochs 20 \
       --batch_size 64

# 3‑c  Evaluate
python model_evaluation.py \
       --fold_dir folds \
       --model_dir models
```

Example overall output:

```
==== Overall ====
Mean Accuracy: 97.68%
Confusion Matrix (sum over folds):
                    background     cotton film_on_background film_on_cotton
background                 2426         0                  0              0
cotton                        0      1247                  0              3
film_on_background            5         0                863              2
film_on_cotton                0        15                  0            721
```

## 4. Script details

### `data_preprocessing.py`

* **Auto‑detects** label column (`label`) or uses last column.
* Drops `id/x/y` columns; selects numeric bands only.
* Saves:

  * `folds/fold_{k}_train.npz` & `folds/fold_{k}_test.npz` (k = 1‑5)
  * `folds/label_map.json` (class→index)
  * `folds/spectra_examples.png` (if `--vis`)

### `model.py`

```text
Input 1×288  →
Conv1d(1→16,k=5,pad=2) → ReLU → MaxPool/2  →
Conv1d(16→32,k=3,pad=1) → ReLU → MaxPool/2 →
Flatten → Linear(32*72→64) → ReLU → Linear(64→4)
```

* Two 2× down‑sampling pools → final length 288/4 = 72.

### `model_training.py`

* Loads each train/test NPZ fold.
* Converts to `torch.FloatTensor`, adds channel dim `(N,1,288)`.
* Adam (lr 1e‑3) + CrossEntropy.
* Saves: `models/model_fold_k.pt`.

### `model_evaluation.py`

* Loads saved weights + test fold.
* Prints per‑fold accuracy and aggregated confusion matrix with
  dynamic column sizing.

## 5. Customisation

| Task                      | How                                                               |
| ------------------------- | ----------------------------------------------------------------- |
| Change network depth      | Edit `model.py`: add more Conv‑ReLU‑Pool blocks or larger FC      |
| Different optimiser       | Modify `optimizer = torch.optim.Adam(...)` in `model_training.py` |
| Epochs / batch size       | Pass `--epochs` / `--batch_size`                                  |
| Force CPU                 | `--device cpu`                                                    |
| Visualise training curves | Log `loss/accuracy` each epoch to a list and plot with Matplotlib |

## 6. Citation

If this repo helps your research or teaching, please cite:

```
@article{LI2023104731,
title = {SCNet: A deep learning network framework for analyzing near-infrared spectroscopy using short-cut},
journal = {Infrared Physics & Technology},
volume = {132},
pages = {104731},
year = {2023},
issn = {1350-4495},
doi = {https://doi.org/10.1016/j.infrared.2023.104731},
author = {Zhenye Li and Dongyi Wang and Tingting Zhu and Chao Ni and Chao Zhou},
}
```

