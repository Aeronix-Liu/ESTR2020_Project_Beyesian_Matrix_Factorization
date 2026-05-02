# Bayesian Matrix Factorization — Quick Start

BMF-VI vs MF-Adam on MovieLens-style ratings (`ratings.csv`). Details: `report.pdf`.

---

## Environment

- **Python 3**
- **Packages:** `torch`, `numpy`, `pandas`, `matplotlib`, `tqdm`
- **Notebook:** same packages in your Jupyter/VS Code kernel; add `jupyter` / `ipykernel` if needed.
- **GPU (optional):** for mode `2`, install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org); otherwise mode `2` falls back to CPU.

```bash
pip install torch numpy pandas matplotlib tqdm
```

---

## File layout and how to launch

**Layout (project root — your shell / notebook `cwd` should be this folder):**

```text
SrcCode/
├── ratings.csv              # required: MovieLens ratings
├── SrcCode.py               # run: python SrcCode.py
├── SrcCode.ipynb            # or open this notebook here
├── README.md
└── BMF_MF_performance.png   # produced after a successful run
```

1. Put **`ratings.csv`** (header `userId,movieId,rating,timestamp`) in the **same folder** as **`SrcCode.py`** / **`SrcCode.ipynb`**.
2. Open a terminal **in that folder** (the code uses `pd.read_csv("ratings.csv")` relative to the **current working directory**).
3. The code in .py and .ipynb are the same, you can choose to run either one. We choose the seed to be the current time. You can change the seed to be a fixed value if you want to reproduce the results.
4. When prompted, choose **`1`** full CPU, **`2`** full GPU (CUDA if available), **`3`** minimal subset (~top **120** users and **280** movies by rating count). Anything else defaults to **`3`**.

**Script:**

```bash
cd path/SrcCode
python SrcCode.py
```

**Notebook:** open `SrcCode.ipynb` with the working directory set to this folder (or start Jupyter from this folder), then run all cells.


**Output:** metrics and a sample table in the console; figure **`BMF_MF_performance.png`** saved in the **same directory** you ran from (300 DPI).
