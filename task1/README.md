# 🌸 Task 1: Exploring and Visualizing the Iris Dataset

**DevelopersHub Corporation — AI/ML Engineering Internship**

---

## 📌 Task Objective

Load, inspect, and visualize the Iris dataset to understand data trends, distributions, and relationships between features. This forms the foundation of any data science workflow — understanding your data before modeling.

---

## 📂 Dataset Used

| Property | Detail |
|---|---|
| **Name** | Iris Dataset |
| **Source** | Loaded via `seaborn.load_dataset('iris')` |
| **Rows** | 150 |
| **Columns** | 5 (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species`) |
| **Classes** | 3 species: *setosa*, *versicolor*, *virginica* |
| **Missing Values** | None |

---

## 🛠️ Libraries Used

- `pandas` — data loading and inspection
- `numpy` — numerical operations
- `matplotlib` — base plotting
- `seaborn` — statistical visualizations

---

## 📊 Analysis Performed

### 1. Data Inspection
- `.shape`, `.head()`, `.tail()` for basic inspection
- `.info()` for data types and null check
- `.describe()` for summary statistics
- `.groupby('species').describe()` for per-class stats

### 2. Visualizations Created

| Plot | Purpose | File |
|---|---|---|
| Pair Plot (Scatter Matrix) | Pairwise feature relationships by species | `scatter_pairplot.png` |
| Scatter Plot (Petal L vs W) | Best feature separation view | `scatter_petal.png` |
| Histograms (all 4 features) | Value distributions per species | `histograms.png` |
| Box Plots (all 4 features) | Spread, median, and outlier detection | `boxplots.png` |
| Violin Plots | Distribution density + spread | `violin_plots.png` |
| Correlation Heatmap | Feature correlation strengths | `correlation_heatmap.png` |

---

## 🔍 Key Results and Findings

1. **Petal length and petal width** are the most discriminating features — they form clear, well-separated clusters for each species.
2. **Sepal width** shows the most overlap between species and is the weakest predictor.
3. **Setosa** is easily separable from the other two species using petal measurements alone.
4. **Versicolor and Virginica** slightly overlap, making them harder to distinguish without petal features.
5. **Strong correlation (r = 0.96)** exists between petal length and petal width.
6. The dataset is perfectly **balanced** (50 samples per class) with **no missing values**.

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/developershub-aiml-internship.git
cd developershub-aiml-internship/task1

# Install dependencies
pip install pandas numpy matplotlib seaborn

# Launch the notebook
jupyter notebook task1_iris_eda.ipynb
```

---

## 📁 File Structure

```
task1/
├── task1_iris_eda.ipynb       # Main Jupyter Notebook
├── scatter_pairplot.png       # Pair plot visualization
├── scatter_petal.png          # Petal scatter plot
├── histograms.png             # Feature histograms
├── boxplots.png               # Box plots
├── violin_plots.png           # Violin plots
└── README.md                  # This file
```

---

*Submitted as part of the DevelopersHub Corporation AI/ML Engineering Internship — Task 1*
