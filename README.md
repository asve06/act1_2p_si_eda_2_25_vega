# Women Perfume Recommendation Based on Note Similarities

## Project Description

This project aims to develop a recommendation system for women’s perfumes based on the similarity of their olfactory notes and accords.
The main objective is to analyze perfume characteristics, identify underlying patterns, and use them to recommend fragrances with similar scent profiles-accords.

---

## Task

Recommend similar perfumes based on their descriptions and shared characteristics.
The project will use textual and categorical data (notes, accords, brands, and ratings) to identify relationships and similarities among perfumes.

---

## Task Type

**`Classification`**

This project applies a **classification and similarity-based approach** to analyze perfumes and group them according to shared olfactory characteristics.

Using techniques such as **`K-Nearest Neighbors (KNN)`**, the system aims to identify perfumes with similar note compositions and accords, enabling meaningful and data-driven fragrance recommendations.

---

## Dataset

**Name:** `fraganciasdt.csv`
**Source:** [Fragrantica Perfume Dataset – Kaggle](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset?select=fra_cleaned.csv)

The dataset contains detailed information about perfumes, including:

- Perfume name and brand
- Country and gender
- Rating value and count
- Year of release
- Notes (Top, Middle, Base)
- Main accords (1–5)
- Perfumers

---

## Tools Used

- **Python 3**
- **Google Colab** – execution environment
- **Pandas** – data manipulation, cleaning, and feature engineering
- **NumPy** – numerical operations
- **Matplotlib** and **Seaborn** – visualization (EDA)
- **Scikit-learn (sklearn)** – TF-IDF vectorization, K-NN modeling, and similarity computation
- **NLTK / Stopwords (if applicable)** – basic text preprocessing
- **SciPy** – cosine distance calculations (via sklearn dependencies)


---

## Repository Structure

```
ACT1_2P_SI_EDA_2_25_VEGA/
│
├── code/                        # Google Colab notebook
│   └── act1_2p_si_eda_2_25_vega.ipynb
│   └── act2_2p_si_tech_2_25_vega.ipynb
│
├── dataset/                     # Datasets used in this project
│   └── fraganciasdt.csv
│   └── df_women.csv
│
└── documentation/               # Report and visual outputs
    ├── graphs/                  # Graphs obtained from EDA
    │   ├── graph1.png
    │   ├── graph2.png
    │   ├── graph3.png
    │   ├── graph4.png
    │   └── graph5.png
    ├── images/                  # Images obtained from TECH
    │   ├── results.png
    │   ├── results2.png
    └── report.md                # Markdown report
    └── report_act2.md           # Encoding & feature processing report
```

---

## Exploratory Data Analysis (EDA)

The **EDA** process was performed to clean, visualize and understand the perfume dataset’s composition in preparation for developing the KNN-based recommendation system.
The key steps included:

1. **Data Cleaning:** Detection and removal of missing or duplicated records.
2. **Univariate Analysis:** Exploration of individual variables such as main accords, ratings, and perfume release year.
3. **Bivariate Analysis:** Examination of relationships between two variables (Brand vs Scent Profiles)
4. **Multivariate Analysis:** Combined analysis of how scent accord frequency has evolved over time (Accord vs. Year vs. Perfume Count).

---

## Key Findings

- The **fruity** and **white floral** accords dominate women’s fragrances, defining the main scent families used in the dataset.
- **France**, **USA**, and **Italy** lead global perfume production, influencing dominant scent trends.
- Each **brand** shows distinct olfactory preferences, reinforcing brand identity in scent design.
- The **distribution of ratings** is highly concentrated at the upper range (3.7–4.4), indicating consistently positive consumer evaluations.
- The analysis confirms that **scent-related variables (notes and accords)** are the most meaningful features for perfume similarity modeling, while popularity or origin-related data play a secondary role.

## Documentation

The detailed report describing the EDA process, findings, and visual analysis is available in:
`documentation/report.md`

All visual outputs are stored in:
`documentation/graphs/`

The complete EDA implementation can be reviewed in the Google Colab notebook:
`code/act1_2p_si_eda_2_25_vega.ipynb`

---

# **Encoding Techniques and Model Evaluation**

## Purpose of Encoding

The dataset contains textual descriptions of fragrance notes and accords.
To build a similarity-based recommendation model using **K-NN + Cosine Distance**, it was necessary to convert this aromatic information into numerical representations.

---

## Techniques Considered

### **2.1 Frequency Encoding (Explored, not applied)**

This method was evaluated for categorical fields such as _Brand_, but it was discarded because brand frequency does not contribute to scent-based similarity and does not reflect aromatic composition.

### **2.2 Text Feature Aggregation (Applied)**

The columns `Top`, `Middle`, `Base`, and `mainaccord1–5` were merged into a single textual field (`features`).
This unified description is required so that TF-IDF can process the complete fragrance profile of each perfume.

### **2.3 TF-IDF Encoding (Applied)**

TF-IDF transforms text into weighted numerical vectors based on term relevance.
It reduces the influence of generic notes and highlights distinctive ones.
This was the main encoding method used to build the similarity matrix for K-NN.

---

## Models Compared

### **Model A – Baseline (No encoding)**

- Uses raw concatenated text.
- Similarity calclated with **Jaccard**, based on word overlap.
- Highly sensitive to noise and unable to capture deeper aromatic structure.

### **Model B – TF-IDF + K-NN (Final Model)**

- Uses aggregated and vectorized text.
- Similarity measured with **cosine distance**.
- Produces more coherent and stable fragrance recommendations.

---

## Summary of Findings

- TF-IDF significantly improves scent-based similarity by capturing distintic aromatic patterns.
- The baseline model only detects simple textual coincidences.
- The enhanced model provides more accurate rankings aligned with the real fragrance composition.

---
