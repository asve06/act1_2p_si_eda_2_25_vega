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

* Perfume name and brand
* Country and gender
* Rating value and count
* Year of release
* Notes (Top, Middle, Base)
* Main accords (1–5)
* Perfumers

---

## Tools Used

* **Python 3**
* **Google Colab** – execution environment
* **Pandas** – data manipulation and cleaning
* **NumPy** – numerical operations
* **Matplotlib** and **Seaborn** – visualization

---

## Repository Structure

```
ACT1_2P_SI_EDA_2_25_VEGA/
│
├── code/                        # Google Colab notebook
│   └── act1_2p_si_eda_2_25_vega.ipynb
│
├── dataset/                     # Dataset used in this project
│   └── fraganciasdt.csv
│
└── documentation/               # Report and visual outputs
    ├── graphs/                  # Graphs obtained from EDA
    │   ├── graph1.png
    │   ├── graph2.png
    │   ├── graph3.png
    │   ├── graph4.png
    │   └── graph5.png
    └── report.md                # Markdown report 
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

* The **fruity** and **white floral** accords dominate women’s fragrances, defining the main scent families used in the dataset.
* **France**, **USA**, and **Italy** lead global perfume production, influencing dominant scent trends.
* Each **brand** shows distinct olfactory preferences, reinforcing brand identity in scent design.
* The **distribution of ratings** is highly concentrated at the upper range (3.7–4.4), indicating consistently positive consumer evaluations.
* The analysis confirms that **scent-related variables (notes and accords)** are the most meaningful features for perfume similarity modeling, while popularity or origin-related data play a secondary role.

## Documentation

The detailed report describing the EDA process, findings, and visual analysis is available in:
`documentation/report.md`

All visual outputs are stored in:
`documentation/graphs/`

The complete EDA implementation can be reviewed in the Google Colab notebook:
`code/act1_2p_si_eda_2_25_vega.ipynb`

---
