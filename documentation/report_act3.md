# Activity 3 – Data Splitting and Final Model

## Overview
*The following section applies K-Fold and Stratified K-Fold to the dataset as required by the assignment. While the core objective of my project is a TF-IDF-based recommendation system that does not rely on supervised labels or explicit train/test splits, this analysis reframes the problem as a classification task to evaluate how much information about `mainaccord1` can be extracted from the combined note text.*

## Splitting Techniques
1. **K-Fold Cross-Validation**  
   - **Description:** Partition the dataset into *k* equal folds and rotate the test fold to estimate generalization.  
   - **Typical Use:** Balanced evaluation when data size is limited and no temporal ordering is required.  
   - **Python Library:** `sklearn.model_selection.KFold`.

2. **Stratified K-Fold Cross-Validation**  
   - **Description:** Ensures each fold preserves the label distribution (the dominant accord `mainaccord1`).  
   - **Typical Use:** Classification tasks with uneven class frequencies to avoid biased folds.   
   - **Python Library:** `sklearn.model_selection.StratifiedKFold` 

## Experimental Setup
- **Algorithm:** K-Nearest Neighbors classifier using TF-IDF vectors and cosine similarity (`n_neighbors=5`, `metric='cosine'`).
- **Target Label:** `mainaccord1` (dominant scent accord).
- **Features:** Precomputed `features` text field combining notes and accord descriptors, vectorized with `TfidfVectorizer(stop_words='english')`.
- **The train_test:** Function encapsulates the core evaluation logic used in each fold. It receives the training and testing splits, converts them into TF-IDF vectors, trains the KNN model with cosine similarity, and returns the accuracy for that fold. This allows both K-Fold and Stratified K-Fold to reuse the same procedure, keeping the evaluation consistent across all splits.
- **Baseline Without Splitting:** Majority-class predictor -> represents the simplest possible model always predicting the most frequent mainaccord1 in the dataset.

## Results
| Configuration | Accuracy |
| --- | --- |
| Majority-class baseline (no split) | 0.153 |
| 5-Fold Cross-Validation | 0.398 |
| Stratified 5-Fold Cross-Validation | 0.403 |

The majority-class baseline established a fundamental reference point with an accuracy of 0.153, strictly reflecting the prevalence of the most frequent accord in the dataset. This baseline serves as a necessary lower bound to verify that the supervised models are discerning meaningful patterns rather than relying on chance.
Both cross-validation methods performed substantially better. K-Fold Cross-Validation reached 0.398, demonstrating that the combined note text contains useful signal for predicting mainaccord1.
Stratified K-Fold achieved the highest accuracy (0.403) by preserving class proportions in every split, which prevented rare accords from being underepresented during training and testing. This makes it the most stable and balanced evaluation method among the three.



## Data Splitting Results Conclusion
Applying data-splitting techniques such as K-Fold and Stratified K-Fold produced higher accuracy compared to the majority class baseline, showing that the text features contain meaningful information for predicting `mainaccord1`.

I chose the Stratified K-Fold KNN model as the best option because it achieved the highest mean accuracy (0.403) and better preserved the class distribution of `mainaccord1`, making it more reliable.

However, these results apply only within the supervised classification framework required for the assignment and are not meaningful for the actual recommendation system, which is fully similarity-based and does not rely on label prediction or data-splitting techniques.

---

## Final Model Conclusion and Future Steps
Overall, the final TF-IDF + KNN recommendation model provides a solid and reliable content-based framework for identifying fragrance similarity through note composition and semantic patterns. By converting each perfume into a TF-IDF vector and comparing these representations with cosine similarity, the system produces recommendations that reflect real olfactory relationships rather than simple textual overlap or categorical labels. This makes the model well aligned with the practical goal of guiding users toward perfumes that genuinely resemble the aromatic structure of their preferences.

To further strengthen the system, improvements can focus on refining text preprocessing, standardizing note terminology, and enriching the dataset with more complete fragrance descriptions. These enhancements would increase the precision of the TF-IDF representations and further optimize the relevance of the recommendations—while maintaining the same algorithmic approach, which has already demonstrated strong consistency with the objectives and requirements of this project.


## References

*Scikit-learn Documentation – K-Fold.*
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

*Scikit-learn Documentation – StratifiedKFold.*
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html