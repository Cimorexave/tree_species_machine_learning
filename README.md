# Predictive Modeling of Urban Tree Species via Morphological Traits
**Author:** Sina Sadeghi

## 📝 Abstract
This experiment investigates whether a machine learning model can accurately identify tree species based on physical growth metrics (height, circumference, and age). Utilizing open data from the City of Vienna, we train a Random Forest classifier to categorize trees into their respective genera. [cite_start]This study simulates a real-world scenario of automated environmental mapping. [cite: 19, 28, 42, 43]

## 📊 Input Data Source
[cite_start]The dataset is retrieved from **[data.gv.at](https://www.data.gv.at/)**, a certified national open data portal. [cite: 25] [cite_start]It consists of the **"Baumkataster Wien"** (Tree Inventory), which provides real-world measurements. [cite: 21]
* [cite_start]**Compliance:** This experiment uses existing data from a certified national repository, fulfilling the requirements of Part 1 of the FAIR Data Science exercise. [cite: 23, 25, 48]
* [cite_start]**Rights:** Data is used under open government licenses with the full right to reuse for academic purposes. [cite: 22]

## ⚙️ Implementation
[cite_start]The project is implemented in **Python** using `pandas` for data handling and `scikit-learn` for modeling. [cite: 32, 45]
* [cite_start]**Data Split:** The dataset was cleaned and partitioned into **80% training, 10% testing, and 10% validation sets**. [cite: 21]
* [cite_start]**Model:** We employed a **Random Forest Classifier** to handle the non-linear relationships between growth rate and species. [cite: 28]

### Data Flow Diagram
[cite_start]`Raw Data (data.gv.at) -> Preprocessing -> Train/Test/Val Split -> Random Forest Training -> Evaluation` [cite: 47]

## 📈 Results
[cite_start]The experiment produced a Random Forest model with classification accuracy visualized in the outputs below. [cite: 29, 46]

1. [cite_start]**Confusion Matrix:** The model is most successful at identifying *Acer campestre* (Field Maple), while some confusion exists between *Tilia cordata* (Small-leaved Lime) and *Fraxinus excelsior* (Common Ash) due to overlapping morphological traits. [cite: 29]
2. [cite_start]**Histogram:** The distribution confirms a high density of trees in the **10–20m** height range, providing insight into the urban forest structure. [cite: 29]

| Output File | Description |
| :--- | :--- |
| `tree_height_histogram.png` | [cite_start]Distribution of tree heights in the dataset. [cite: 29] |
| `confusion_matrix.png` | [cite_start]Species prediction accuracy and error rates. [cite: 29] |

---