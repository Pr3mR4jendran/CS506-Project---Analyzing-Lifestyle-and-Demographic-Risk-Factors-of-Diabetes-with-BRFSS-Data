# CS506 Project Proposal (9/22)

## Data Collection Plan 

### Type of Data Needed
Our project requires **structured, tabular health data** that includes both metabolic and demographic features. Specifically, the dataset must contain indicators such as:
- **Metabolic features**: plasma glucose concentration, blood pressure, skinfold thickness, insulin levels, and body mass index (BMI).
- **Demographic features**: age, number of pregnancies, and diabetes pedigree function (a measure of family history risk).
- **Label**: a single binary indicator of diabetes diagnosis (yes/no).

This type of data is well suited for **supervised classification**, where the goal is to predict whether an individual has diabetes given a set of health related features. Models such as logistic regression, random forests, or gradient boosting are appropriate for this dataset.

### Chosen Dataset
We will rely on publicly available, de-identified datasets that are widely used in epidemiological and machine learning studies.

- **Pima Indian Diabetes Dataset (PID)**: A benchmark dataset with **768 samples** of Pima Indian women, collected by the National Institute of Diabetes and Digestive and Kidney Diseases. It includes all the metabolic and demographic features listed above, with a binary diabetes outcome label

- **Diabetes Health Indicators Dataset**: A larger dataset curated from the CDC Behavioral Risk Factor Surveillance System (BRFSS), encompassing responses from U.S. adults about health status, lifestyle, and risk factors. This dataset expands beyond Pima by providing more recent and diverse population data.

By combining insights from both datasets, we can balance the controlled, smaller-scale benchmark dataset (PID) with the larger, population-level dataset (BRFSS).

### Method of Collection
Since these datasets are publicly available and de-identified, our collection method will be limited to:

1. **Dataset Access**: Download the Pima Indian Diabetes Dataset via Kaggle [here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data). Download the Diabetes Health Indicators Dataset from Kaggle [here](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).
2. **Data Cleaning**: Though much of the data is clean, there remain imbalances. We would handle missing or inconsistent values (ex, insulin level = 0, which may indicate missing data rather than actual measurement). Normalize or standardize features where necessary.
3. **Label Encoding**: Ensure the target variable (diabetes diagnosis) is consistently represented as a binary label across datasets.
4. **Train-Test Splitting**: Partition the datasets into training and testing subsets to evaluate generalization. If combining datasets, we will clearly separate by source to avoid leakage.

### Justification of Choice
Using established and openly available datasets offers numerous advantages:
- **Reproducibility**: Both PID and BRFSS-derived datasets are widely studied in academic and ML contexts, allowing comparison with prior results.
- **Ethical Compliance**: The datasets come de-identified, avoiding privacy concerns that would arise with collecting new patient data.
- **Practical feasibility**: Collecting new clinical data would require significant resources and patient recruitment, which is not feasible in the scope of this course project.




## Visualization Plan

Visualizations will include both result presentation and primary exploratory data analysis. Since the BRFSS dataset contains health and lifestyle traits for over 200,000 individuals in the U.S., we will use plots to 
(1) understand the distribution of the data and 
(2) visualize how well the model performs


### Exploratory Data Visualizations (before modeling)

  1. US Demographic Heat Map
    - Purpose: to highlight regional distribution of diabetes risk factors across the U.S.
    - This will include a choropleth map showing the prevalence (percentage of respondents within each category) of diabetes and prediabetes in U.S. states.
    - The visualization may reflect biological risk shaped by environmental, socioeconomic, and healthcare access factors.

  2. BMI and Age vs. Diabetes Status
    - Purpose: to examine how weight and age patterns differ by No Diabetes, Prediabetes, and Diabetes.
    - This will be represented by box and scatter plots of BMI and age groups by diabetes categories.

  3. Lifestyle Factor Comparisons
    - Purpose: uncover behavioral risk factors that are potentially predictive of diabetes.
    - This will include grouped bar charts for smoking, physical activity, and heavy drinking by diabetes status.
    - The plots could depict how behaviors like smoking, inactivity, and heavy drinking exacerbate metabolic stress, accelerating progression from prediabetes to diabetes.


### Result Visualizations (after modeling)

  1. PCA or t-SNE Scatterplot (Unsupervised Learning)
    - Purpose: visualize whether lifestyle and demographic characteristics tend to cluster naturally according to diabetes outcomes.
    - These would allow dimensionality reduction of the data to 2D.
    - The plots could essentially have two panels: one colored by actual diabetes labels and one colored by unsupervised groups (e.g., KMeans with k=3).
    - This would reflect on novel relational patterns between different dataset attributes

  2. Confusion Matrix Heatmap (Supervised Learning)
    - Purpose: show which categories the classifier performs well on, and which it gets confused between (e.g., Prediabetes vs. Diabetes).
    - In this matrix, the rows will be true classes and the columns will be predicted classes.
    - Following that, the normalized values will be displayed as a heatmap for readability.

  3. ROC Curves / Precision-Recall Curves (Supervised Learning)
    - Purpose: test model discrimination capacity, especially for imbalanced classes.
    - This would include one curve per category of diabetes using a One-vs-Rest approach.
    - It will demonstrate how well biological risk factors enable discrimination between healthy, prediabetic, and diabetic states.


### Summary:
Collectively, these plots will deliver:
  - A clear view of geographic, demographic, and lifestyle trends in the BRFSS dataset.
  - Intuitive representations of model performance (confusion matrix, ROC/PR).
  - Uncover the underlying data structure with unsupervised methods (PCA/t-SNE).
