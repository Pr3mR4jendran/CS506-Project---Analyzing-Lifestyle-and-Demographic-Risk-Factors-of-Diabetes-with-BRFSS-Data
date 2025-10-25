# Analyzing Lifestyle and Demographic Risk Factors of Diabetes with BRFSS Data

## Team Members
- Arnav Singh (U19589314)
- Jyoti Shree (U74678990)
- Prem Rajendran (U99248729)
- Saneeya Vichare (U75237907)
- Sara Alsowaimel (U86273437)


## Description of the Project
Our project focuses on predicting diabetes among the U.S. population using structured health and lifestyle data. The dataset we selected, the CDC Behavioral Risk Factor Surveillance System (BRFSS) Dataset, contains survey responses from hundreds of thousands of adults and includes features such as body mass index (BMI), physical activity, general health, and demographic traits.  

Our goal is to use these features to train machine learning models that can predict whether an individual has diabetes or is pre-diabetic. This project also gives us the opportunity to study which health indicators are most strongly associated with diabetes risk, and to explore how lifestyle and demographic factors contribute to outcomes.  

By working with this dataset, we will go through the full data science pipeline:  
- Cleaning and preparing the data  
- Selecting and extracting features  
- Training multiple models  
- Visualizing results  
- Testing performance


## Data Collection Plan 

### Type of Data Needed
Our project used the structured, tabular health data that includes both metabolic and demographic features. Specifically, the following categories of data were used:
- **Metabolic features**: plasma glucose concentration, blood pressure, skinfold thickness, insulin levels, and body mass index (BMI).
- **Demographic features**: age, number of pregnancies, and diabetes pedigree function (a measure of family history risk).
- **Label**: a three class indicator of diabetes diagnosis, where the 3 classes are no diabetes, pre-diabetes, and diabetes.

This data will be used for supervised classification, where the goal is to predict whether an individual has diabetes or pre-diabetes given a set of health related features.

### Chosen Dataset
We relied on the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 dataset, which is publicly available and fully de-identified. BRFSS is one of the largest continuously conducted health surveys in the world, with hundreds of thousands of U.S. adult respondents each year.  

The dataset included a wide range of demographic, lifestyle, and health-related indicators, such as body mass index (BMI), physical activity, smoking, alcohol use, and self-reported health conditions. Importantly, it also contained information on diabetes status, which served as the target label for our classification task.  

By using the BRFSS 2024 dataset, we gained access to a large and diverse sample of the U.S. population. This allowed us to build models that generalize better to real-world populations while also uncovering patterns in diabetes risk across different demographic and lifestyle groups.

### Method of Collection
Since these datasets are publicly available and de-identified, our collection method was limited to:

1. **Dataset Access**: The dataset is publicly available in ASCII and SAS format on the [CDC website](https://www.cdc.gov/brfss/annual_data/annual_2024.html), along with a codebook that contains the schema of the dataset.
2. **Data Cleaning**: The data was converted into a tabular .csv file in Python for easier import into Pandas. Although much of the data was clean, there remain some imbalances. We handled missing or inconsistent values (ex, insulin level = 0, which may indicated missing data rather than actual measurement). Finally, we normalized or standardized features where necessary.
3. **Label Encoding**: Ensure the target variable (diabetes diagnosis) is consistently represented as a 3-class label.
4. **Train-Test Splitting**: Partition the datasets into training and testing subsets to evaluate generalization.

### Justification of Choice
Using the established and openly available BRFSS dataset offers numerous advantages:
- **Reproducibility**: The BRFSS dataset is widely studied in academic and ML contexts, allowing comparison with prior results.
- **Ethical Compliance**: The dataset comes de-identified, avoiding privacy concerns that would arise with collecting new patient data.
- **Practical feasibility**: Collecting new clinical data would require significant resources and patient recruitment, which is not feasible in the scope of this course project.

## Data Structure and Documentation

### Overview

- **Size**: ~475,000 survey responses (rows) across several hundred variables (columns)
- **Unit of Observation**: Each row corresponds to one adult respondent interviewed via telephone
- **Weighting**: Columns _LLCPWT (final weight) and _STRWT (stratum weight) ensure national representativeness.

### Key Target Variable
- DIABETE4 (Diabates Status)
    - 1 = Yes (diagnosed with diabetes)
    - 2 = Yes, but only during pregnancy (gestational)
    - 3 = No
    - 4 = No, pre-diabetes or borderline
    - 7/9 = Don't know / Refused / Missing

This was our outcome variable for modelling/analysis. Most reponses were concentrated towards classes 1, 3 and 4, with only 4,429 responses belonging to other classes. To align with the scope of our project, we dropped all responses that belonged to classes 2, 7 and 9, along with all missing values. 

### Column Types (High-Level Categories)

The variables are grouped into three types:
  1. Demographics: age (AGE5YR), sex (SEXVAR), race/ethnicity (RACEGR3), education (EDUCA), income (INCOME2).
  2. Health Behaviors & Risk Factors: smoking (SMOKE100), alcohol (ALCDAY5), exercise (EXERANY2), diet.
  3. Health Outcomes & Access: BMI (BMI5), blood pressure (BPHIGH4), cholesterol (BLOODCHO), health insurance (HLTHPLN1), personal doctor (PERSDOC2)

Each column in the codebook provides:
- Variable name (used in the dataset).
- Label/Question (survey wording).
- Response codes (including missing categories like 7=Don’t know, 9=Refused).

## Accessing the Data
To fully work with the BRFSS dataset, the following files were needed:

- Data File (LLCP2024.ASC) – The main data file in ASCII format containing all survey responses for 2024. Each line representedone respondent, and each column corresponds to a survey variable.
- Codebook (USCODE24_LLCP_082125.html) – The variable codebook that defined each column in the dataset, including question wording, value labels, and missing code definitions.  

Using these files, the provided Jupyter notebook (BRFSS-2024-Extraction.ipynb) contains code to extract the raw survey data from the ASCII file and process it into an easily readable .csv format for easier viewing and future imports for model training. This .csv file then served as our starting point for data cleaning and analysis.   

## Feature Selection and Extraction Plan
Given that the BRFSS dataset contained hundreds of variables, not all features were relevant for predicting diabetes. We focused on variables that were either directly or indirectly associated with metabolic health and lifestyle behavior.

### Feature Selection
1. **Manual Filtering:** Based on the BRFSS codebook, we first identified the variables that are theoretically linked to diabetes risk (e.g., BMI, physical activity, smoking, age, income, education, blood pressure, cholesterol, and general health status). These features were selected based on a literature review of scholarly research in the field of diabetes.
2. **Missing Values:** We removed invalid or missing entries in the target variable (DIABETE4), and filtered to retain only valid classes (1 = diabetes, 3 = no diabetes, 4 = pre-diabetes). Coded missing values (such as 7, 9, 77, 99, etc.) were detected using regular expression matching and replaced with NaN. Variables with more than 30% missing data were dropped, resulting in 122 retained features.
3. **Correlation Analysis and Chi Square Test:** To assess the correlations between each of the remaining variables and diabetes status (diabetes, pre-diabetes, and no diabetes), a range of statistical analyses were undertaken to assess the strength and significance of these correlations. Categorical variables were analyzed using Chi-Square tests, while continuous variables were tested using ANOVA and, where necessary, the Kruskal–Wallis test as a non-parametric alternative.

    Those with a normalized importance value of 0.6 or higher (tau value of 0.6) are of high importance for modeling. A total of 78 values were found to be of high importance as a key predictor for diabetes diagnosis. The final cleaned dataset has a total of 453,241 rows and 79 columns (78 selected features plus the target variable). 

### Feature Encoding
- **Categorical Variables:** One-hot encoding and ordinal encoding techniques were used depending on the variables. Ordinal variables like educational level and general state of health of a person were assigned numerical values according to their order, while the non-ordinal categorical variables were one-hot encoded.
- **Continuous Variables:** Continuous variables were standardized to ensure all features were on a comparable scale, preventing any variable with larger numerical values from dominating the model.
- Encoding and scaling led to a dataset with a total of 201 features. Further improvement of the feature space was obtained by training an L1 regularized logistic regression model with embedded feature selection to penalize less important coefficients. This method reduced the dimensionality of the space while maintaining interpretability.
- **L1 Regularization:** used to obtained a validation accuracy of 83.6%, selecting to keep 162 of the encoded variables. This dataset of key variables holds the most informative predictors for diabetes diagnosis and was compiled into a final dataset for modeling.
- Two finalized versions of the dataset have been stored under Results folder:
    - BRFSS_2024_model_ready.csv: was used for modeling and analysis, containing encoded and selected features.
    - BRFSS_2024_visualization.csv: was used for primary visualizations, without feature encoding to preserve interpretability for graphical analysis 

All steps were implemented and documented in the Data_Cleaning_Feature_Extraction.ipynb notebook under data-cleaning directory.

## Modeling Plan

Our modeling strategy will employ various supervised machine learning methods, where the focus is to compare a wide range of supervised classification algorithms and identify which model best predicts diabetes status from the BRFSS dataset.

### Baseline Models

We will begin with simpler models to establish a baseline:

- **Multi-Nomial Logistic Regression**
- **Naïve Bayes Classifier**
- **k-Nearest Neighbors (kNN)**
- **Decision Trees**
- **Support Vector Machines (SVM)**

### Advanced Models

To capture non-linear interactions and complex dependencies in the data, we will then implement ensemble and boosting methods:

- **Random Forests**
- **Gradient Boosting Models (XGBoost, LightGBM, CatBoost)**

We speculate that gradient boosting methods will outperform other approaches because of their ability to handle large, tabular, imbalanced datasets while capturing higher-order feature interactions.  

### Model Evaluation

- All models will be trained on the training set and evaluated on the test set.  
- Performance will be compared using:  
  - **Confusion matrix** (to observe misclassification patterns)  
  - **Precision, Recall, and F1 Score** (to balance false positives and false negatives)  
  - **ROC-AUC** (to assess discrimination capability) 
- Among these, the F1 Score will serve as our primary metric since diabetes prediction is a class-imbalanced problem where both false negatives and false positives carry real-world significance.

### Loss Function

For the proposed classification task, we believe that using categorical cross-entropy (log loss) would be ideal, since it penalizes incorrect classifications with stronger weights as prediction confidence increases. To further address imbalance, we will also use weighted cross-entropy loss, where positive and negative classes are assigned different weights based on prevalence. This will ensure that underrepresented classes are not overlooked by the models.

### Model Scope and Practical Constraints
While several algorithms are listed, our focus will be on comparing a smaller subset of well-performing models after initial evaluation. We expect to prioritize models like Logistic Regression, Random Forest, and XGBoost based on baseline performance and interpretability. This will be clarified further once all baseline models are computed.


## Visualization Plan

Visualizations will include both result presentation and primary exploratory data analysis. Since the BRFSS dataset contains health and lifestyle traits for over 400,000 individuals in the U.S., we will use plots to 
(1) understand the distribution of the data and 
(2) visualize how well the model performs


### Exploratory Data Visualizations (before modeling)

1. **US Demographic Heat Map:**
- Purpose: to highlight regional distribution of diabetes risk factors across the U.S.
- This will include a choropleth map showing the prevalence (percentage of respondents within each category) of diabetes and prediabetes in U.S. states.
- The visualization may reflect biological risk shaped by environmental, socioeconomic, and healthcare access factors.

2. **BMI and Age vs. Diabetes Status:**
- Purpose: to examine how weight and age patterns differ by No Diabetes, Prediabetes, and Diabetes.
- This will be represented by box and scatter plots of BMI and age groups by diabetes categories.

3. **Lifestyle Factor Comparisons:**
- Purpose: uncover behavioral risk factors that are potentially predictive of diabetes.
- This will include grouped bar charts for smoking, physical activity, and heavy drinking by diabetes status.
- The plots could depict how behaviors like smoking, inactivity, and heavy drinking exacerbate metabolic stress, accelerating progression from prediabetes to diabetes.


### Result Visualizations (after modeling)

1. **Confusion Matrix Heatmap:**
- Purpose: show which categories the classifier performs well on, and which it gets confused between (e.g., Prediabetes vs. Diabetes).
- In this matrix, the rows will be true classes and the columns will be predicted classes.
- Following that, the normalized values will be displayed as a heatmap for readability.

2. **ROC Curves / Precision-Recall Curves:**
- Purpose: test model discrimination capacity, especially for imbalanced classes.
- This would include one curve per category of diabetes using a One-vs-Rest approach.
- It will demonstrate how well biological risk factors enable discrimination between healthy, prediabetic, and diabetic states.


### Summary:
Collectively, these plots will deliver:
- A clear view of geographic, demographic, and lifestyle trends in the BRFSS dataset.
- Intuitive representations of model performance (confusion matrix, ROC/PR).


## Test Plan

To ensure reliable model evaluation and prevent overfitting, we will follow a systematic testing strategy.

We plan to divide the data into a training set (80% ~ 360,000 observations) which will be used for model fitting and hyperparameter tuning, and a test set (20% ~ 90,000 observations) which will be used to evaluate the generalization ability of the final model.
The 80/20 split is chosen as it provides enough data for the models to be able to find complex patterns, and still retains a very large, statistically significant remainder for us to use as a test set. With over 400,000 individuals in BRFSS, even 20% makes for a robust test set.
