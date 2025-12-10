# Analyzing Lifestyle and Demographic Risk Factors of Diabetes with BRFSS Data

## Team Members
- Arnav Singh (U19589314)
- Jyoti Shree (U74678990)
- Prem Rajendran (U99248729)
- Saneeya Vichare (U75237907)
- Sara Alsowaimel (U86273437)

Midterm Report YouTube Video: https://youtu.be/Qgp6xDooT5o
Final Report YouTube Video:


## Getting Started

### Project Structure
- `Code/`
  - `data-extraction/`: scripts and notebooks to pull and preprocess BRFSS 2024 data  
    - `download_gdrive_folder.py` – downloads all required project data from Google Drive  
    - `BRFSS-2024-Extraction.ipynb` – optional re-extraction from the raw ASCII files
  - `data-cleaning/`: data cleaning, feature engineering, and hyperparameter tuning  
    - `Data_Cleaning_Feature_Extraction.ipynb` – main data cleaning + feature extraction pipeline  
    - `hyperparam_tuning.py` – grid search over Missing Rate Threshold (`mrt`), statistical importance threshold (`tau`), and L1 regularization strength (`C`)
  - `models/`: notebooks to train and evaluate all model variants
  - `data-visualization/`: notebooks to generate all figures and visual summaries
- `Data/`: raw and processed datasets (populated by the extraction and download scripts)
- `Results/`: saved model outputs, metrics, and visualizations
- `Makefile`: installs dependencies and downloads required data
- `requirements.txt`: Python dependencies for the project

### Reproducing the Pipeline
1. **Install dependencies & download data**  
   From the project root, run:
      `make`
   This installs all packages from `requirements.txt` and downloads the required data via `Code/data-extraction/download_gdrive_folder.py`.

2. **(Optional) Re-extract BRFSS 2024 from raw ASCII**  
   Open and run `Code/data-extraction/BRFSS-2024-Extraction.ipynb` to regenerate the processed datasets from the original ASCII files.

3. **(Optional) Re-run data cleaning & feature extraction**
   - Run `Code/data-cleaning/Data_Cleaning_Feature_Extraction.ipynb` to perform data cleaning and feature extraction.
   - To explore different values of `mrt`, `tau`, and `C`, edit the hyperparameter grid in `Code/data-cleaning/hyperparam_tuning.py` and run:
        `python Code/data-cleaning/hyperparam_tuning.py`
   
   **Note:** this grid search is computationally expensive and may take a long time.

4. **Reproduce model results**  
   Run each notebook in `Code/models/` to train and evaluate the models. These notebooks will reproduce the metrics and artifacts reported in the project.

5. **Reproduce visualizations**  
   Run each notebook in `Code/data-visualization/` to regenerate all plots and visual summaries used in the analysis.


## Description of the Project
Our project focuses on predicting diabetes among the U.S. population using structured health and lifestyle data. The dataset we selected, the CDC Behavioral Risk Factor Surveillance System (BRFSS) Dataset, contains survey responses from hundreds of thousands of adults and includes features such as body mass index (BMI), physical activity, general health, and demographic traits.  

Our goal is to use these features to train machine learning models that can predict whether an individual has diabetes or is pre-diabetic. This project also gives us the opportunity to study which health indicators are most strongly associated with diabetes risk, and to explore how lifestyle and demographic factors contribute to outcomes.  

By working with this dataset, we will go through the full data science pipeline:  
- Cleaning and preparing the data  
- Selecting and extracting features  
- Training multiple models  
- Visualizing results  
- Testing performance


## Data Collection 

### Type of Data 
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

### Rationale
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

## Feature Selection and Extraction 
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

## Pre-liminary Visualizations

### Demographics vs Diabetes
<img width="5366" height="3787" alt="demographics_vs_diabetes" src="https://github.com/user-attachments/assets/a25151e3-e4be-429d-8506-19601878461d" />

Across all demographic and lifestyle variables, individuals with diabetes (red) are consistently overrepresented in older age groups, lower education categories, lower income brackets, and those not currently employed, highlighting strong socioeconomic and age-associated risk patterns. Race also shows notable disparities, with certain race categories showing higher diabetic counts relative to their group size. Respondents without healthcare coverage (HLTHPL2 = 2) have a visibly elevated proportion of diabetes cases, supporting the link between poor healthcare access and unmanaged metabolic risk. Overall, the distribution demonstrates that diabetes prevalence is not uniform — it clusters heavily around age, socioeconomic disadvantage, and reduced healthcare access, aligning with known public-health patterns.

### Diabetes Distribution Plot
<img width="2322" height="1429" alt="diabetes_distribution_plot" src="https://github.com/user-attachments/assets/f6dfd004-d8de-4139-9b68-ac94c402c5b9" />

The dataset is heavily imbalanced: 83% of respondents report no diabetes, while only 14.5% report a diabetes diagnosis and 2.5% report pre-diabetes. This extreme skew means that without correction, models will naturally default toward predicting the majority class. The small size of the pre-diabetes group in particular highlights the need for class weighting or resampling to ensure clinically meaningful minority-class detection. Overall, this distribution confirms that diabetes prediction in BRFSS is an imbalanced classification problem, shaping all later modeling choices.

### Feature Retention Bar Graph
<img width="2031" height="1900" alt="feature_retention_bar" src="https://github.com/user-attachments/assets/a0db6575-6a66-4624-8153-83eedb2cb290" />

The feature count drops from 301 to 77, showing that preprocessing removed a large number of variables with excessive missingness, redundancy, or little relevance to diabetes prediction. This reduction improves model efficiency by eliminating noise and preventing overfitting. Keeping only higher-quality variables ensures that downstream models learn from features with real predictive value, rather than being diluted by poorly populated or uninformative fields.

### Health vs Diabetes Bar Graph
<img width="6075" height="4714" alt="health_vs_diabetes" src="https://github.com/user-attachments/assets/1cd68265-9bdf-4356-8e83-b13d0de2f65d" />

Diabetes prevalence (red) is visibly higher among individuals with elevated BMI categories, weekly inactivity, and those reporting chronic conditions like stroke, asthma, or heart disease. Smokers and heavy drinkers also show disproportionately more diabetes cases relative to their group sizes, reflecting well-established metabolic risk patterns. Exercise variables (EXERANY2, _TOTINDA) show the opposite trend—those who report no regular physical activity contain a higher share of diabetic respondents. Overall, these patterns reinforce that diabetes strongly co-occurs with obesity, inactivity, cardiovascular burden, and other lifestyle-related risk factors.

### Missing Value Correlation Heatmaps
<img width="4255" height="2234" alt="missing-value_correlation_heatmap" src="https://github.com/user-attachments/assets/0c8000a0-eb76-4bd5-96d0-957c098a8bae" />

Before cleaning, many variables show strong, clustered missingness correlations, meaning entire groups of features tended to be missing together—an indication of structural survey gaps or low-response sections. After cleaning, these dense blocks largely disappear, and most variables exhibit weak or near-independent missingness, showing that preprocessing successfully removed problematic or sparsely populated fields. This shift indicates a healthier dataset: missingness is no longer driving artificial relationships between variables, reducing bias and improving the reliability of downstream modeling.

### Missingness Pie Charts
<img width="4332" height="2600" alt="missingness_pie_chart" src="https://github.com/user-attachments/assets/4d2fa042-6e5e-4e31-8a70-96aaf9974858" />

Before cleaning, nearly half of all entries (48.4%) were missing or invalid, showing that the raw BRFSS dataset contains extensive non-response and unusable codes. After cleaning, the proportion of valid data surges to 89.1%, with all forms of invalid or ambiguous responses reduced to small single-digit percentages. This demonstrates that preprocessing successfully removed low-quality records and standardized missingness, resulting in a far more reliable and analysis-ready dataset.

### Health & Behaviors Radar Chart
<img width="3036" height="2402" alt="radar_chart" src="https://github.com/user-attachments/assets/cb121ac6-fea3-454a-9b01-d3740053f4d4" />

Across nearly every health domain, the diabetic group (red) shows higher rates of chronic conditions and risk factors than the borderline/pre-diabetic group (yellow). Diabetic respondents exhibit markedly greater prevalence of heart disease, poor mental health days, and higher BMI categories, all aligning with known metabolic comorbidities. They also show slightly reduced physical activity and higher rates of smoking, while alcohol use is lower—consistent with clinical patterns where diagnosed individuals often reduce alcohol consumption. Overall, the radar chart highlights a clear escalation of health burdens as individuals progress from pre-diabetic to diabetic status.

## Modeling 

Our modeling strategy employed various supervised machine learning methods, where the focus is to compare a wide range of supervised classification algorithms and identify which model best predicts diabetes status from the BRFSS dataset.

### Tuned Models

We began with four baseline classifiers to establish performance benchmarks:

- **Multi-Nomial Logistic Regression**
- **Naïve Bayes Classifier**
- **k-Nearest Neighbors (kNN)**
- **Decision Trees**

Each model was trained on the same training–testing split and evaluated using multiple metrics to ensure comparability. Then we tuned each model, whose results are shown below. 

| Model                | Accuracy  | Precision (Macro) | Recall (Macro)  | F1 Score (Macro) | Log Loss |
|----------------------|-----------|------------------:|----------------:|-----------------:|---------:|
| Naïve Bayes          | 0.4594    | 0.4129            | 0.4754          | 0.3420           | 8.1898   |
| Decision Tree        | 0.7772    | 0.4087            | 0.4072          | 0.4057           | NA       | 
| kNN (Euclidean)      | 0.3921    | 0.4223            | 0.4706          | 0.3142           | 1.1728   |
| kNN (Manhattan)      | 0.6171    | 0.4270            | 0.4999          | 0.4103           | 0.8575   |
| Logistic Regression  | 0.6043    | 0.4395            | 0.5303          | 0.4160           | 0.9061   |

Naïve Bayes achieved modest performance (accuracy = 0.69) but exhibited high bias toward the majority class (label 3). While simple and computationally efficient, it failed to capture complex dependencies between features. Decision Tree improved accuracy (0.75) but still struggled with minority classes, often overfitting to dominant patterns. kNN (Euclidean and Manhattan) achieved strong overall accuracy (~0.83) but severely underperformed on minority classes, with near-zero recall for class 1 and 4. This suggests strong class imbalance effects, where the majority class dominates nearest-neighbor voting. Logistic Regression produced the highest overall performance among baseline models, with accuracy = 0.8360 and the lowest log loss = 0.4329. Although minority-class recall remained low, the model balanced interpretability, stability, and probabilistic calibration better than other baselines.

Across all models, the dominance of class 3 (non-diabetic) in the dataset led to skewed predictions and poor detection of diabetic (class 1) and borderline (class 4) cases. We plan to explicitly handle class imbalance using oversampling or undersampling methods.

### Advanced Models

To capture non-linear interactions, complex dependencies and address the imbalance problem in the data, the next phase will focus on ensemble and boosting methods:

- **Support Vector Machines (SVM)**
- **Random Forests**
- **Gradient Boosting Models (XGBoost, LightGBM, CatBoost)**

We speculate that gradient boosting methods will outperform other approaches because of their ability to handle large, tabular, imbalanced datasets while capturing higher-order feature interactions.  

| Model                | Accuracy  | Precision (Macro) | Recall (Macro)  | F1 Score (Macro) | Log Loss |
|----------------------|-----------|------------------:|----------------:|-----------------:|---------:|
| Random Forest        | 0.8364    | 0.4683            | 0.3806          | 0.3869           | 0.4508   |
| SVM                  | 0.6171    | 0.4390            | 0.5309          | 0.4203           | NA       | 
| XGBoost              | 0.8320    | 0.4879            | 0.3995          | 0.4096           | 0.4458   |


### Model Evaluation

- All models were trained on the training set and evaluated on the test set.  
- Performance will be compared using:  
  - **Confusion matrix** (to observe misclassification patterns)  
  - **Precision, Recall, and F1 Score** (to balance false positives and false negatives)  
  - **ROC-AUC** (to assess discrimination capability) 
  - **Log-Loss** (to measure probabilistic calibration)
- Among these, the F1 Score remains the primary comparison metric, given the imbalanced nature of the dataset and the clinical significance of both false positives and false negatives.

### Loss Function

For this classification task, categorical cross-entropy (log-loss) is the most suitable objective, as it heavily penalizes incorrect high-confidence predictions. To further mitigate class imbalance, future models will use weighted cross-entropy, assigning higher penalties to underrepresented classes based on class frequencies.


## Testing 

To ensure reliable model evaluation and prevent overfitting, we will follow a systematic testing strategy.

We plan to divide the data into a training set (80% ~ 360,000 observations) which will be used for model fitting and hyperparameter tuning, and a test set (20% ~ 90,000 observations) which will be used to evaluate the generalization ability of the final model.
The 80/20 split is chosen as it provides enough data for the models to be able to find complex patterns, and still retains a very large, statistically significant remainder for us to use as a test set. With over 400,000 individuals in BRFSS, even 20% makes for a robust test set.


## Visualization 

Visualizations will include both result presentation and primary exploratory data analysis. Since the BRFSS dataset contains health and lifestyle traits for over 400,000 individuals in the U.S., we will use plots to 
(1) understand the distribution of the data and 
(2) visualize how well the model performs


### Exploratory Data Visualizations (before modeling)

1. **Data Cleaning Assessment:**
- Purpose: to measure the frequency and correlation of missing data, and learn the impact of preprocessing on dataset quality.
- Heatmaps (missingness correlation before vs. after cleaning), Pie Charts (composition of valid vs. missing data), and a Bar Plot (feature retention count).
- Interpretation: Cleaning reduced correlated missingness, validated responses from ~48% to 89%, and removed duplicate variables (301 → 79) to create a sound basis for modeling.

2. **Target Exploration:**
- Purpose: To examine the overall class distribution of diabetes-related responses and detect any imbalance.
- A Categorical Bar Plot with percentages of respondents having and not having diabetes, including borderline/prediabetes.
- Interpretation: The data showed a strong class imbalance, with most respondents reporting “No diabetes,” guiding later model balancing strategies.
  
3. **Demographic and Lifestyle Factors vs. Diabetes:**
- Purpose: To explore the correlation between diabetes rate and behavioral and demographic characteristics.
- A faceted bar plot grid compared diabetes status by factor such as age, race, income, education, employment, and healthcare access.
- Interpretation: Stark demographic and socioeconomic disparities were observed — older individuals, lower-income groups, and those with poor healthcare access showed higher diabetes rate.


### Result Visualizations (after modeling)

1. **Confusion Matrix Heatmap:**
- Purpose: show which categories the classifier performs well on, and which it gets confused between (e.g., Prediabetes vs. Diabetes).
- In this matrix, the rows will be true classes and the columns will be predicted classes.
- Following that, the normalized values will be displayed as a heatmap for readability.

2. **ROC Curves / Precision-Recall Curves:**
- Purpose: test model discrimination capacity, especially for imbalanced classes.
- This would include one curve per category of diabetes using a One-vs-Rest approach.
- It will demonstrate how well biological risk factors enable discrimination between healthy, prediabetic, and diabetic states.

### Interactive Plots:

- [**Naive Bayes: Probability Plot**](https://raw.githack.com/Pr3mR4jendran/CS506-Project---Analyzing-Lifestyle-and-Demographic-Risk-Factors-of-Diabetes-with-BRFSS-Data/main/docs/NB_plot.html)
- [**SVM: PCA Visualization (Correct vs Misclassified)**](https://raw.githack.com/Pr3mR4jendran/CS506-Project---Analyzing-Lifestyle-and-Demographic-Risk-Factors-of-Diabetes-with-BRFSS-Data/main/docs/SVM_plot.html)
- [**Logistic Regression: Probability Space Visualization**](https://raw.githack.com/Pr3mR4jendran/CS506-Project---Analyzing-Lifestyle-and-Demographic-Risk-Factors-of-Diabetes-with-BRFSS-Data/main/docs/LR_plot.html)



### Summary:
Collectively, these plots will deliver:
- A clear view of geographic, demographic, and lifestyle trends in the BRFSS dataset.
- Intuitive representations of model performance (confusion matrix, ROC/PR).
