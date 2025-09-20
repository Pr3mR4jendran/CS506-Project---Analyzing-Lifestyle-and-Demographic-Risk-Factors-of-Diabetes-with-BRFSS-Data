# CS506 Project Proposal (9/22)

## Data Collection Plan 

### Type of Data Needed
Our project requires **retinal fundus** images—photographic scans of the interior surface of the eye, including the retina, optic disc, macula, and posterior pole. 
These images are widely used in ophthalmology to detect conditions such as diabetic retinopathy (DR), glaucoma, and age-related macular degeneration.
Beyond eye health, prior research has shown that deep learning models trained on fundus images can predict broader systemic health risk factors such as age, gender, smoking history, blood pressure, and even cardiovascular risk.

For this project, we will focus on the specific healthcare problem of diabetic retinopathy detection. DR is a progressive eye disease caused by diabetes that can lead to blindness if untreated, and automated detection from fundus images is a well-established and clinically relevant machine learning task.

### Chosen Dataset
We will use a consolidated dataset available on Kaggle that combines three well-known retinal imaging datasets: EyePACS, APTOS, and Messidor. Together, these datasets provide a diverse and representative collection of labeled fundus images, with severity labels for diabetic retinopathy.

- **EyePACS**: Contains ~35,000 high-resolution fundus images labeled across five stages of DR severity (no DR, mild, moderate, severe, proliferative). Originally used in the 2015 Kaggle Diabetic Retinopathy Detection competition.

- **APTOS 2019**: Curated by the Asia Pacific Tele-Ophthalmology Society for a Kaggle competition. It provides labeled DR images with strong quality control.

- **Messidor**: A smaller but highly cited dataset of retinal images, supported by the French Ministry of Research and Defense within the 2004 TECHNO-VISION program.

By using the combined Kaggle dataset, we ensure both breadth (multiple sources, improving generalizability) and depth (thousands of images with consistent DR annotations).

### Method of Collection
The dataset has already been publicly curated and made accessible through Kaggle. Our collection method will therefore consist of:

1. **Dataset Access**: Downloading the Kaggle dataset [here](https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy). This provides standardized train/test splits and metadata (DR labels, image IDs, and severity levels).
2. **Preprocessing**: Images will be cleaned and resized to a consistent resolution. Quality-control preprocessing steps such as image normalization, contrast enhancement, and cropping around the fundus will be performed, as recommended in prior DR studies.
3. **Augmentation**: To address class imbalance (ex, fewer severe DR cases compared to normal), data augmentation techniques such as horizontal flips, random rotations, and brightness adjustments will be applied.
4. **Label Utilization**: The provided categorical DR labels (ranging from no DR to proliferative DR) will serve as ground truth for supervised training.

### Justification of Choice
Using a curated Kaggle dataset is preferable to attempting new data collection (ex, scraping medical images) for several reasons:
- **Ethical and privacy considerations**: Fundus images are classified as medical data. Using publicly available, de-identified datasets avoids privacy risks.
- **Clinical validity**: EyePACS, APTOS, and Messidor are widely used in ophthalmology AI research, ensuring our results are comparable to prior work.
- **Practical feasibility**: Collecting new fundus scans would require specialized equipment and patient consent, which is beyond the scope of a semester-long course project.