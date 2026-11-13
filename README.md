# Fake Review Detection System

An AI-based Fake Review Detection System that uses Machine Learning and Web Scraping to identify suspicious product reviews from e-commerce platforms.

## Project Overview

Online shopping platforms contain thousands of customer reviews that influence purchasing decisions. However, some reviews may be misleading, manipulated or artificially generated to influence product ratings.

The **Fake Review Detection System** aims to identify potentially fake or suspicious product reviews using Machine Learning techniques.

The application extracts customer reviews from product pages, processes the extracted information and classifies each review as either **Suspicious** or **Genuine**.

It also provides an analysis summary with review statistics and a visual representation of the prediction results.

## Key Features

- **Multi-Platform Support:** Extracts product reviews from Amazon, Flipkart, Myntra and Meesho.
- **Web Scraping:** Uses Python Requests and BeautifulSoup to extract available product reviews.
- **Machine Learning Classification:** Uses TF-IDF and machine learning algorithms to classify reviews.
- **Model Comparison:** Evaluates Naive Bayes, Logistic Regression and Linear Support Vector Machine (SVM).
- **Review Metadata Extraction:** Extracts review title, rating, verified purchase status and product category wherever available.
- **Analysis Dashboard:** Displays total reviews, suspicious reviews, genuine reviews and suspicious review percentage.
- **Visual Representation:** Displays a pie chart showing the distribution of suspicious and genuine reviews.
- **Individual Review Analysis:** Displays the prediction and available details for each extracted review.
- **User-Friendly Interface:** Provides a desktop application developed using Tkinter.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| GUI | Tkinter |
| Machine Learning | Scikit-learn |
| Text Processing | TF-IDF |
| Web Scraping | Requests, BeautifulSoup |
| Data Processing | Pandas, NumPy |
| Model Storage | Joblib |
| Visualization | Tkinter Canvas |
| Version Control | Git and GitHub |

## Machine Learning Approach

The system uses Natural Language Processing (NLP) and supervised Machine Learning techniques.

### 1. Dataset

This project uses an Amazon product reviews research dataset containing 21,000 labelled reviews across 30 product categories.

The dataset includes review text, review title, rating, verified purchase status and product category.

**Original Dataset Repository:** [Deception Detection on Amazon Reviews Dataset](https://github.com/aayush210789/Deception-Detection-on-Amazon-reviews-dataset)

### Dataset Setup

The dataset is not included in this repository because its redistribution permissions could not be verified.

To run model training:

1. Visit the original dataset repository.
2. Review its usage and redistribution terms.
3. Obtain the `amazon_reviews.txt` file if permitted.
4. Place the file inside the project's `data/` directory.

Expected location:

`data/amazon_reviews.txt`

The dataset is used for academic and educational research purposes.

### 2. Data Preprocessing

The following preprocessing steps are performed:

- Loading the dataset.
- Mapping review labels into binary classes.
- Combining review titles and review text.
- Handling missing values.
- Preparing categorical and numerical features.
- Converting textual information into numerical features using TF-IDF.

### 3. Feature Extraction

The model uses the following features:

- **Review Text:** TF-IDF features generated from the combined review title and body.
- **Verified Purchase:** Purchase verification information.
- **Product Category:** Product category information.
- **Rating:** Numerical review rating.

### 4. Model Training

Three machine learning algorithms are evaluated:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (SVM)

The dataset is divided into training and testing sets using an 80:20 stratified split.

The model is selected based on the F1-score of the suspicious class.

### 5. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Model Performance

The following are the recorded results from the 80:20 held-out test split:

| Algorithm | Accuracy | Fake Class F1-Score |
|---|---:|---:|
| Naive Bayes | 80.24% | 79.49% |
| Logistic Regression | 80.95% | 80.07% |
| Linear SVM | 80.83% | 80.32% |

**Selected Model: Linear Support Vector Machine (SVM)**

The Linear SVM model was selected because it achieved the highest F1-score for the suspicious review class.

Its recorded evaluation results were:

- Accuracy: 80.83%
- Precision: 82.52%
- Recall: 78.24%
- F1-Score: 80.32%

These results represent performance on the held-out test portion of the research dataset. They do not guarantee equivalent performance on reviews collected from other e-commerce platforms.

## System Architecture

The application follows the workflow below:

1. The user enters a product URL.
2. The application validates the URL.
3. The web scraper extracts available product reviews and metadata.
4. The extracted reviews are passed to the prediction module.
5. The trained ML model classifies each review.
6. The application calculates summary statistics.
7. The results are displayed through the Tkinter interface.

## Project Structure

```text
fake-review-detection/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── sample_reviews.csv
│   └── amazon_reviews.txt
│
├── model/
│   ├── best_model_metadata.pkl
│   ├── model_metrics_metadata.json
│   ├── best_model.pkl
│   └── model_metrics.json
│
├── src/
│   ├── __init__.py
│   ├── scraper.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── utils.py
│
├── test_prediction.py
├── test_scraper.py
├── test_scraped_prediction.py
└── test_model_dataset.py
```

*Note: The structure above represents the intended project organisation. Include only files that are actually present in your repository.*

## Installation and Setup

### Prerequisites

- Python 3.9 or later.
- Git.
- Visual Studio Code or another Python-compatible IDE.

### 1. Clone the Repository

```bash
git clone https://github.com/PurvaPatil210/fake-review-detection.git
```

### 2. Navigate to the Project Directory

```bash
cd fake-review-detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

The Tkinter desktop application will open.

### 7. Analyze Product Reviews

1. Copy a product URL from a supported e-commerce platform.
2. Paste the URL into the application.
3. Click **Analyze Reviews**.
4. Wait for the scraper to extract available reviews.
5. View the suspicious and genuine review predictions.
6. Examine the analysis summary and pie chart.

## Testing

The project includes separate test scripts for different functionalities.

| Test File | Purpose |
|---|---|
| `test_prediction.py` | Tests review prediction functionality. |
| `test_scraper.py` | Tests product review extraction. |
| `test_scraped_prediction.py` | Tests the integration of scraping and prediction. |
| `test_model_dataset.py` | Tests predictions using dataset records. |

Run individual test scripts using:

```bash
python test_prediction.py
```

```bash
python test_scraper.py
```

```bash
python test_scraped_prediction.py
```

```bash
python test_model_dataset.py
```

## Limitations

- Review extraction depends on the website's HTML structure and accessibility.
- Some platforms dynamically load reviews, which may not be accessible through standard Requests-based scraping.
- Websites may restrict automated requests or display CAPTCHA verification.
- The model is trained using an Amazon review dataset, so performance may vary when analysing reviews from other platforms.
- Missing review metadata, such as verified purchase information, may affect prediction reliability.
- The model's predictions are experimental and should not be treated as definitive proof that a review is fake or genuine.

## Future Scope

- Improve model performance using larger and more diverse datasets.
- Explore advanced NLP and deep learning techniques.
- Improve extraction of dynamically loaded reviews where permitted.
- Add more e-commerce platform support.
- Introduce confidence scores for individual predictions.
- Develop a web-based version of the application.
- Improve review analysis using additional linguistic and behavioural features.

## Disclaimer

This project is developed for academic and educational purposes.

The system identifies potentially suspicious reviews using a trained Machine Learning model. Its predictions are not definitive verification of review authenticity.

**Project:** Fake Review Detection System

**Domain:** Artificial Intelligence and Machine Learning