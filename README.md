# Spam Detection with K-Nearest Neighbors (KNN)

## Student Information
* Sandikha Rahardi
* РИМ-130908

## Project Overview
This project is a lab assignment focused on classifying email messages as either "spam" or "not-spam" using the K-Nearest Neighbors (KNN) algorithm. The notebook demonstrates the process of preprocessing the data, building a custom KNN classifier, and evaluating the model's performance.

## Dataset
The dataset used in this project is sourced from Kaggle:
- **[Email Classification: Ham or Spam](https://www.kaggle.com/datasets/prishasawhney/email-classification-ham-spam)**

It contains two columns:
- **Email**: The text content of the email/message.
- **Label**: The classification of the message, either "Ham" (non-spam) or "Spam".

### Dataset Preprocessing
To prepare the data for modeling, the following steps are applied:
1. **Text cleaning**: Removing unnecessary characters, symbols, and converting text to lowercase.
2. **Tokenization**: Breaking the email text into individual words (tokens).
3. **Vectorization**: Converting the tokenized text into a numeric format.
4. **Label Encoding**: Mapping the "Ham" and "Spam" labels to numeric values (Ham = 0, Spam = 1).

## Implementation
The core algorithm used in this project is **K-Nearest Neighbors (KNN)**. In this notebook, we build the KNN classifier from scratch without relying on external libraries for the algorithm itself.

### Steps Involved:
1. **Data Exploration**: Visualizing and understanding the structure of the dataset.
2. **Text Preprocessing**: Cleaning and transforming the email content into a suitable format for model training.
3. **KNN Classifier Implementation**: Manually implementing the KNN algorithm and applying it to the dataset.
4. **Model Evaluation**: Testing the model and measuring its performance using common metrics like accuracy.

## How to Run the Project

### Prerequisites
Make sure you have the following Python libraries installed:
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- NLTK

```bash
    pip install -r requirements.txt
```

### Instructions
1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    ```

2. **Download the dataset** from Kaggle and place it in the project directory.
   
3. **Run the Jupyter notebook**:
    - Open the notebook `01_Spam_Detection_Sandikha_Rahardi_(РИМ_130908).ipynb`.
    - Execute the cells in order to preprocess the data, build the KNN model, and evaluate its performance.

4. **Modify the code**:
    - You can experiment with different preprocessing techniques, adjust hyperparameters, or try different classification models to improve the results.

## Project Structure
```bash
├── Spam_Detection.ipynb    # Main Jupyter notebook
├── README.md               # Project documentation
└── data                    # Directory for dataset
