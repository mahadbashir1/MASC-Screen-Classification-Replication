import warnings

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix, \
    roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')


# Function to load data
def load_data():
    """
    Load the dataset and labels from CSV files.

    Returns:
        data (DataFrame): The dataset containing features.
        y (Series): The target labels.
    """
    data = pd.read_csv('dataset.csv')
    labels = pd.read_csv('labels.csv')
    y = labels['class']
    data['keywords'].fillna('', inplace=True)
    return data, y


# Function to preprocess text
def preprocess_text(text_data):
    """
    Preprocess the text data by removing punctuation, special characters, stop words, and stemming.

    Args:
        text_data (list): List of text strings to preprocess.

    Returns:
        preprocessed_text (list): List of preprocessed text strings.
    """
    stemmer = PorterStemmer()
    preprocessed_text = []
    stop_words = set(stopwords.words('english'))
    for text in text_data:
        if not isinstance(text, str):
            text = str(text)
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'url', text)
        text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_.]))@([A-Za-z]+[A-Za-z0-9]+)', '', text)
        text = re.sub(r'[\s]+', ' ', text)
        text = re.sub(r'#([^\s]+)', r'\1', text)
        text = re.sub(r'-', ' ', text)
        text = re.sub(r'/', ' ', text)
        text = re.sub(r',', ' ', text)
        text = re.sub(r'\.', ' ', text)
        # Convert text to lowercase
        text = text.lower()
        # Remove stop words
        text = ' '.join([word for word in text.split() if word not in stop_words])
        # Stem words using NLTK's Porter Stemmer
        stemmed_text = ' '.join([stemmer.stem(word) for word in text.split()])
        preprocessed_text.append(stemmed_text)
    return preprocessed_text


# Function to write array to file
def write_array_to_file(text_array, filename, encoding="utf-8"):
    """
    Write an array of text strings to a file.

    Args:
        text_array (list): List of text strings to write.
        filename (str): The name of the file to write to.
        encoding (str): The encoding to use when writing the file.
    """
    with open(filename, "w", encoding=encoding) as f:
        for text in text_array:
            f.write(text + "\n")


# Function to preprocess data
def preprocess_data(data):
    """
    Preprocess the data by cleaning text and combining numeric features with TF-IDF transformed text features.

    Args:
        data (DataFrame): The dataset containing features.

    Returns:
        X (DataFrame): The preprocessed feature data.
    """
    X = data['keywords']
    cleaned_text = preprocess_text(X)
    numeric_cols = [col for col in data if col != 'keywords']
    X_numeric = data[numeric_cols]
    vect = TfidfVectorizer()
    X_text = vect.fit_transform(cleaned_text)
    joblib.dump(vect, 'tfidf_vectorizer.joblib')
    X = pd.concat([X_numeric, pd.DataFrame(X_text.toarray())], axis=1)
    return X


# Function to train and evaluate models
def train_and_evaluate_models(features, y_train):
    """
    Train and evaluate multiple machine learning models on the given dataset.

    Args:
        features (DataFrame): The feature data.
        y_train (Series): The target labels.

    Returns:
        accuracies (dict): A dictionary containing model names as keys and their respective
                         accuracy and classification report as values.
    """
    class_names = ["Chat", "Home", "List", "Login", "Map", "Menu", "Profile", "Search", "Setting", "Welcome"]
    # Split the data into train, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(features, y_train, test_size=0.25, random_state=41,
                                                        stratify=y_train)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.6, random_state=41, stratify=y_test)

    # Define a list of machine learning models
    models = [
        ('XGBoost', XGBClassifier()),
        ('Gradient Boosting', GradientBoostingClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('Multi-Layer Perceptron', MLPClassifier(max_iter=1000)),
        ('Adaboost', AdaBoostClassifier()),
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('Support Vector Machine - rbf', SVC(kernel='rbf', C=1, random_state=0, probability=True)),
        ('Support Vector Machine - linear', SVC(kernel='linear', C=1, random_state=0, probability=True)),
    ]

    accuracies = {}
    best_classifier = None
    best_model = None
    best_accuracy = 0.0

    print("train_accuracy", "test_accuracy", "val_accuracy", "precision", "roc_auc", "recall", "f1")

    for name, model in models:
        # Train the model
        model.fit(X_train, y_train)
        # Test the model on the test set
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        accuracy = np.mean(score)
        # Validate the model on the validation set
        val_accuracy = model.score(X_val, y_val)
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        # Calculate ROC-AUC
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, average='macro', multi_class='ovr')
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        accuracies[name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ROC_AUC': roc_auc
        }

        formatted_train_accuracy = "{:.2f}".format(score * 100)
        formatted_test_accuracy = "{:.2f}".format(accuracy * 100)
        formatted_val_accuracy = "{:.2f}".format(val_accuracy * 100)
        formatted_roc_auc = "{:.2f}".format(roc_auc * 100)
        formatted_f1 = "{:.2f}".format(f1 * 100)
        formatted_precision = "{:.2f}".format(precision * 100)
        formatted_recall = "{:.2f}".format(recall * 100)

        print(formatted_train_accuracy, formatted_test_accuracy, formatted_val_accuracy, formatted_precision,
              formatted_roc_auc, formatted_recall, formatted_f1)

        plot_confusion_matrix(y_test, y_pred, class_names, name)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = name
            best_model = model

    print("Best Classifier:", best_classifier)
    print("Best Accuracy:", best_accuracy)
    joblib.dump(best_model, 'best_classifier_model.joblib')
    return accuracies


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, algorithm_name):
    """
    Plot the confusion matrix for a given classifier.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list): List of class names.
        algorithm_name (str): Name of the algorithm.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {algorithm_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{algorithm_name}.png')
    plt.close()


# Function to plot classifier accuracies
def plot_classifier_accuracies(accuracies):
    """
    Plot the accuracies of different classifiers.

    Args:
        accuracies (dict): Dictionary containing model names and their accuracies.
    """
    classifiers = list(accuracies.keys())
    accuracies_values = [entry['accuracy'] for entry in accuracies.values()]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classifiers, accuracies_values, color='g', label="Comparative analysis Classifiers")

    # Add labels and a title
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy Comparison')
    plt.xticks(rotation=45)

    # Add accuracy values on top of the bars
    for bar, acc in zip(bars, accuracies_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, '{:.2%}'.format(acc), va='bottom',
                 ha='center')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data, y = load_data()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X = preprocess_data(data)
    accuracies = train_and_evaluate_models(X, y_encoded)
    plot_classifier_accuracies(accuracies)