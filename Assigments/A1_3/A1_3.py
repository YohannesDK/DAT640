from typing import List, Tuple, Union
import string

from numpy import ndarray

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_data(path: str) -> Tuple[List[str], List[int]]:
    """Loads data from file. Each except first (header) is a datapoint
    containing ID, Label, Email (content) separated by "\t". Lables should be
    changed into integers with 1 for "spam" and 0 for "ham".

    Args:
        path: Path to file from which to load data

    Returns:
        List of email contents and a list of lobels coresponding to each email.
    """
    email_contents = []
    email_labels = []
    with open(path, "r", encoding="utf8") as f:
        data = f.readlines()
        
        if len(data) < 1:
            return

        for email in data[1:]:
            email_data = email.split("\t")
            email_contents.append(email_data[2].replace('"', "").replace("\n", ""))
            email_labels.append(0 if email_data[1] == "ham" else 1)
    return email_contents, email_labels


def preprocess(doc: str) -> str:
    """Preprocesses text to prepare it for feature extraction.

    Args:
        doc: String comprising the unprocessed contents of some email file.

    Returns:
        String comprising the corresponding preprocessed text.
    """

    doc = doc.lower()
    # punctuation removal 
    for c in string.punctuation:
        doc = doc.replace(c, "")

    # #stopword removal - assumes that punkt and stopwords are downloaded
    stop_words = stopwords.words("english")
    words = doc.split()
    doc_stopworded = [word for word in words if word not in stop_words]
    doc = " ".join(doc_stopworded)
    return doc

    


def preprocess_multiple(docs: List[str]) -> List[str]:
    """Preprocesses multiple texts to prepare them for feature extraction.

    Args:
        docs: List of strings, each consisting of the unprocessed contents
            of some email file.

    Returns:
        List of strings, each comprising the corresponding preprocessed
            text.
    """
    return [preprocess(doc) for doc in docs]


def extract_features(
    train_dataset: List[str], test_dataset: List[str]
) -> Union[Tuple[ndarray, ndarray], Tuple[List[float], List[float]]]:
    """Extracts feature vectors from a preprocessed train and test datasets.

    Args:
        train_dataset: List of strings, each consisting of the preprocessed
            email content.
        test_dataset: List of strings, each consisting of the preprocessed
            email content.

    Returns:
        A tuple of of two lists. The lists contain extracted features for 
          training and testing dataset respectively.
    """

    # create the transform
    vectorizor = CountVectorizer()

    # build vocabulary
    train_features = vectorizor.fit_transform(train_dataset)

    # encode document
    test_features = vectorizor.transform(test_dataset)

    # return features
    return train_features, test_features



def train(X: ndarray, y: List[int]) -> object:
    """Trains a classifier on extracted feature vectors.

    Args:
        X: Numerical array-like object (2D) representing the instances.
        y: Numerical array-like object (1D) representing the labels.

    Returns:
        A trained model object capable of predicting over unseen sets of
            instances.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def evaluate(y: List[int], y_pred: List[int]) -> Tuple[float, float, float, float]:
    """Evaluates a model's predictive performance with respect to a labeled
    dataset.

    Args:
        y: Numerical array-like object (1D) representing the true labels.
        y_pred: Numerical array-like object (1D) representing the predicted
            labels.

    Returns:
        A tuple of four values: recall, precision, F_1, and accuracy.
    """
    recall, precision, f1, accuracy  = recall_score(y, y_pred), precision_score(y, y_pred), \
        f1_score(y, y_pred), accuracy_score(y, y_pred)
    return recall, precision, f1, accuracy


if __name__ == "__main__":
    print("Loading data...")
    train_data_raw, train_labels = load_data("data/train.tsv")
    test_data_raw, test_labels = load_data("data/test.tsv")

    print("Processing data...")
    train_data = preprocess_multiple(train_data_raw)
    test_data = preprocess_multiple(test_data_raw)

    print("Extracting features...")
    train_feature_vectors, test_feature_vectors = extract_features(
        train_data, test_data
    )

    print("Training...")
    classifier = train(train_feature_vectors, train_labels)

    print("Applying model on test data...")
    predicted_labels = classifier.predict(test_feature_vectors)

    print("Evaluating")
    recall, precision, f1, accuracy = evaluate(test_labels, predicted_labels)

    print(f"Recall:\t{recall}")
    print(f"Precision:\t{precision}")
    print(f"F1:\t{f1}")
    print(f"Accuracy:\t{accuracy}")
