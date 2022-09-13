from typing import List


def get_confusion_matrix(
    actual: List[int], predicted: List[int]
) -> List[List[int]]:
    """Computes confusion matrix from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        List of two lists of length 2 each, representing the confusion matrix.
    """
    matrix = [[0, 0], [0, 0]]
    for i in range(len(actual)):
        if actual[i] == 0 and predicted[i] == 0:
            matrix[0][0] += 1
        elif actual[i] == 0 and predicted[i] == 1:
            matrix[0][1] += 1
        elif actual[i] == 1 and predicted[i] == 0:
            matrix[1][0] += 1
        elif actual[i] == 1 and predicted[i] == 1:
            matrix[1][1] += 1
    return matrix

def accuracy(actual: List[int], predicted: List[int]) -> float:
    """Computes the accuracy from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Accuracy as a float.
    """
    acc = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            acc += 1
    return acc / len(actual) 


def precision(actual: List[int], predicted: List[int]) -> float:
    """Computes the precision from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Precision as a float.
    """
    conf_matrix = get_confusion_matrix(actual, predicted)
    TruePositives, FalsePositives = conf_matrix[1][1], conf_matrix[0][1]
    return TruePositives / (TruePositives + FalsePositives)



def recall(actual: List[int], predicted: List[int]) -> float:
    """Computes the recall from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Recall as a float.
    """
    conf_matrix = get_confusion_matrix(actual, predicted)
    TruePositives, FalseNegatives = conf_matrix[1][1], conf_matrix[1][0]
    return TruePositives / (TruePositives + FalseNegatives)

def f1(actual: List[int], predicted: List[int]) -> float:
    """Computes the F1-score from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of harmonic mean of precision and recall.
    """
    Precision, Recall = precision(actual, predicted), recall(actual, predicted)
    return 2 * (Precision * Recall) / (Precision + Recall)


def false_positive_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false positive rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as positive divided
            by number of actually negative instances.
    """
    conf_matrix = get_confusion_matrix(actual, predicted)
    FalsePositives, TrueNegatives = conf_matrix[0][1], conf_matrix[0][0]
    return FalsePositives / (FalsePositives + TrueNegatives)


def false_negative_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false negative rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as negative divided
            by number of actually positive instances.
    """
    conf_matrix = get_confusion_matrix(actual, predicted)
    FalseNegatives, TruePositives = conf_matrix[1][0], conf_matrix[1][1]
    return FalseNegatives / (FalseNegatives + TruePositives)
