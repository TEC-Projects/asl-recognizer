# Model validation module. This module generates insights from the model performance results.

import pickle
import matplotlib.pyplot as plt
from os.path import join

import numpy as np
import seaborn as sn

from classes.StatisticalClassifier import StatisticalClassifier


def load_object(load_path):
    """
    Function that load the list of features from disk to python
    :param load_path: path in which the features are saved
    :return: list of tuples containing observations and labels
    """
    with open(join(load_path, "features_dump.pkl"), "rb") as reader:
        return pickle.load(reader)


def load_model(load_path):
    """
    Function that load a pretrained model from disk
    :param load_path: path in which the model was saved
    :return: Classification model
    """
    with open(load_path, "rb") as reader:
        return pickle.load(reader)


def save_model(features, save_path):
    """
    Function that saves an array of features and labels to a pickle file
    :param save_path: path in which the features will be saved
    :param features: array of features to save
    """
    with open(join(save_path, "trained_model.pkl"), "wb") as writer:
        writer.write(pickle.dumps(features))


def show_confusion_matrix(confusion_matrix):
    sn.heatmap(confusion_matrix, annot=True)
    plt.title("Matriz de confusión para la clasificación del abecedario ASL")
    plt.xlabel("Predicciones")
    plt.ylabel("Objetivo")
    plt.show()


def get_classifier(trained_model_path=""):
    """
    Function that instantiates the classifier, trains it and returns it
    :param trained_model_path: if not empty, it will load a trained model
    :return: classifier and test dataset
    """
    if trained_model_path == "":
        statisticalClassifier = StatisticalClassifier()
        # Load default features
        vectors = load_object("./features/")
        # Split dataset into training and testing
        train_dataset, test_dataset = statisticalClassifier.split(vectors)
        # Train the classifier
        statisticalClassifier.fit(train_dataset)
        # return trained classifier and test dataset
        return statisticalClassifier, test_dataset
    else:
        return load_model(trained_model_path)


def main():
    """
    Procedure that retrieves features, trains the classifier and tests it
    """
    statisticalClassifier, test_dataset = get_classifier()
    score, confusion_matrix = statisticalClassifier.score(test_dataset)

    show_confusion_matrix(confusion_matrix)

    for i in range(confusion_matrix.shape[0]):
        print(f"Class {i}: accuracy: {confusion_matrix[i, i] / np.sum(confusion_matrix[i])}")

    print("Overall score: ", score)


# Example validation run: python ./validator.py
if __name__ == '__main__':
    # This function will only be run if the current file is run directly
    main()
