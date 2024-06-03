# Feature extraction module.
from os import listdir, curdir
from os.path import isfile, isdir, join
import cv2
import mediapipe as mp
import pickle
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def save_features(features, save_path):
    """
    Function that saves an array of features and labels to a pickle file
    :param save_path: path in which the features will be saved
    :param features: array of features to save
    """
    with open(join(save_path, "features_dump.pkl"), "wb") as writer:
        writer.write(pickle.dumps(features))


def draw_landmarks(image, landmarks, save_path, image_name):
    """
    Procedure that draws landmarks on the given image if they were found and saves it
    :param image:
    :param landmarks:
    :param save_path:
    :param image_name:
    :return:
    """
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(save_path + image_name + ".png", image)


def extract_features(dataset_path, save_path):
    features_vectors = {}

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
    ) as hands:
        dirs = [d for d in listdir(dataset_path) if isdir(join(dataset_path, d))]
        for directory in dirs:
            full_dir = join(dataset_path, directory)
            file_names = [f for f in listdir(full_dir) if isfile(join(full_dir, f))]

            count = 0
            features_vectors[directory] = []

            for file_name in file_names:
                if count > 10:
                    break
                image = cv2.flip(cv2.imread(join(full_dir, file_name)), 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                vector = []
                if results.multi_hand_landmarks:
                    for landmark in results.multi_hand_landmarks[0].landmark:
                        vector += [landmark.x, landmark.y, landmark.z]
                    features_vectors[directory].append(vector)
                    count += 1

        save_features(features_vectors, save_path)


if __name__ == "__main__":
    # This function will only be run if the current file is run directly
    if len(sys.argv) < 3:
        raise Exception("Missing machine code argument.")
    extract_features(sys.argv[1], sys.argv[2])
