# Main execution module.
from feature_extractor import video_feature_extraction
from validator import get_classifier

if __name__ == '__main__':
    print("ASL recognizer.")
    classifier = get_classifier("./model/trained_model.pkl")
    video_feature_extraction(classifier)
