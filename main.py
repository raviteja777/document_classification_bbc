from data_process import DataProcessor
from model_config import ModelGenerator
from sklearn.metrics import classification_report

import pickle as pkl
import os
from datetime import datetime


TRAIN_DATA_DIR = "data"
SAVE_MODEL_PATH = "save_model"


def save_model(name, model_files):
    ts = datetime.now().strftime('_%y%m%d_%H%M%S')
    filename = os.path.join(SAVE_MODEL_PATH, name + ts + '.pkl')
    with open(filename, 'wb') as fh:
        pkl.dump(model_files, fh)


def evaluate_model(cl_model, test_data):
    print(cl_model.best_estimator_)
    cols = test_data.columns.difference(['sentences', 'labels'])
    print(test_data.head())
    predictions = list(map(int, cl_model.predict(test_data[cols])))
    print(classification_report(test_data['labels'], predictions))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # process data
    proc = DataProcessor(data_dir=TRAIN_DATA_DIR)
    encoder = proc.label_encoder
    vectorizer = proc.tfidf_vectorizer
    data_train, data_test = proc.process_data()

    # generate and train models
    model_gen = ModelGenerator(data_train)
    nb = model_gen.naive_bayes()
    knn = model_gen.knn()
    svm = model_gen.svm()
    rf = model_gen.random_forest()
    models = {'nb': nb, 'knn': knn, 'svm': svm, 'rf': rf}

    # evaluate on validation and test set
    [evaluate_model(m, data_test) for m in models.values()]

    # save models - save to pickle file : model object & processor object
    [save_model(name, {'model': model, 'processor': proc}) for name, model in models.items()]
