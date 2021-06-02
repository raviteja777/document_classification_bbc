import pickle as pkl
import pandas as pd

TEST_DATA_DIR = "test_data"


# load a save model and predict from given test files
def run_save_model(model_file_path):
    model_files = pkl.load(open(model_file_path, 'rb'))
    processor = model_files['processor']
    model = model_files['model']
    test_data = processor.process_test_data(TEST_DATA_DIR)
    cols = test_data.columns.difference(['sentences', 'file_names'])
    predict = list(map(int, model.predict(test_data[cols])))
    print(pd.DataFrame({'file_name': test_data['file_names'],
                        'classify': processor.label_encoder.inverse_transform(predict)}))


saved_model_path = 'save_model/nb_210602_122946.pkl'
run_save_model(saved_model_path)
