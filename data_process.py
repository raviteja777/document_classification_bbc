import pandas as pd
import os
import spacy
import codecs

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# process the input text files
# convert text to tf idf vectors
# split into train and test sets
class DataProcessor:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_encoder = self.set_labels()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, decode_error='replace', min_df=2)

    def set_labels(self):
        categories = [f.lower() for f in os.listdir(self.data_dir)
                      if os.path.isdir(os.path.join(self.data_dir, f))]
        le = LabelEncoder()
        le.fit(categories)
        return le

    def process_data(self):
        print("===== Processing data =====")
        dataset = pd.DataFrame()
        categories = self.label_encoder.classes_
        for cat in categories:
            nlp = spacy.load('en_core_web_sm')
            counter = 1
            for f in os.listdir(os.path.join(self.data_dir, cat)):
                try:
                    if counter % 10 == 0:
                        print("======Processes {0} files in category {1}".format(counter, cat))
                    f_read = open(os.path.join(self.data_dir, cat, f), 'r')  # .encode(encoding='utf-8',errors='ignore')
                    doc = " ".join([line.strip().lower() for line in f_read])
                    doc_vec = " ".join([word.lemma_ for word in nlp(doc)
                                        if not (word.is_space or word.is_stop or word.is_punct or ('\n' in word.text))])
                    # print(doc_vec)
                    label = pd.Series(self.label_encoder.transform([cat]), dtype='Int64')
                    # print(doc_vec)
                    if len(doc_vec.strip()) > 0:
                        dataset = dataset.append({'sentences': doc_vec, 'labels': label[0]}, ignore_index=True)
                    counter += 1
                except ValueError as e:
                    print(f)
                    print(e)

        vectors = pd.DataFrame(self.tfidf_vectorizer.fit_transform(dataset['sentences']).toarray())
        dataset = dataset.join(vectors)
        data_train, data_test = train_test_split(dataset, test_size=0.3, random_state=1234)
        print(data_train.head())
        print(data_test.head())
        return data_train, data_test

    # process test phrases on new data
    def process_test_data(self, test_data):
        test_files = []
        if os.path.isdir(test_data):
            test_files += [os.path.join(test_data, f) for f in os.listdir(test_data)]
        else:
            test_files.append(test_data)
        test_dataset = pd.DataFrame()
        nlp = spacy.load('en_core_web_sm')
        for f in test_files:
            doc = " ".join([line.lower() for line in open(f, 'r')])
            doc_vec = " ".join([word.lemma_ for word in nlp(doc)
                                if not (word.is_space or word.is_stop or word.is_punct or ('\n' in word.text))])
            test_dataset = test_dataset.append({'sentences': doc_vec, 'file_names': f}, ignore_index=True)
        return test_dataset.join(pd.DataFrame(self.tfidf_vectorizer.transform(test_dataset['sentences']).toarray()))
