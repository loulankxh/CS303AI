import argparse
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training_set', type=str, default='train.json')
    parser.add_argument('-i', '--testing_set', type=str, default='testdataexample.txt')
    args = parser.parse_args()
    train_path = args.training_set
    test_path = args.testing_set

    train_data = []
    train_label = []
    f = open(train_path, "r")
    info = json.load(f)     # list[dictionary]
    for line in info:
        train_data.append(line['data'])
        train_label.append(line['label'])
    f.close()

    # X_train_data = CountVectorizer().fit_transform(train_data)     # to feature vectors
    # X_train_data_tfidf = TfidfTransformer().fit_transform(X_train_data)    # tfidf
    #
    # # linearSVC
    # svc = SVC(kernel='rbf', class_weight='balanced', )
    # c_range = np.logspace(-5, 15, 11, base=2)
    # gamma_range = np.logspace(-9, 3, 13, base=2)
    # param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    # grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # clf = grid.fit(X_train_data_tfidf, train_label)

    # SVM GridSearch
    text_clf_svm = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                       alpha=1e-3,
                                                       random_state=42))])
    text_clf_svm = text_clf_svm.fit(train_data, train_label)
    parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
                      'tfidf__use_idf': (True, False),
                      'clf-svm__alpha': (1e-2, 1e-3, 1e-4, 1e-5)}
    gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
    gs_clf_svm = gs_clf_svm.fit(train_data, train_label)

    ft = open(test_path, "r")
    info = json.load(ft)
    test_data = []
    for line in info:
        test_data.append(line)
    ft.close()

    # predict
    # X_test_data = CountVectorizer().fit_transform(train_data)  # to feature vectors
    # X_test_data_tfidf = TfidfTransformer().fit_transform(X_train_data)  # tfidf
    # predicted = clf.best_estimator_.predict(X_test_data_tfidf)

    predicted = gs_clf_svm.best_estimator_.predict(test_data)

    out = open("output.txt", "w")
    for num in predicted:
        out.write(str(num) + '\n')
    out.close()


