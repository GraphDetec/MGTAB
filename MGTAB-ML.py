from Dataset import MGTAB
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from utils import sample_mask
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='bot', help='detection task of stance or bot')
parser.add_argument('--models_list', type=int, default=[1,2,3,5], nargs='+', help='Selection of classifiers')
parser.add_argument('--random_seed', type=int, default=[0,1,2,3,4], nargs='+', help='Selection of random seeds')
args = parser.parse_args()
print(args)

modelDict = {
    1: "AdaBoost",
    2: "RandomForest",
    3: "DecisionTree",
    4: "XGBoot",
    5: "SVM",
    6: "Lr",
    7: "GB",
    8: "knn"
}


assert set(args.models_list).issubset(modelDict.keys()), 'models should be choose in modelDict'
dataset = MGTAB('./Dataset/MGTAB')
data = dataset[0]


if args.task == 'stance':
    out_dim = 3
    data.y = data.y1
else:
    out_dim = 2
    data.y = data.y2


x = np.array(data.x)
labels = np.array(data.y)
sample_number = len(labels)

for i in args.models_list:
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for j in range(len(args.random_seed)):

        shuffled_idx = shuffle(np.array(range(sample_number)), random_state=args.random_seed[j])
        train_idx = shuffled_idx[:int(0.7 * sample_number)]
        val_idx = shuffled_idx[int(0.7 * sample_number):int(0.9 * sample_number)]
        test_idx = shuffled_idx[int(0.9 * sample_number):]
        data.train_mask = sample_mask(train_idx, sample_number)
        data.val_mask = sample_mask(val_idx, sample_number)
        data.test_mask = sample_mask(test_idx, sample_number)

        x_train = x[data.train_mask]
        y_train = labels[data.train_mask]
        x_test = x[data.test_mask]
        y_test = labels[data.test_mask]

        if i == 1:
            clf = AdaBoostClassifier(
                random_state=args.random_seed[j],
                n_estimators=50,
                learning_rate=1.0,
                algorithm='SAMME.R',
            )
        elif i == 2:
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=args.random_seed[j],
                n_jobs=-1
            )
        elif i == 3:
            clf = DecisionTreeClassifier(
                random_state=args.random_seed[j],
                criterion='gini',
                splitter='best',
                min_samples_split=2,
                min_samples_leaf=1
            )
        elif i == 4:
            clf = XGBClassifier(
                learning_rate=0.1,
                random_state=args.random_seed[j],
                n_estimators=200,
                max_depth=5,
                min_child_weight=1,
                colsample_bytree=0.8,
                objective='binary:logistic'
            )
        elif i == 5:
            clf = SVC(
                kernel='rbf',
                C=10,
                random_state=args.random_seed[j],
                probability=True
            )
        elif i == 6:
            clf = LogisticRegression(
                C=0.1,
                random_state=args.random_seed[j],
                max_iter=500
            )
        elif i == 7:
            clf = GaussianNB()
        elif i == 8:
            clf = KNeighborsClassifier(n_neighbors=7)

        clf.fit(X=x_train, y=y_train)
        y_pred = clf.predict(x_test)
        acc_list.append(accuracy_score(y_true=y_test, y_pred=y_pred)*100)
        precision_list.append(precision_score(y_true=y_test, y_pred=y_pred, average='macro')*100)
        recall_list.append(recall_score(y_true=y_test, y_pred=y_pred, average='macro')*100)
        f1_list.append(f1_score(y_true=y_test, y_pred=y_pred,  average='macro')*100)

    print('\n'+'*'*30)
    print('model:     {}'.format(modelDict[i]))
    print('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    print('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    print('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    print('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list)))
