import numpy as np
from scipy.stats import randint
import pandas as pd
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from pandas import set_option
plt.style.use('ggplot') # nice plots
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold # for cross validation
from sklearn.model_selection import GridSearchCV # for tuning parameter
from sklearn.model_selection import RandomizedSearchCV  # Randomized search on hyper parameters.
from sklearn.preprocessing import StandardScaler, MinMaxScaler # for normalization
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os


def apply_resampling(df, user_input):
    if user_input == 'No resampling' or user_input is None:
        return df
    else:
        count_class_0, count_class_1 = df.Default.value_counts()
        data_default = df[df['Default'] == 1]
        data_non_default = df[df['Default'] == 0]
        if user_input == 'Undersampling':
            resampled_data = data_non_default.sample(count_class_1)
            return pd.concat([resampled_data, data_default], axis=0)
        else:
            resampled_data = data_default.sample(count_class_0, replace = True)
            return pd.concat([resampled_data, data_non_default], axis=0)

def split_df(df, target, selected_columns ):
    y = df[target]
    X = df[selected_columns]
    return train_test_split(X,y,test_size=0.2,random_state=27)



def apply_scaling(X_train, X_test, type):
    if type == 'MinMax':
        norm = MinMaxScaler().fit(X_train)
        X_train_norm = norm.transform(X_train)
        X_test_norm = norm.transform(X_test)
        return X_train_norm, X_test_norm
    elif type == 'Standard':
        sc = StandardScaler().fit(X_train)
        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)
        return X_train_sc, X_test_sc
    else:
        return X_train, X_test

def create_model(X_train, X_test, y_train, y_test, model_type):
    model_dictionary = {'logistic regression': LogisticRegression(), 'decision tree': DecisionTreeClassifier(criterion= 'gini', max_depth= 5, 
                                            min_samples_leaf= 2, 
                                            random_state=0), 'knn':  KNeighborsClassifier(), 'random forest': RandomForestClassifier()}
    try:
        model = model_dictionary[model_type]
    except:
        model = model_dictionary['random forest']
    model.fit(X_train, y_train)
    return model

def predict_with_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics_dictionary = {'accuracy':metrics.accuracy_score(y_pred,y_test), 'precision': metrics.precision_score(y_pred,y_test), 'recall': metrics.recall_score(y_pred,y_test) ,
                         'f1': metrics.f1_score(y_pred,y_test), 'roc_auc': metrics.roc_auc_score(y_pred,y_test)}
    return y_pred, metrics_dictionary


    
def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(10,10))
    ConfMatrix = confusion_matrix(y_test,y_pred)
    sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 
                xticklabels = ['Non-default', 'Default'], 
                yticklabels = ['Non-default', 'Default'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Confusion Matrix")
    plt.savefig('/Users/louis.reinaldo/Documents/side_projects/flask/static/plot.png', format="png")
    print('plot_saved')
    #plt.show()
    

def model_pipeline(data, target, selected_columns, sampling_type, scaling_type, model_type):
    data = apply_resampling(data, sampling_type)
    X_train, X_test, y_train, y_test = split_df(data, target , selected_columns)
    X_train, X_test = apply_scaling(X_train, X_test, scaling_type)
    model = create_model(X_train, X_test, y_train, y_test, model_type)
    y_pred, metrics_dictionary = predict_with_model(model, X_test, y_test)
    model_details = {'target': target, 'features': selected_columns, 'resampling': sampling_type, 'scaling': scaling_type, 'model': model_type, 'metrics': metrics_dictionary}
    return model_details
    print('Model Details')
    print(model_details)
    plot_confusion_matrix(y_test, y_pred)



