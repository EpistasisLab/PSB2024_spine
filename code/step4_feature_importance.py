import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# read the data

# medicare tmc
data = pd.read_csv("/norm_datasets.csv")


print("Shape of the data: ", data.shape)
print("Columns in the data: ", data.columns)

# change the cluster_label (target) to 0 and 1
data['cluster_label'] = data['cluster_label'].replace({1: 0, 2: 1})

# Define the list of feature categories - commercial and medicare
feature_categories = []


tpot_config = {
    # Classifiers
    'sklearn.naive_bayes.GaussianNB': {
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },

    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },

    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1],
        'verbosity': [0]
    },

    'sklearn.linear_model.SGDClassifier': {
        'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001],
        'learning_rate': ['invscaling', 'constant'],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    'sklearn.neural_network.MLPClassifier': {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
    }
}


# create empty list to store the results
results = []

# Iterate over the feature categories
for feature_label in feature_categories:
    # If the feature category is a list, select all features in the list
    if isinstance(feature_label, list):
        features = data[feature_label]
    else:
        features = data[[feature_label]]
    
    # Select the target column
    target = data['cluster_label']

    features_train, features_test, target_train, target_test = train_test_split(features, target, train_size=0.8, test_size=0.2, random_state=42)
    
    # Instantiate TPOTClassifier
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, config_dict= tpot_config, random_state=42)
    
    # Fit TPOTClassifier
    tpot.fit(features_train, target_train)
    target_predicted = tpot.predict(features_test)
    
    # Save the best pipeline
    optimized_pipeline_str = tpot.clean_pipeline_string(
                    tpot._optimized_pipeline
                )
    # Get the name of the final pipeline
    final_pipeline_name = optimized_pipeline_str

    accuracy = accuracy_score(target_test, target_predicted)
    balanced_accuracy = balanced_accuracy_score(target_test, target_predicted)
    f1_score_value = f1_score(target_test, target_predicted)

    
    # Store the results as a dictionary
    result = {
        'feature_label': feature_label,
        'accuracy': accuracy,
        'balanced accuracy': balanced_accuracy,
        'f1_score': f1_score_value,
        'model_name': final_pipeline_name
    }
    
    # Append the result to the list
    results.append(result)
    

# run entire model
features = data.iloc[:, 0:-1]
target = data.iloc[:,-1]

features_train, features_test, target_train, target_test = train_test_split(features, target, train_size=0.8, test_size=0.2, random_state=42)
    
# Instantiate TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, config_dict= tpot_config, random_state=42)
    
# Fit TPOTClassifier
tpot.fit(features_train, target_train)
target_predicted = tpot.predict(features_test)
    
# Save the best pipeline
optimized_pipeline_str = tpot.clean_pipeline_string(
                    tpot._optimized_pipeline
                )
# Get the name of the final pipeline
final_pipeline_name = optimized_pipeline_str
accuracy = accuracy_score(target_test, target_predicted)
balanced_accuracy = balanced_accuracy_score(target_test, target_predicted)
f1_score_value = f1_score(target_test, target_predicted)

# 
# Store the results as a dictionary
result = {
        'feature_label': 'all',
        'accuracy': accuracy,
        'balanced accuracy': balanced_accuracy,
        'f1_score': f1_score_value,
        'model_name': final_pipeline_name
    }
    
# Append the result to the list
results.append(result)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('out.csv', index=False)