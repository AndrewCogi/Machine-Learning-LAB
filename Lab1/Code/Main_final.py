# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

# score list (modelName, scalerName, parameters, score)
score_list = []


# =====================================================================
# Function Name : load_and_modification_dataset
# Function Description : Loads and modifies the dataset.
# Input :
# fileName = dataset fileName (including Path, if you need)
# Output : Show Data Exploration(original, modified), return Modified dataset
# =====================================================================

def load_and_modification_dataset(fileName):

    # Load dataset
    df = pd.read_csv(fileName)

    # Data Exploration -----------------------------------

    print('='*27, '<Original Dataset>', '='*27)
    print(df.info(), end='\n\n')
    display(df)

    # Data Preprocessing ---------------------------------

    # Drop ID columns
    df = df.drop(['ID'], axis=1)

    # Clean the dirty data in Bare Nuclei using mean
    df['Bare Muclei'].value_counts()
    df['Bare Muclei'] = df['Bare Muclei'].replace("?", "")
    df['Bare Muclei'] = pd.to_numeric(df['Bare Muclei'])
    df['Bare Muclei'] = df['Bare Muclei'].fillna(df['Bare Muclei'].mean())

    # get correlations of pairs of features in the dataset
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))

    # plot the heatmap
    g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()
    plt.clf()

    # Drop the Mitoses feature because lower correlation with target feature
    df = df.drop(['Mitoses'], axis=1)

    # Modified Dataset
    modified_df = df

    # Plot dataset
    for feature in modified_df.columns.values:
        group_df = modified_df.groupby([feature], dropna=False, as_index=False)
        plt.clf()
        plt.pie(group_df.size()['size'], labels=group_df.size()[
                feature].unique(), autopct="%1.2f%%")
        plt.title("Pie chart ("+feature+")")
        plt.show()
        print(group_df.size(), end="\n\n")

    # Check modified dataset
    print('='*27, '<Modified Dataset>', '='*27)
    print(df.info(), end='\n\n')
    display(df)

    # Return modified dataset
    return modified_df


# =====================================================================
# Function Name : scaling
# Function Description : Scale the input dataset
# Input :
    # dataset = modified_df for scaling
    # method = scaling method (minmax, standard, None)
# Output : plotting(before, after), return scaled_df
# =====================================================================

def scaling(dataset, method='None'):

    # drop label columns
    dataset_X = dataset.drop(['Class'], axis=1)

    # Select Scaling method
    if method == "minmax":
        scaler = preprocessing.MinMaxScaler()
    elif method == "standard":
        scaler = preprocessing.StandardScaler()
    elif method == 'None':
        return dataset

    # Fit & Transform
    scaled_df = scaler.fit_transform(dataset_X)
    scaled_df = pd.DataFrame(scaled_df, columns=dataset_X.columns)

    # Make subplot
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7, 5))

    # Before Scaling plot
    ax1.set_title("Before Scaling")
    ax1.set_xlabel("values")
    for feature in dataset_X.columns.values:
        sns.kdeplot(dataset_X[feature], ax=ax1)

    # After Scaling plot
    ax2.set_title("After Scaling("+method+")")
    ax2.set_xlabel("values")
    for feature in scaled_df.columns.values:
        sns.kdeplot(scaled_df[feature], ax=ax2)
    plt.show()

    # Attach label columns
    scaled_df = pd.concat([scaled_df, dataset['Class']], axis=1)

    # Return scaled_df
    return scaled_df


# =====================================================================
# Function Name : modeling_testing
# Function Description : Get the model and parameters and proceed with modeling, print cross validation score.
# Input :
    # scaled_df = Dataset with scaling completed
    # scalerName = The name of the scaling function
    # modelName = The name of the model you want to use
    # model_params = Parameters values used in each model
    # test_size = Specifying the size of the testset when performing data split (default = 0.3)
# Output : Output of evaluation scores with Cross Validation & result plotting.
# =====================================================================

def modeling_testing(scaled_df, scalerName, modelName, model_params, test_size=0.3):

    # X,y
    X = scaled_df.iloc[:, :-1]
    y = scaled_df.iloc[:, -1]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=42)

    # =====================================================================
    # Modeling (DT_entropy)
    if modelName == 'DT_entropy':

        # max_features = ['auto','sqrt','log2']
        for max_features in list(list(list(model_params.values())[0].values())[1]):

            # DT (entropy)
            model = DecisionTreeClassifier(criterion='entropy')
            model.fit(X_train, y_train)
            y_pred_tr = model.predict(X_test)

            print('='*70)
            print("Predict accuracy (using X_test, y_test)-->%.5f"
                  % accuracy_score(y_test, y_pred_tr))

            # Plotting DT(entropy)
            plt.figure()
            plot_tree(model, filled=True)
            plt.title("Decision Tree (entropy)")
            plt.show()

            # Testing
            print('='*27, '<', 'Cross Validation', '>', '='*27)
            for k in [5, 7, 9]:

                # KFold
                kf = KFold(n_splits=k)
                score = cross_val_score(model, X, y, cv=kf)

                # Show result
                print('Average CV score({}): {}'.format(
                    'model='+modelName +
                    ' , scaler='+scalerName+' , max_features='+max_features +
                    ' , CV='+str(k), score.mean()))
                # Save result
                score_list.append(['model='+modelName+' , scaler='+scalerName +
                                   ' , max_features='+max_features+' , CV='+str(k), score.mean()])
            print()

    # =====================================================================
    # Modeling (DT_gini)
    elif modelName == 'DT_gini':

        # max_features = ['auto','sqrt','log2']
        for max_features in list(list(list(model_params.values())[0].values())[1]):

            # DT (gini)
            model = DecisionTreeClassifier(criterion='gini')
            model.fit(X_train, y_train)
            y_pred_tr = model.predict(X_test)

            print('='*70)
            print("Predict accuracy (using X_test, y_test)-->%.5f"
                  % accuracy_score(y_test, y_pred_tr))

            # Plotting DT(gini)
            plt.figure()
            plot_tree(model, filled=True)
            plt.title("Decision Tree (gini)")
            plt.show()

            # Testing
            print('='*27, '<', 'Cross Validation', '>', '='*27)
            for k in [5, 7, 9]:

                # KFold
                kf = KFold(n_splits=k)
                score = cross_val_score(model, X, y, cv=kf)

                # Show result
                print('Average CV score({}): {}'.format(
                    'model='+modelName +
                    ' , scaler='+scalerName+' , max_features='+max_features +
                    ' , CV='+str(k), score.mean()))

                # Save result
                score_list.append(['model='+modelName+' , scaler='+scalerName +
                                   ' , max_features='+max_features+' , CV='+str(k), score.mean()])
            print()

    # =====================================================================
    # Modeling (SVM)
    elif modelName == 'SVM':

        # kernel = ['linear','poly','rbf','sigmoid']
        for kernel in list(list(list(model_params.values())[2].values())[0]):

            # gamma = ['scale','auto']
            for gamma in list(list(list(model_params.values())[2].values())[1]):

                # C = ['0.01','0.1','1']
                for C in list(list(list(model_params.values())[2].values())[2]):

                    # SVM (SVC)
                    model = SVC(kernel=kernel, gamma=gamma, C=C)
                    model.fit(X_train, y_train)
                    y_pred_tr = model.predict(X_test)

                    print('='*70)
                    print("Predict accuracy (using X_test, y_test)-->%.5f"
                          % accuracy_score(y_test, y_pred_tr))

                    # Testing
                    print('='*27, '<', 'Cross Validation', '>', '='*27)
                    for k in [5, 7, 9]:

                        # KFold
                        kf = KFold(n_splits=k)
                        score = cross_val_score(model, X, y, cv=kf)

                        # Show result
                        print('Average CV score({}): {}'.format(
                            'model='+modelName +
                            ' , scaler='+scalerName+' , kernel='+kernel +
                            ' , gamma='+str(gamma)+' , C='+str(C) +
                            ' , CV='+str(k), score.mean()))

                        # Save result
                        score_list.append(['model='+modelName+' , scaler='+scalerName +
                                           ' , kernel='+kernel+' , gamma='+str(gamma)+' , C='+str(C) +
                                           ' , CV='+str(k), score.mean()])
                    print()

    # =====================================================================
    # Modeling (ligistic regression)
    elif modelName == 'logistic_regression':

        # solver = ['newton-cg','lbfgs','liblinear','sag','saga']
        for solver in list(list(list(model_params.values())[3].values())[0]):

            # LogisticRegression
            model = LogisticRegression(solver=solver)
            model.fit(X_train, y_train)
            y_pred_tr = model.predict(X_test)
            print('='*70)
            print("Predict accuracy (using X_test, y_test)-->%.5f"
                  % accuracy_score(y_test, y_pred_tr))

            # Testing
            print('='*27, '<', 'Cross Validation', '>', '='*27)
            for k in [5, 7, 9]:

                # KFold
                kf = KFold(n_splits=k)
                score = cross_val_score(model, X, y, cv=kf)

                # Show result
                print('Average CV score({}): {}'.format(
                    'model='+modelName +
                    ' , scaler='+scalerName+' , solver='+solver +
                    ' , CV='+str(k), score.mean()))

                # Save result
                score_list.append(['model='+modelName+' , scaler='+scalerName +
                                   ' , solver='+solver+' , CV='+str(k), score.mean()])
            print()


# =====================================================================
# Function Name : do_classification
# Function Description : Proceed whole process
# Input : -None-
# Output : return top_5 classification methods and scores
# =====================================================================

def do_classification():

    # load and modification dataset
    modified_df = load_and_modification_dataset('breast-cancer-wisconsin.csv')

    # Scaling
    for scalerName in ['minmax', 'standard', 'None']:
        print('='*27, '< Scaler name :', scalerName, '>', '='*27, end='\n\n')
        scaled_df = scaling(modified_df, scalerName)

        # Model Parameters
        model_params = {
            'DT_entropy': {
                'criterion': 'entropy', 'max_features': ['auto', 'sqrt', 'log2']
            },
            'DT_gini': {
                'criterion': 'gini', 'max_features': ['auto', 'sqrt', 'log2']
            },
            'SVM': {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': [0.001, 0.01, 0.1, 1, 10],
                'C': [0.01, 0.1, 1]
            },
            'logistic_regression': {
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }
        }

        # Modeling & Testing
        for modelName in ['DT_entropy', 'DT_gini', 'SVM', 'logistic_regression']:
            print('='*27, '< Model name :', modelName, '>', '='*27, end='\n\n')
            modeling_testing(scaled_df, scalerName,
                             modelName, model_params, 0.3)

    # Find top_5 scores (with modelName, params, etc)
    score_list.sort(key=lambda x: x[-1], reverse=True)
    top_5 = score_list[:5]

    # Check whole scores
    # for scores in score_list:
    #     print(scores)
    # print()

    # Return top_5
    return top_5


# =====================================================================
# Main Code
# Describe whole process
# =====================================================================
# import libraries
# data exploration
# data preparation (drop unusable columns)
# data scaling (2 scaling methods, 1 original dataset used -> ['minmax','standard','None'])
    # modeling (4 models and each different parameters)
    # testing (kFold with k=[5,7,9])
    # Save model scores
# Print top 5 model scores
# =====================================================================

# Run the entire process and return top_5
top_5 = do_classification()

# Result top_5 mothod, parameters, scores
print('< Top_5 >')
for rank in top_5:
    print(rank)
