from dataset import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils import *
from sklearn.svm import SVC

if __name__ == '__main__':
    Train_path = 'D:\\project\\MRN\\dataset\\BasicDataset_Training_MRN.csv'  # PUT YOUR FILE PATH
    Test_path = 'D:\\project\\MRN\\dataset\\BasicDataset_Test_MRN.csv'

    train_data, train_label = load_data(Train_path)
    test_data, test_label = load_data(Test_path)

    # Choose k of K-Dold Validation Procedure
    splits = 2

    # Select ML Classifiers (here just one simple example)
    # LR: model has 1 Hyper-Parameter, which is the
    # regularisaton coefficient C (ref. : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
    # Its value should be tuned according to a K-Fold Validation Procedure.
    names = ["Random Forest"]
    raw_classifiers = [RandomForestClassifier()]

    # Choose Hyper-Parameters Candidate Values
    # LR: in this example a grid search is performed over a set of 3 candidate
    # values for hyper-parameter C
    parameters_grid = [
        {'n_estimators': [50, 75, 100, 125, 150, 175, 200, 225, 250], 'max_depth': [None, 5, 10, 15, 20, 25, 30, 35]}]

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    dataset = (train_data - mean) / std
    dataset.dropna(inplace=True)  # check data consistency

    print(mean)
    print(std)
    # Print Info
    print(20 * '*')
    print('% of Class 1 in Training : ', len(dataset[dataset == 1]) / len(dataset),
          '; % of Class 0 in Training: ', len(dataset[dataset == 0]) / len(dataset))
    print('Shape Training Set :', dataset.shape, ',', dataset.shape)
    print(20 * '*')

    # Select Best Hyperparameter Value for This Fold
    # In this example, only LR classifier is selected.
    hp_values = hyperparameter_tuning(dataset, train_label, names,
                                      raw_classifiers, parameters_grid, n_splits_in=splits)

    print(hp_values.head())
    print(hp_values)

    reg_coeff = hp_values.loc["Random Forest", "BestHP_Values"][0]
    print(reg_coeff)

    classifiers = [RandomForestClassifier(max_depth=hp_values.loc["Random Forest", "BestHP_Values"][0],
                                          n_estimators=hp_values.loc["Random Forest", "BestHP_Values"][
                                              1])]  # Tuned configuration of the classifier

    ## NORMALISE DATA (Always using mean and sts computed from training Data)
    mean = test_data.mean(axis=0)
    std = test_data.std(axis=0)
    dataset_test = (test_data - mean) / std
    dataset_test.dropna(inplace=True)  # check data consistency

    # Print Info
    print(20 * '*')
    print('% of Class 1 in Testing : ', len(dataset_test[test_label == 1]) / len(dataset_test),
          '; % of Class 0 in Testing: ', len(dataset_test[test_label == 0]) / len(dataset_test))
    print('Shape Test Set :', dataset_test.shape, ',', dataset_test.shape)
    print(20 * '*')

    # Print Performance on Test Set
    performance = direct_prediction(dataset_test, test_label,
                                    dataset_test, test_label,
                                    names, classifiers)

    print(performance)
