import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib.pyplot import figure
import pickle
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def gradient_step_stah(X,y,w,alpha,ind):
        return w - (alpha * 2.0 / X.shape[0])*X[ind] * (np.dot(X[ind], w)-y[ind])

def sgd(X_train, y_train, X_test, y_test, w, alpha = 1e-4, max_it = 10e6):
    iter_num = 0
    errors = []
    errors_test = []
    r2 = []
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values
    while (iter_num < max_it):
        ind = np.random.randint(X_train.shape[0])
        w = gradient_step_stah(X_train, y_train, w, alpha, ind)
        if iter_num%(int(max_it/20))==0:
            print('Выполнено:', int(iter_num/max_it * 100), '%')
            error = mean_squared_error(y_train, np.dot(X_train, w))
            errors.append(error)
            print('Mse train:', error)
            error = mean_squared_error(y_test,np.dot(X_test, w))
            errors_test.append(error)
            print('Mse test:', error)
            R = r2_score(y_test,np.dot(X_test, w))
            r2.append(R)
            print('R2:', R)
        iter_num += 1
    return w, errors, errors_test, r2

def main(IsLearn = True):
    df = pd.read_csv('advertising.csv')
    print(df.head())
    sns.clustermap(df.corr())
    plt.show()
    df = df.sample(frac=1).reset_index(drop=True)
    print(len(df))
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Sales', axis=1), df[['Sales']], test_size=0.3)
    print(X_train.shape[0])
    print(y_train.shape[0])
    if IsLearn:
        w, mse_train, mse_test, r2 = sgd(X_train, y_train, X_test, y_test, np.ones(X_train.shape[1]))
        with open('file.pkl', 'wb') as file: 
            pickle.dump((w, mse_train, mse_test, r2), file)
    else:
        with open('file.pkl', 'rb') as file: 
            w, mse_train, mse_test, r2 = pickle.load(file) 
    print(np.dot(X_test[:10], w) - y_test[:10].values.T)

    plt.figure()
    plt.grid()
    plt.ylim(ymax=10)
    plt.plot(mse_train, label = 'train')
    plt.plot(mse_test, label = 'test')
    plt.legend()

    plt.figure()
    plt.grid()
    plt.ylim(ymin = 0)
    plt.plot(r2, label = 'r2')
    plt.legend()
    plt.show()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = make_pipeline(StandardScaler(), SGDRegressor())
        reg.fit(X_train.values, y_train.values.ravel())
        print('Mse sgd (sklearn): ', r2_score(y_test, reg.predict(X_test)))
        print(reg.predict(X_test)[:10] - y_test[:10].values.T)

if __name__ == "__main__":
    main(False)