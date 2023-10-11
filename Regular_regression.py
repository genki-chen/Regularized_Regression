import os
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, ElasticNetCV, OrthogonalMatchingPursuitCV, LarsCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('./cc.csv',index_col=0)
    data_x = df.values 
    df = pd.read_csv('./price.csv',index_col=0)
    data_y = df.values
    data_y = np.log1p(data_y)
    #data_y = data_y - np.mean(data_y)
    scaler = StandardScaler()  
    #scaler = MinMaxScaler()
    scaler.fit(data_x)
    x_scal = scaler.transform(data_x)
    x_train, x_test, y_train, y_test = train_test_split(x_scal,data_y,test_size=0.7,random_state=12)
    clf_ridge = RidgeCV(alphas=np.linspace(1e-3,1000,100)).fit(x_train,y_train)
    #clf_ridge = Ridge(alpha=0.01).fit(x_train,y_train)
    print("Ridge alpha = " + str(np.round(clf_ridge.alpha_,4)))
    #print(clf_ridge.coef_)
    print("Ridge score = "+ str(np.round(clf_ridge.score(x_test,y_test),4)))
    clf_lasso = LassoCV(cv=5,random_state=0).fit(x_train,y_train.ravel())
    #print(clf_lasso.coef_)
    print("Lasso alpha = " + str(np.round(clf_lasso.alpha_,4)))
    print("Lasso score = " + str(np.round(clf_lasso.score(x_test,y_test.ravel()),4)))

    clf_elasticNet = ElasticNetCV(l1_ratio=np.linspace(0.01,1,11),random_state=0).fit(x_train,y_train.ravel())
    print("ElasticNet alpha = " + str(np.round(clf_elasticNet.alpha_,4)))
    print("ElasticNet l1_ratio = " + str(np.round(clf_elasticNet.l1_ratio_,2)))
    print("ElasticNet score = " + str(np.round(clf_elasticNet.score(x_test,y_test.ravel()),4)))


    #clf_lars = LarsCV().fit(x_train,y_train.ravel())
    #print(clf_lars.score(x_test,y_test.ravel()))
    clf_omp = OrthogonalMatchingPursuitCV().fit(x_train,y_train.ravel())
    print(clf_omp.score(x_test,y_test.ravel()))

    """
    np.random.seed = 41014 
    #x = np.linspace(-20,10,2000).reshape(-1,1)
    x = 10*np.random.normal(0,1,1000).reshape(-1,1)
    def Y(x):
        return 1.4*x**2+4.3*x + 3.7
    y_clear = Y(x)
    y_noise = y_clear + np.random.normal(0,1,size=len(x)).reshape(-1,1)
    poly = PolynomialFeatures(degree=10).fit(x)
    x_poly = poly.transform(x)
    scaler = StandardScaler()  
    #scaler = MinMaxScaler()
    scaler.fit(x_poly)
    x_scal = scaler.transform(x_poly)
    x_train, x_test, y_train, y_test = train_test_split(x_scal,y_noise,test_size=0.3,random_state=52)
    #clf = RidgeCV(alphas=np.linspace(1e-4,5,1000),cv=5).fit(x_train,y_train)
    #print(clf.alpha_)
    #print(clf.coef_)
    #print(clf.score(x_test,y_test))
    clf = LassoCV(random_state=0).fit(x_train,y_train.ravel())
    print(clf.coef_)
    print(clf.alpha_)
    print(clf.score(x_test,y_test.ravel()))
    new_x = np.linspace(1000,2000,20).reshape(-1,1)
    new_x_poly = poly.transform(new_x)
    new_x_scal = scaler.transform(new_x_poly)
    new_y_true = Y(new_x)
    print(clf.score(new_x_scal,new_y_true.ravel()))

    clf_elasticNet = ElasticNetCV(l1_ratio=np.linspace(0.01,1,11),random_state=0).fit(x_train,y_train.ravel())
    print("ElasticNet alpha = " + str(np.round(clf_elasticNet.alpha_,4)))
    print("ElasticNet l1_ratio = " + str(np.round(clf_elasticNet.l1_ratio_,2)))
    print("ElasticNet score = " + str(np.round(clf_elasticNet.score(x_test,y_test.ravel()),4)))
    print(clf_elasticNet.score(new_x_scal,new_y_true.ravel()))


    print("OMP")

    clf_omp = OrthogonalMatchingPursuitCV().fit(x_train,y_train.ravel())
    print(clf_omp.coef_)
    print(clf_omp.score(x_test,y_test.ravel()))
    print(clf_omp.score(new_x_scal,new_y_true.ravel()))

    print("LARS")
    clf_lars = LarsCV().fit(x_train,y_train.ravel())
    print(clf_lars.coef_)
    print(clf_lars.score(x_test,y_test.ravel()))
    print(clf_lars.score(new_x_scal,new_y_true.ravel()))
    """