import numpy as np
#from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, OrthogonalMatchingPursuitCV, LarsCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
#import matplotlib.pyplot as plt
from operator import itemgetter


if __name__ == '__main__':
    ## 预设置
    show_ridge_b = False
    show_lasso_b = False
    show_ElasticNet_b = True
    show_Top_num = 20
    ## 读取数据
    df = pd.read_csv('./x_data_precode.csv',index_col=0)
    data_raw = df.values 
    val_title = df.columns
    df = pd.read_csv('./price.csv',index_col=0)
    price = df.values.reshape(-1,)
    """
    df = pd.read_csv('./raw_data.csv',index_col=0)
    df.fillna('None',inplace=True)
    data_raw = df.loc[:,df.columns != 'Sale_Price'].values
    val_title = df.loc[:,df.columns != 'Sale_Price'].columns
    price = df['Sale_Price'].values
    ## 数据标签化
    _,n = data_raw.shape
    le = LabelEncoder()
    for idx in range(n):
        if type(data_raw[0,idx]) == str:
            le.fit(data_raw[:,idx])
            data_raw[:,idx] = le.transform(data_raw[:,idx]) + 1
    """
    ## 数据预处理
    data_y = np.log1p(price)
    #scaler = StandardScaler()  
    scaler = MinMaxScaler()
    scaler.fit(data_raw)
    x_scal = scaler.transform(data_raw)
    ## 数据分成训练集，测试集
    x_train, x_test, y_train, y_test = train_test_split(x_scal,data_y,test_size=0.3,random_state=1)
    ## OLS
    clf_ols = LinearRegression().fit(x_train,y_train)
    print("OLS score = "+str(np.max([np.round(clf_ols.score(x_test,y_test),4),0])))
    print("===========================================")
    ## 岭回归
    #clf_ridge = RidgeCV(alphas=np.logspace(-3,2,500)).fit(x_train,y_train)
    clf_ridge = Ridge(alpha=7).fit(x_train,y_train)
    if show_ridge_b:
        Ridge_coef = clf_ridge.coef_
        coef_dict = {}
        idx = 0
        for var in Ridge_coef:
            coef_dict[val_title[idx]] = np.abs(var)
            idx += 1
        sorted_dict = dict(sorted(coef_dict.items(), key=itemgetter(1),reverse=True))
        count = 1  # 用于计数前10个键值对
        for key, value in sorted_dict.items():
            print(str(count)+' '+key+' '+str(np.round(value,4)))
            count += 1
            if count > show_Top_num:
                break
    #print("Ridge alpha = " + str(np.round(clf_ridge.alpha_,4)))
    print("Ridge score = "+ str(np.round(clf_ridge.score(x_test,y_test),4)))
    print("===========================================")
    ## Lasso回归
    #clf_lasso = LassoCV(cv=5,random_state=0,alphas=np.logspace(-4,2,100),max_iter=10000,n_jobs=-1).fit(x_train,y_train)
    clf_lasso = Lasso(alpha=0.0005).fit(x_train,y_train)
    if show_lasso_b:
        Lasso_coef = clf_lasso.coef_
        coef_dict = {}
        idx = 0
        for var in Lasso_coef:
            coef_dict[val_title[idx]] = np.abs(var)
            idx += 1
        sorted_dict = dict(sorted(coef_dict.items(), key=itemgetter(1),reverse=True))
        count = 1  # 用于计数前10个键值对
        for key, value in sorted_dict.items():
            print(str(count)+' '+key+' '+str(np.round(value,4)))
            count += 1
            if count > show_Top_num:
                break
    #print("Lasso alpha = " + str(np.round(clf_lasso.alpha_,4)))
    print("Lasso score = " + str(np.round(clf_lasso.score(x_test,y_test),4)))
    print("===========================================")
    ## 弹性网
    #clf_elasticNet = ElasticNetCV(l1_ratio=np.linspace(0.1,1,11),alphas=np.logspace(-4,2,100),random_state=0,max_iter=10000,n_jobs=-1).fit(x_train,y_train)
    clf_elasticNet = ElasticNet(l1_ratio=0.05,alpha=0.0019).fit(x_train,y_train)
    if show_ElasticNet_b:
        ElasticNet_coef = clf_elasticNet.coef_
        coef_dict = {}
        idx = 0
        for var in ElasticNet_coef:
            coef_dict[val_title[idx]] = np.abs(var)
            idx += 1
        sorted_dict = dict(sorted(coef_dict.items(), key=itemgetter(1),reverse=True))
        count = 1  # 用于计数前20个键值对
        for key, value in sorted_dict.items():
            print(str(count)+' '+key+' '+str(np.round(value,4)))
            count += 1
            if count > show_Top_num:
                break
    #print("ElasticNet alpha = " + str(np.round(clf_elasticNet.alpha_,4)))
    #print("ElasticNet l1_ratio = " + str(np.round(clf_elasticNet.l1_ratio_,2)))
    print("ElasticNet score = " + str(np.round(clf_elasticNet.score(x_test,y_test),4)))

    #clf_lars = LarsCV(eps=1e-2).fit(x_train,y_train.ravel())
    #print(clf_lars.score(x_test,y_test.ravel()))
    #clf_omp = OrthogonalMatchingPursuitCV().fit(x_train,y_train.ravel())
    #print(clf_omp.score(x_test,y_test.ravel()))

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