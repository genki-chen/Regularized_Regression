import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    ## 预设置
    test_rate_list = np.linspace(0.95,0.05,20)
    repeat_num = 1000
    ## 读取数据
    df = pd.read_csv('./data_precode.csv',index_col=0)
    ## 数据预处理
    df['Sale_Price'] = np.log1p(df['Sale_Price'])
    df = df.drop(df[(df['Gr_Liv_Area']>4000)&(df['Sale_Price']<12.5)].index)
    df = df.drop(df[(df['Gr_Liv_Area']<1000)&(df['Sale_Price']<10)].index)
    data_raw = df.loc[:,df.columns != 'Sale_Price'].values 
    val_title = df.loc[:,df.columns != 'Sale_Price'].columns
    data_y = df['Sale_Price'].values
    scaler = MinMaxScaler()
    scaler.fit(data_raw)
    x_scal = scaler.transform(data_raw)
    MSE_arr = np.zeros((len(test_rate_list),repeat_num,3))
    for idx_i in range(len(test_rate_list)):
        for idx_j in range(repeat_num):
            x_total, _,y_total, _ = train_test_split(x_scal,data_y,test_size=test_rate_list[idx_i])
            ## 数据分成训练集，测试集
            x_train, x_test, y_train, y_test = train_test_split(x_total,y_total,test_size=0.3)
            ## 岭回归
            clf_ridge = Ridge(alpha=2.381).fit(x_train,y_train)
            y_test_pre = clf_ridge.predict(x_test)
            MSE_arr[idx_i,idx_j,0] = mean_squared_error(y_test,y_test_pre)
            ## Lasso回归
            clf_lasso = Lasso(alpha=0.0002,max_iter=2000).fit(x_train,y_train)
            y_test_pre = clf_lasso.predict(x_test)
            MSE_arr[idx_i,idx_j,1] = mean_squared_error(y_test,y_test_pre)
            ## 弹性网
            clf_elasticNet = ElasticNet(l1_ratio=0.55,alpha=0.0003).fit(x_train,y_train)
            y_test_pre = clf_elasticNet.predict(x_test)
            MSE_arr[idx_i,idx_j,2] = mean_squared_error(y_test,y_test_pre)
    np.save('./data.npy',MSE_arr)
    MSE_mean = np.mean(MSE_arr,axis=1)
    print(MSE_mean.shape)
    fig = plt.figure(figsize=(16,10),dpi=300)
    plt.plot((1-test_rate_list)*2930,MSE_mean[:,0],lw=2,ls='-',color='r',label='Ridge')
    plt.plot((1-test_rate_list)*2930,MSE_mean[:,1],lw=2,ls='-.',color='g',label='Lasso')
    plt.plot((1-test_rate_list)*2930,MSE_mean[:,2],lw=2,ls='--',color='b',label='ElasticNet')
    plt.xlabel('Data Set size')
    plt.ylabel('MSE')
    plt.legend(loc=1,fontsize='large',framealpha=0.7)
    plt.grid(True,which='both',linestyle='--')
    plt.savefig('./10.png')
