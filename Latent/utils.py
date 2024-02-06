import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from keras.layers import Input, Embedding, multiply, Dense, Flatten, Concatenate
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.neural_network import MLPRegressor
from itertools import permutations

def get_rmse(R, P, Q, non_zeros):

    error = 0 
    full_pred_matrix = np.dot(P, Q.T)

    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros] 
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros] 
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind] 

    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind] 


    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros) 
    rmse = np.sqrt(mse) 

    return rmse

def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda = 0.01): 
    num_users, num_items = R.shape

    np.random.seed(1) 
    P = np.random.normal(scale=1./K, size=(num_users, K)) 
    Q = np.random.normal(scale=1./K, size=(num_items, K)) 

    break_count = 0 

    non_zeros = [ (i, j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0 ] 

    for step in range(steps): 
        for i, j, r in non_zeros: 

            eij = r - np.dot(P[i, :], Q[j, :].T) 
            P_temp = P[i,:] + learning_rate*(eij * Q[j, :] - r_lambda*P[i,:]) 
            Q_temp = Q[j,:] + learning_rate*(eij * P[i, :] - r_lambda*Q[j,:]) 
            P[i,:] = P_temp
            Q[j,:] = Q_temp
        

    rmse = get_rmse(R, P, Q, non_zeros) 
    if (step % 10) == 0 : 
        print("### iteration step : ", step," rmse : ", rmse) 
    return P, Q

def MF(matrix, matrix_index, matrix_column):
    P, Q = matrix_factorization(matrix.values, K=50, steps=200, learning_rate=0.01, r_lambda = 0.01) 
    pred_matrix = np.dot(P, Q.T)
    matrix = pd.DataFrame(pred_matrix, index = matrix_index, columns=matrix_column)

    return matrix

from sklearn.decomposition import PCA

def pca(matrix,matrix_index,matrix_column,index_name):
    pca = PCA(n_components=5)
    printcipalComponents = pca.fit_transform(matrix)
    PCA_matrix = pd.DataFrame(data=printcipalComponents, index = matrix_index, columns = matrix_column)
    PCA_matrix.index.name=index_name

    return PCA_matrix

def predict_test(train_input, test_input, train_target, test_target):

    ss = StandardScaler()
    train_scaled = ss.fit_transform(train_input)
    test_scaled = ss.transform(test_input)

    train_target = train_target.reshape(-1,1)
    test_target = test_target.reshape(-1,1)

    ss2 = StandardScaler()
    train_target_scaled = ss2.fit_transform(train_target)
    test_target_scaled = ss2.transform(test_target)

    train_target = train_target.reshape(-1)
    test_target = test_target.reshape(-1)

    train_target_scaled = train_target_scaled.reshape(-1)
    test_target_scaled = test_target_scaled.reshape(-1)

    lr = RandomForestRegressor()
    lr.fit(train_scaled, train_target_scaled)


    y_test_pred = lr.predict(test_scaled)

    MSE = mean_squared_error(test_target_scaled, y_test_pred)
    MAE = mean_absolute_error(test_target_scaled, y_test_pred)

    y_test_pred = y_test_pred.reshape(-1,1)
    y_test_pred = ss2.inverse_transform(y_test_pred)

    return MAE, MSE

def merge_matrix2(d1_matrix,d2_matrix):

    col = ['UF1', 'UF2','UF3','UF4','UF5','IF1', 'IF2','IF3','IF4','IF5']

    train_input = d1_matrix[col].to_numpy()
    train_target = d1_matrix['label'].to_numpy()

    col = ['UF1', 'UF2','UF3','UF4','UF5','IF1', 'IF2','IF3','IF4','IF5']

    test_input = d2_matrix[col].to_numpy()
    test_target = d2_matrix['label'].to_numpy()


    F_data = ['UF1', 'UF2','UF3','UF4','UF5','IF1', 'IF2','IF3','IF4','IF5']

    MAE_list = list()
    F_list = list()
    for i in permutations(['UF1','UF2','UF3','UF4','UF5'], 5):
        for j in permutations(['IF1', 'IF2','IF3','IF4','IF5'], 5):
            F_data = list(i+j)

            test_input = d2_matrix[F_data].to_numpy()

            from sklearn.preprocessing import StandardScaler
            ss = StandardScaler()
            train_scaled = ss.fit_transform(train_input)
            test_scaled = ss.transform(test_input)

            train_target = train_target.reshape(-1,1)
            test_target = test_target.reshape(-1,1)

            ss2 = StandardScaler()
            train_target_scaled = ss2.fit_transform(train_target)
            test_target_scaled = ss2.transform(test_target)

            train_target = train_target.reshape(-1)
            test_target = test_target.reshape(-1)

            train_target_scaled = train_target_scaled.reshape(-1)
            test_target_scaled = test_target_scaled.reshape(-1)

            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor

            lr = LinearRegression() # # 선형 회귀 훈련
            lr.fit(train_scaled, train_target_scaled)

            y_test_pred = lr.predict(test_scaled)

            from sklearn.metrics import mean_absolute_error
            MAE_list.append(mean_absolute_error(test_target_scaled, y_test_pred))

            F_list.append([F_data[0],F_data[1],F_data[2],F_data[3],F_data[4],F_data[5],F_data[6],F_data[7],F_data[8],F_data[9]])

    tmp = min(MAE_list)
    index = MAE_list.index(tmp)

    print(F_list[index])

    train_input = d1_matrix[['UF1', 'UF2','UF3','UF4','UF5','IF1', 'IF2','IF3','IF4','IF5']].to_numpy()
    test_input = d2_matrix[F_list[index]].to_numpy()
    train_target = d1_matrix['label'].to_numpy()
    test_target = d2_matrix['label'].to_numpy()

    data = np.concatenate((train_input, test_input), axis=0)
    target = np.concatenate((train_target, test_target), axis=0)

    return F_list[index], data, target