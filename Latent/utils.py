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

def gan(d1_matrix,d2_matrix):
    df = d1_matrix
   
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    df2 = d2_matrix

    X2 = df2.iloc[:, :-1].values
    y2 = df2.iloc[:, -1].values

    num_features = 10
    num_labels = len(np.unique(y)) + 1
    latent_dim = 100

    def build_generator():
        model = Sequential()
        model.add(Dense(128, input_dim=latent_dim))
        model.add(Dense(256))
        model.add(Dense(num_features, activation='linear'))

        noise = Input(shape=(latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(num_labels, latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        output = model(model_input)

        return Model([noise, label], output)

    def build_discriminator():
        img = Input(shape=(num_features,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(num_labels, num_features)(label))

        model_input = Concatenate(axis=1)([img, label_embedding])
        model = Sequential()

        model.add(Dense(128, input_dim=num_features + num_features))
        model.add(Dense(64))
        model.add(Dense(1, activation='sigmoid'))

        validity = model(model_input)

        return Model([img, label], validity)

    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    discriminator1 = build_discriminator()
    discriminator1.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

    discriminator2 = build_discriminator()
    discriminator2.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])


    z = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    img = generator([z, label])
    discriminator1.trainable = False
    discriminator2.trainable = False
    valid1 = discriminator1([img, label])
    valid2 = discriminator2([img, label])

    average_validity = (valid1 + valid2) / 2

    combined = Model([z, label], average_validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    def train(epochs, batch_size=128):
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        real_labels2 = np.ones((batch_size, 1))
        fake_labels2 = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X.shape[0], batch_size)
            imgs, labels = X[idx], y[idx]
            idx2 = np.random.randint(0, X2.shape[0], batch_size)
            imgs2, labels2 = X2[idx2], y2[idx2]

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_imgs = generator.predict([noise, labels.reshape(-1, 1)])

            d_loss_real1 = discriminator1.train_on_batch([imgs, labels.reshape(-1, 1)], real_labels)
            d_loss_fake1 = discriminator1.train_on_batch([gen_imgs, labels.reshape(-1, 1)], fake_labels)
            
            d_loss_real2 = discriminator2.train_on_batch([imgs2, labels2.reshape(-1, 1)], real_labels2)
            d_loss_fake2 = discriminator2.train_on_batch([gen_imgs, labels.reshape(-1, 1)], fake_labels)
            
            d_loss1 = 0.5 * np.add(d_loss_real1, d_loss_fake1)
            d_loss2 = 0.5 * np.add(d_loss_real2, d_loss_fake2)

            sampled_labels = np.random.randint(0, num_labels, batch_size).reshape(-1, 1)
            g_loss = combined.train_on_batch([noise, sampled_labels], real_labels)

            print(f"{epoch} [D1 loss: {d_loss1[0]}, accuracy: {100 * d_loss1[1]}] [D2 loss: {d_loss2[0]}, accuracy: {100 * d_loss2[1]}] [G loss: {g_loss}]")

    train(epochs=1000)


    def generate_samples(num_samples, labels):
        noise = np.random.normal(0, 1, (num_samples, latent_dim))
        gen_data = generator.predict([noise, labels.reshape(-1, 1)])
        return gen_data

    num_samples_to_generate=10000
    labels_to_generate = np.array([i % (num_labels - 1 ) for i in range(num_samples_to_generate)]) + 1

    generated_data = generate_samples(num_samples=len(labels_to_generate), labels=labels_to_generate)
    generated_df = pd.DataFrame(generated_data, columns=[f'feature_{i}' for i in range(generated_data.shape[1])])

    test_input = d2_matrix.iloc[:, :-1].to_numpy()
    test_target = d2_matrix.iloc[:, -1].to_numpy()

    gen_data = generated_df.to_numpy()

    data = np.concatenate((test_input, gen_data), axis=0)
    target = np.concatenate((test_target, labels_to_generate), axis=0)

    train_input = d1_matrix.iloc[:, :-1].to_numpy()
    train_target = d1_matrix.iloc[:, -1].to_numpy()

    data = np.concatenate((train_input, data), axis=0)
    target = np.concatenate((train_target, target), axis=0)

    return None, data, target
