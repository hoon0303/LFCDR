import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from Latent.utils import merge_matrix, merge_matrix2, merge_matrix3, gan
from Latent.init import LF
from Latent.utils import predict_test

def Heterogenous_domain_recommend(d1_review, d1_meta, d2_review, d2_meta, k = 0, train_size_num = 0.8):
    
    d1_user = d1_review.drop_duplicates('reviewerID')
    d2_user = d2_review.drop_duplicates('reviewerID')

    D1_count = d1_review['reviewerID'].value_counts() 
    D1_K_user = D1_count[D1_count>=k]

    D2_count = d2_review['reviewerID'].value_counts() 
    D2_K_user = D2_count[D2_count>=k]

    d1_review = d1_review[d1_review['reviewerID'].isin(D1_K_user.index)]
    d2_review = d2_review[d2_review['reviewerID'].isin(D2_K_user.index)]

    D1_LF, D1_user_LF, D1_Item_LF = LF(d1_review, d1_meta,'category').extraction()
    D2_LF, D2_user_LF, D2_Item_LF = LF(d2_review, d2_meta,'category').extraction()


    col = ['UF1', 'UF2','UF3','UF4','UF5','IF1', 'IF2','IF3','IF4','IF5']

    test_x = D2_LF[col].to_numpy()
    test_y = D2_LF["label"].to_numpy()

    train_input,test_input, train_target,test_target = train_test_split(test_x,test_y, random_state=30, shuffle =  test_y, train_size = train_size_num)

    D2_train = pd.DataFrame(train_input,columns=col)
    D2_train["label"] = train_target

    F_list, data, target = merge_matrix2(D1_LF,D2_train)
    test_input = pd.DataFrame(test_input,columns=col)[F_list].to_numpy()
    

    MAE, MSE = predict_test(data, test_input, target, test_target)
    print(MAE)
    print(MSE)
    return MAE, MSE

