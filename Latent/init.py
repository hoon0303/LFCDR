
import numpy as np 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

from Latent.utils import MF, pca, predict_test

def category_list(category_all):
    category=[]
    for x in category_all:
        try:
            category.extend(x.split(', '))
        except:
            pass


    my_dict = Counter(category)
    #filtered = filter(lambda e: e[1] > 100, my_dict.items())
    filtered = sorted(my_dict.items(), key = lambda item: item[0], reverse = True)
    new_dict = dict(filtered[:100])
    #new_dict = dict(filtered)
    #print(new_dict)
    c = new_dict.keys()
    return c


class LF:
    def __init__(self, d1_review, d1_meta, category_column_name):
        self.d1_review = d1_review
        self.d1_meta = d1_meta
        self.category_column_name = category_column_name

    def extraction(self):
        self.d1_meta = self.d1_meta[self.d1_meta['asin'].isin(self.d1_review['asin'].values)]

        rating_merge = pd.merge(self.d1_meta, self.d1_review, on="asin")
        #print(rating_merge.head())

        category = category_list(rating_merge[self.category_column_name].values)

        dummies = pd.DataFrame(rating_merge, columns=category)
        dummies = dummies.fillna(0)

        for i, gen in enumerate(rating_merge[self.category_column_name].values):
            gen = str(gen)
            indices = dummies.columns.get_indexer(gen.split(', '))
            dummies.iloc[i, indices] = rating_merge['overall'][i]
        #print(dummies)

        rating = pd.concat([rating_merge,dummies], axis=1)
        #print(rating)

        Item_category_matrix = rating.groupby('asin').mean()
        Item_category_index = Item_category_matrix.index
        Item_category_matrix = Item_category_matrix[category]
        #print(Item_category_matrix[category])

        Item_category_matrix = MF(Item_category_matrix, Item_category_index, category)

        #print(Item_category_matrix)

        temp = pd.merge(self.d1_review, Item_category_matrix, on="asin")

        for i in category:
            temp[i] = temp[i] * temp['overall'] /5

        User_category_matrix = temp.groupby('reviewerID').mean()
        User_category_index = User_category_matrix.index
        User_category_matrix = User_category_matrix[category]
        #print(User_category_matrix)
        #print(Item_category_matrix)

        User_category_PCA_matrix = pca(User_category_matrix,User_category_index,['UF1', 'UF2','UF3','UF4','UF5'], "reviewerID")
        Item_category_PCA_matrix = pca(Item_category_matrix,Item_category_index,['IF1', 'IF2','IF3','IF4','IF5'], "asin")

        rating_temp = rating.merge(User_category_PCA_matrix,on="reviewerID")
        rating_temp = rating_temp.merge(Item_category_PCA_matrix,on="asin")
        rating_temp = rating_temp[['reviewerID','asin','UF1', 'UF2','UF3','UF4','UF5','IF1', 'IF2','IF3','IF4','IF5','overall']]
        rating_temp = rating_temp.rename(columns={'overall':'label'})
        rating_temp=rating_temp.fillna(0)

        return rating_temp, User_category_PCA_matrix, Item_category_PCA_matrix

    def recommend(self):
        Latent_Features, User_Features, Item_Features = self.extraction()
        data = Latent_Features[['UF1', 'UF2','UF3','UF4','UF5','IF1', 'IF2','IF3','IF4','IF5']].to_numpy()
        target = Latent_Features['label'].to_numpy()
        train_input, test_input, train_target, test_target = train_test_split(data, target,test_size=0.2, random_state = 42) # 훈련 데이터와 테스트 데이터를 나눈다.
        MAE, MSE = predict_test(train_input, test_input, train_target, test_target)

        return MAE, MSE

    