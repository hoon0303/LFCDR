import pandas as pd
from Latent.init import LF
from Latent.heterogenous import Heterogenous_domain_recommend


d1_review = pd.read_csv("data/food.csv")[['reviewerID','asin','overall']]
d1_meta = pd.read_csv("data/food_meta.csv")[['asin','category']]

d2_review = pd.read_csv("data/movie and tv.csv")[['reviewerID','asin','overall']]
d2_meta = pd.read_csv("data/movie and tv_meta.csv")[['asin','category']]

d4_review = pd.read_csv("data/video game.csv")[['reviewerID','asin','overall']]
d4_meta = pd.read_csv("data/video game_meta.csv")[['asin','category']]

d5_review = pd.read_csv("data/books.csv")[['reviewerID','asin','overall']]
d5_meta = pd.read_csv("data/book_meta.csv")[['asin','category']]


x = 0.8
y = 5
df = pd.DataFrame(columns=['source','target','user_size','train_size','MAE','MSE'])

temp = 0.2

for i in range(1,5):
    x = temp * i
    print(y, x)

    MAE, MSE = Heterogenous_domain_recommend(d2_review, d2_meta,d1_review, d1_meta,k = y, train_size_num=x)
    df = df.append({'source': 'Movies_and_TV','target':'Grocery_and_Gourmet_Food','user_size':y,'train_size':x, 'MAE': MAE, 'MSE': MSE},ignore_index=True)

    MAE, MSE = Heterogenous_domain_recommend(d2_review, d2_meta,d4_review, d4_meta,k = y, train_size_num=x)
    df = df.append({'source': 'Movies_and_TV','target':'Video_Games','user_size':y,'train_size':x, 'MAE': MAE, 'MSE': MSE},ignore_index=True)

    MAE, MSE = Heterogenous_domain_recommend(d2_review, d2_meta,d5_review, d5_meta,k = y, train_size_num=x)
    df = df.append({'source': 'Movies_and_TV','target':'Books','user_size':y, 'train_size':x,'MAE': MAE, 'MSE': MSE},ignore_index=True)
print(df)


df = pd.DataFrame(columns=['source','target','user_size','train_size','MAE','MSE'])

for i in range(1,5):
    x = temp * i
    print(y, x)

    MAE, MSE = Heterogenous_domain_recommend(d5_review, d5_meta,d1_review, d1_meta,k = y, train_size_num=x)
    df = df.append({'source': 'Books','target':'Grocery_and_Gourmet_Food','user_size':y,'train_size':x, 'MAE': MAE, 'MSE': MSE},ignore_index=True)

    MAE, MSE = Heterogenous_domain_recommend(d5_review, d5_meta,d2_review, d2_meta,k = y, train_size_num=x)
    df = df.append({'source': 'Books','target':'Movies_and_TV','user_size':y, 'train_size':x,'MAE': MAE, 'MSE': MSE},ignore_index=True)

    MAE, MSE = Heterogenous_domain_recommend(d5_review, d5_meta,d4_review, d4_meta,k = y, train_size_num=x)
    df = df.append({'source': 'Books','target':'Video_Games','user_size':y,'train_size':x, 'MAE': MAE, 'MSE': MSE},ignore_index=True)

print(df)
