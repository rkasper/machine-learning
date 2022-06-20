import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from tabulate import tabulate

'''
According to the documentation, the evaluate() method 
was deprecated in version 1.0.5 (functionally replaced 
by model_selection.cross_validate()) and was removed in version 1.1.0, 
which is likely what you have installed.
'''
# from surprise import evaluate
from surprise.model_selection.validation import cross_validate

sns.set_style("darkgrid")

# Load the training files into memory

def readFile(file_path, rows=100000):
    data_dict = {'Cust_Id': [], 'Movie_Id': [], 'Rating': [], 'Date': []}
    f = open(file_path, "r")
    count = 0
    for line in f:
        count += 1
        if count > rows:
            break

        if ':' in line:
            movidId = line[:-2]  # remove the last character ':'
            movieId = int(movidId)
        else:
            customerID, rating, date = line.split(',')
            data_dict['Cust_Id'].append(customerID)
            data_dict['Movie_Id'].append(movieId)
            data_dict['Rating'].append(rating)
            data_dict['Date'].append(date.rstrip("\n"))
    f.close()

    return pd.DataFrame(data_dict)

# df stands for pandas.DataFrame, which is a table
print("Reading customers' movie ratings from disk...")
df1 = readFile('./data/netflix/combined_data_1.txt', rows=100000)
df2 = readFile('./data/netflix/combined_data_2.txt', rows=100000)
df3 = readFile('./data/netflix/combined_data_3.txt', rows=100000)
df4 = readFile('./data/netflix/combined_data_4.txt', rows=100000)

# df1.head()
# df2.head()
# df3.head()
# df4.head()

# Rating is a float
df1['Rating'] = df1['Rating'].astype(float)
df2['Rating'] = df2['Rating'].astype(float)
df3['Rating'] = df3['Rating'].astype(float)
df4['Rating'] = df4['Rating'].astype(float)

# Merge the data files together into one big data file
df = df1.copy()
df = df.append(df2)
df = df.append(df3)
df = df.append(df4)

df.index = np.arange(0,len(df))
df.shape
print(tabulate(df.head(10), headers='keys', tablefmt='psql'))
print("etc.")

# df.head(10)

# df['Movie_Id'].value_counts()

reader = Reader()

# load data from Pandas DataFrame and create a SVD model instance
print("Removing the date of the rating - it's irrelevant...")
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)
print(tabulate(data.df.head(10), headers='keys', tablefmt='psql'))
print("etc.")

svd = SVD()
# Run 5-fold cross-validation and print results
print("Training the model with '5-fold cross-validation'.  (This takes a while. In a real app, we would do this offline.) ...")
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Load movie titles into memory
print("Loading movies data...")
df_title = pd.read_csv('./data/netflix/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'], on_bad_lines = "warn")
# df_title.head(10)
print(tabulate(df_title.head(10), headers='keys', tablefmt='psql'))
print("etc.")

# df_785314 = df[(df['Cust_Id'] == '785314') & (df['Rating'] >= 5)]
# df_785314 = df_785314.set_index('Movie_Id')
# df_785314 = df_785314.join(df_title)['Name']
# df_785314.head(df_785314.shape[0])

# re-train the model using the entire training dataset

print("Training the model with full data set. (This takes a while. In a real app, we would do this offline.) ...")
trainset = data.build_full_trainset()
svd.fit(trainset)

# given a user (e.g., Customer Id 785314), we can use the trained model to predict the ratings given by the user on different products (i.e., Movie titles):
print("Based on the trained model, predict what user 785314 would rate the movies if he watched them. Find other people who liked the same movies as user 785314, find the movies they liked, predict how user 785314 would rate those movies, ...")
titles = df_title.copy()
titles['Estimate_Score'] = titles['Movie_Id'].apply(lambda x: svd.predict(785314, x).est)
print(tabulate(titles.head(10), headers='keys', tablefmt='psql'))
print("etc.")

# To recommend products (i.e., movies) to the given user, we can sort the list of movies in decreasing order of predicted ratings and take the top N movies as recommendations:
titles = titles.sort_values(by=['Estimate_Score'], ascending=False)
print("Recommended movies for customer 785314 (take that previous table and sort it by predicted rating):")
print(titles.head(10))