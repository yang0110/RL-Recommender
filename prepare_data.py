import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
import pyarrow.parquet as pq
from sklearn.preprocessing import OrdinalEncoder
from collections import defaultdict
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler

train = pq.read_table('train.parquet').to_pandas()
test = pq.read_table('test.parquet').to_pandas()

user_count = train.groupby('uid').count().reset_index()
users = user_count[user_count.pic_id>=1].uid.values

users = np.random.choice(users, size=10000)
sub_train = train[train.uid.isin(users)]
sub_test = test[test.uid.isin(users)]

sub_train.to_csv('sub_train.csv', index=False)
sub_test.to_csv('sub_test.csv', index=False)


# sub_train = pd.read_csv('sub_train.csv')
# sub_test = pd.read_csv('sub_test.csv')

#  reindex user and pictures
concat_data = pd.concat([sub_train, sub_test])

oe = OrdinalEncoder()
new_uid = oe.fit_transform(concat_data['uid'].values.reshape(-1,1)).astype(int)
concat_data['uid'] = new_uid

new_pic_id = oe.fit_transform(concat_data['pic_id'].values.reshape(-1,1)).astype(int)
concat_data['pic_id'] = new_pic_id

concat_data.to_csv('concat_data.csv', index=False)
concat_data_no_feature = concat_data[['uid', 'pic_id', 'label', 'day']].copy()
concat_data_no_feature.to_csv('concat_data_no_feature.csv', index=False)


# concat_data = pd.read_csv('concat_data.csv')

#  create pic features
pics = concat_data.pic_id.unique()
pic_data = concat_data.drop_duplicates(subset=['pic_id'])
pic_data = pic_data.drop(['device_model', 'uid', 'day','label'], axis=1)
pic_data.index = pic_data.pic_id.values

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(pic_data.iloc[:,1:])

scaled_df = pd.DataFrame(scaled_features, columns=pic_data.columns[1:])
# scaled_df['pic_id'] = pic_data.pic_id.values

scaled_df.to_csv('scaled_pic_features.csv', index=False)


