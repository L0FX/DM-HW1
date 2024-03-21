

"""1.数据集“Movies Dataset from Pirated Sites”的数据分析"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

"""以下为代码中用到的function定义："""
#绘制标称属性柱状图
def bar_plot(data, feature_list):
  for col in feature_list:
    print(data[col].value_counts())
    v = list(data[col].value_counts().values)
    id = list(data[col].value_counts().index)

    bar = plt.barh(id[:20],v[:20])
    plt.yticks(rotation = 45, size = 7)
    plt.xticks(rotation = 45, size = 7)
    plt.bar_label(bar, v[:20])
    plt.show()
#绘制数值属性盒图
def box_plot(data, feature_list):
  l = []
  for col in feature_list:
    ll = list(data[col])
    l.append(ll)

  plt.grid(True)
  plt.boxplot(l,
        medianprops={'color': 'red', 'linewidth': '1.5'},
        meanline=True,
        showmeans=True,
        meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
        flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 7},
        labels=feature_list)
  plt.show()

# 绘制数值属性的直方图
def hist_plot(data, feature_list):
  for col in feature_list:
    data[col].hist(bins=12,histtype="bar",alpha=0.5)
    plt.title(f'Distribution of {col}')
    plt.show()

#展示新旧数据盒图对比
def box_compare(data1,data2, feature_list):
  for col in feature_list:
    l1 = list(data1[col])
    l2 = list(data2[col])
    labels = [f'{col} of drop_data',f'{col} of fill_data']
    plt.grid(True)
    plt.boxplot([l1,l2],
        medianprops={'color': 'red', 'linewidth': '1.5'},
        meanline=True,
        showmeans=True,
        meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
        flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 7},
        labels=labels)
    plt.show()

#展示新旧数据的直方图对比
def process_show(old_data, new_data, feature):
  feature_count = pd.DataFrame(old_data[feature].value_counts()).rename(columns={feature: 'feature_count'}).sort_values(by='feature_count', ascending=True)
  feature_count_cleaned = feature_count
  feature_count_cleaned['feature_count_cleaned'] = [0] * len(feature_count)

  for level in list(feature_count.index):
    if level in list(new_data[feature].value_counts().index):
      feature_count_cleaned.loc[[level], ['feature_count_cleaned']] = new_data[feature].value_counts().loc[[level]].values[0]
  plt.figure(figsize=(20, 20))
  bar1 = plt.barh([d+0.42 for d in list(range(len(feature_count_cleaned)))], tick_label=feature_count_cleaned.index, width=feature_count_cleaned['feature_count'], label='feature_count', height=0.4)
  bar2 = plt.barh(list(range(len(feature_count_cleaned))), tick_label=feature_count_cleaned.index, width=feature_count_cleaned['feature_count_cleaned'], label='feature_count_cleaned', height=0.4)
  plt.yticks(rotation = 45, fontsize=20)
  plt.xticks(rotation = 45, fontsize=20)
  plt.bar_label(bar1,feature_count_cleaned['feature_count'],fontsize=16)
  plt.bar_label(bar2,feature_count_cleaned['feature_count_cleaned'], fontsize=16)
  plt.title(f"Comparison of {feature}", fontsize=24)
  plt.legend(fontsize=24, loc='lower right')
  plt.show()

#用最高频率值来填补缺失值
def fill_highest_fre(data, feature_list):
  dict = {}
  for col in feature_list:
    top = data[col].value_counts().index
    dict[col] = list(top)[0]
  new_data = data.fillna(dict)
  return new_data

#用相关属性填充缺失值
def feature_corr_fill(data,fill_feature,corr_feature):
  mean_df = data.groupby(fill_feature).agg(avg=(corr_feature, 'mean'))
  corr_data = data.copy(deep=True)
  for i in trange(len(corr_data)):
    if corr_data[fill_feature].iloc[i] is np.nan:
      rate = corr_data[corr_feature].iloc[i]
      dists = []
      for j in range(len(mean_df)):
        dists.append(abs(mean_df.iloc[j]['avg']-rate))
      array = np.array(dists)
      sorted_index = np.argsort(array)
      idx = sorted_index[0]
      corr_data[fill_feature].iloc[i] = mean_df.index[idx]
  return corr_data

#针对缺失率最高的appropriate_for属性，我们尝试使用计算数据对象间的相似性来填补缺失值。
#首先对数值属性进行归一化（regularit函数）；接着，计算各个例子的数值属性的均方差，找到最相似的例子的appropriate_for的取值填充。
def sample_corr_fill(data, fill_feature, feature_list):

  def regularit(data,feature_list):
    new_df = pd.DataFrame(index=data.index)
    for col in feature_list:
        d = data[col]
        MAX = d.max()
        MIN = d.min()
        new_df[col] = ((d - MIN) / (d - MAX))
    return new_df

  def eucliDist(X,Y):
    return np.sqrt(sum(np.power((X - Y), 2)))
  def manhDist(X,Y):
    return np.sum(abs(X-Y))

  reg_data = regularit(data,feature_list)
  print("Show regularized data:")
  print(reg_data.head(20))
  no_nan_data = pd.concat([reg_data,data[fill_feature]],axis=1).dropna(axis=0,inplace=False)

  nan_data = pd.concat([reg_data,data[fill_feature]], axis=1)
  nan_data = nan_data[nan_data[fill_feature].isnull() == True]
  #print(nan_data.head(20))
  sample_corr_data = data.copy(deep=True)
  nan_index = nan_data.index
  no_nan_index = no_nan_data.index
  for id_nan in tqdm(nan_index,total=len(nan_data),desc='Calculating similarity...'):
    dists = {}
    for id_sample in no_nan_index:
      dist = manhDist(np.array(reg_data.loc[id_nan]),np.array(reg_data.loc[id_sample]))
      dists[id_sample] = dist
    order = sorted(dists.items(),key=lambda x:x[1],reverse=False)
    idx = order[0][0]
    sample_corr_data[fill_feature].loc[id_nan] = sample_corr_data[fill_feature].loc[idx]
  return sample_corr_data

"""1.1 将数据读入为DataFrame，并展示数据中的数据类型"""
#此处可修改路径
movie_data = pd.read_csv('/content/data/MyDrive/data/movies_dataset.csv')
print(movie_data.head(5))
print(movie_data.dtypes)


"""1.2 处理部分属性数据

可删去“Unnamed: 0”这个与数据分析无关的属性列。
"""

movie_data = movie_data.drop(['Unnamed: 0'],axis = 1)

#转换downloads和views
for col in 'downloads', 'views':
  movie_data[col] = movie_data[col].str.replace(',','')
  movie_data[col] = movie_data[col].astype('float')

#转换run_time为以min为单位的float型数据
new = []
for item in movie_data['run_time']:
  if "h" in str(item) and ("min" in str(item) or "m" in str(item)):
    item = item.strip('min')
    hour, min = item.split('h ')
    time = float(hour)*60 + float(min)
    new.append(time)
  elif "h" in str(item):
    hour = item.strip('h')
    time = float(hour)*60
    new.append(time)
  elif "min" in str(item) or "m" in str(item):
    min = item.strip('min')
    time = float(min)
    new.append(time)
  else:
    new.append(float(item))
movie_data['run_time'] = new

#将“posted_date”和“release_date”转为datetime数据
movie_data['posted_date'] = pd.to_datetime(movie_data['posted_date'])
movie_data['release_date'] = pd.to_datetime(movie_data['release_date'])

"""1.2 展示数据摘要

（1）首先，检查统计的电影是否有重复，进行去重处理，将数值属性数据的最大值保留。
"""

print(movie_data['id'].value_counts())

non_unique_id_mov_data = movie_data.groupby('id').agg(id_count=('id', 'count')).query('id_count > 1').sort_values(by='id_count', ascending=False).index.to_list()
for id in non_unique_id_mov_data:
  non_unique_id_df = movie_data.query('id == @id').groupby('id').agg(max_downloads=('downloads', 'max'), max_views=('views', 'max'))
  max_downloads = non_unique_id_df['max_downloads'].iloc[0]
  max_views = non_unique_id_df['max_views'].iloc[0]

  movie_data.loc[movie_data['id'] == id, 'downloads'] = max_downloads
  movie_data.loc[movie_data['id'] == id, 'views'] = max_views
movie_data = movie_data.drop_duplicates()

"""（2）接着，展示标称属性的值和频数统计,此处将给出每种可能值的频数统计，并且将展示频数的TOP20的柱状图。"""

nominal = ['appropriate_for','director','industry','language','writer','title','run_time','posted_date','release_date']

"""对于“appropriate_for”中的Unrated和Not Rated进一步处理："""

movie_data['appropriate_for'] = movie_data['appropriate_for'].replace('Unrated', 'Not Rated')

"""接着画出柱状图："""
bar_plot(movie_data,nominal[:-2])

"""统计“language”属性中具体语言种类与频数："""

mov_language = movie_data['language'].astype('str')

for i in range(len(mov_language)):
  mov_language.iloc[i] = mov_language.iloc[i].split(',')
  for j in range(len(mov_language.iloc[i])):
    mov_language.iloc[i][j] = mov_language.iloc[i][j].lstrip(' ')

mov_language_dict = {}

for i in range(len(mov_language)):
  for j in range(len(mov_language.iloc[i])):
    if mov_language.iloc[i][j] not in mov_language_dict:
        mov_language_dict['{}'.format(mov_language.iloc[i][j])] = 0
    mov_language_dict['{}'.format(mov_language.iloc[i][j])] += 1
mov_language_dict = dict(sorted(mov_language_dict.items(),key = lambda x:x[1],reverse = True))
print(mov_language_dict)

x = list(mov_language_dict.keys())[:20]
y = list(mov_language_dict.values())[:20]
bar = plt.barh(x,y)
plt.yticks(rotation = 45, size = 7)
plt.xticks(rotation = 45, size = 7)
plt.bar_label(bar, y)
plt.show()

"""对于上映日期和平台发布日期，通过直方图展示其分布。"""

print(movie_data['release_date'].value_counts())
movie_data['release_date'].hist()
plt.title('release_date distribution')
plt.show()
print(movie_data['posted_date'].value_counts())
movie_data['posted_date'].hist()
plt.title('posted_date distribution')
plt.show()

"""（3）下一步，展示数值属性的5数概括及缺失值个数，并通过直方图展示数据分布及离群点。"""

numeric = ['views','downloads','IMDb-rating']
#利用describe函数给出5数概括
print(movie_data[numeric].describe().loc[['max', '75%', '50%', '25%', 'min']])


hist_plot(movie_data,numeric)

"""1.3 处理缺失值"""

#检查缺失值个数，展现缺失值比例
data_null = movie_data.isnull().sum(axis=0)
print(data_null.T)
rate = data_null/len(movie_data)
print(rate)

"""

（1）将缺失部分剔除:按行删去含有任意空值的数据
"""

drop_data = movie_data.dropna(axis=0)
print(f"未删除含有空值行前的数据数：{len(movie_data)}")
print(f"删除含有空值行后的数据数：{len(drop_data)}")
print(f"数据保留率：{len(drop_data)/len(movie_data)*100}%")

"""
可以根据删除前后的部分属性的柱状图来对比新旧数据差异：
"""
process_show(movie_data,drop_data,'appropriate_for')

"""（2）用最高频率值来填补缺失值"""

def fill_highest_fre(data, feature_list):
  dict = {}
  for col in feature_list:
    top = data[col].value_counts().index
    dict[col] = list(top)[0]
  new_data = data.fillna(dict)
  return new_data

fill_data = fill_highest_fre(movie_data,movie_data.columns)

"""对于标称属性，用最高频率值填充缺失值，并不会影响其分布，只是最高频率值的频率将增加，仍以appropriate_for属性为例："""

process_show(movie_data,fill_data,'appropriate_for')

"""展示经过删除缺失值和用最高频率填充缺失值这两种方法的数值属性的盒图对比："""
box_compare(drop_data,fill_data,numeric)

"""（3）通过属性的相关关系来填补缺失值

针对缺失率最高的appropriate_for属性，我们尝试使用计算其与其他属性的相关系数来填补缺失值。

首先将appropriate_for的值进行编码,用spearman方法计算相关系数，并画出热力图。
"""

code_data = pd.get_dummies(movie_data, columns=['appropriate_for'], dummy_na=True, drop_first=True)
plt.figure(figsize=(40, 40))
sns.heatmap(code_data.corr(method='spearman'), cmap='YlGnBu', annot=True)
plt.title('Correlation Analysis')

"""可以发现IMDb-rating是所有属性中最相关的属性，故利用IMDb-ranking来完成这个补全"""

corr_data = feature_corr_fill(movie_data,'appropriate_for','IMDb-rating')

"""展示填充前后柱状图对比："""

process_show(movie_data,corr_data,'appropriate_for')

"""（4）通过数据对象之间的相似性来填补缺失值"""



sample_corr_data = sample_corr_fill(movie_data.iloc[0:3000],'appropriate_for',numeric)

"""鉴于数据维度较大，运行时间过长，将只展示填充数据前3000行的填充效果："""

process_show(movie_data.iloc[0:3000],sample_corr_data,'appropriate_for')

"""2.数据集“GitHub Dataset”的数据分析

2.1 将数据读入为DataFrame，并展示数据中的数据类型
"""

github_data = pd.read_csv('/content/data/MyDrive/data/github_dataset.csv')
print(github_data.head(5))
print(github_data.dtypes)

"""

2.2 展示数据摘要

（1）首先，检查数据是否有重复，进行去重处理。
"""

print(github_data['repositories'].value_counts())

github_data = github_data.drop_duplicates()
print(github_data['repositories'].value_counts())

"""（2）接着，展示标称属性的值和频数统计,此处将给出每种可能值的频数统计，并且将展示频数的TOP20的柱状图。

以下将处理“repositories”属性，将对该属性的值进行分词处理，例如“octocat/Hello-World”，将以“/”为分词符提取作者名“octcoat”和项目名“Hello-World”，并将作者名与项目名存入github_data数据中,分别为“author”和“project”，两者均为标称属性。
"""

authors = []
projects = []
for item in github_data['repositories']:
    au , pro = item.split('/')
    authors.append(au)
    projects.append(pro)
github_data['author'] = authors
github_data['project'] = projects
print(github_data.head(5))

nominal = ['author','project','language']
bar_plot(github_data,nominal)

"""（3）下一步，展示数值属性的5数概括及缺失值个数，并通过直方图和盒图展示数据分布及离群点。"""

numeric = ['stars_count','forks_count','issues_count','pull_requests','contributors']
print(github_data[numeric].describe().loc[['max', '75%', '50%', '25%', 'min']])


hist_plot(github_data,numeric)
box_plot(github_data,numeric)

"""2.3 处理缺失值"""

#检查缺失值个数，展现缺失值比例
data_null = github_data.isnull().sum(axis=0)
print(data_null.T)
rate = data_null/len(github_data)
print(rate)

"""由以上结果可知，仅“language”属性含缺失值

（1）将缺失部分剔除:按行删去含有任意空值的数据
"""

drop_data = github_data.dropna(axis=0)
print(f"未删除含有空值行前的数据数：{len(github_data)}")
print(f"删除含有空值行后的数据数：{len(drop_data)}")
print(f"数据保留率：{len(drop_data)/len(github_data)*100}%")

"""
（2）用最高频率值来填补缺失值
"""

fill_data = fill_highest_fre(github_data,['language'])
process_show(github_data, fill_data, 'language')

"""（3）通过属性的相关关系来填补缺失值

尝试选择利用“forks_count”属性来填充“language”属性的缺失值
"""
code_data = pd.get_dummies(github_data, columns=['language'], dummy_na=True, drop_first=True)
plt.figure(figsize=(40, 40))
sns.heatmap(code_data.corr(method='spearman'), cmap='YlGnBu', annot=True)
plt.title('Correlation Analysis')

feature_corr_data = feature_corr_fill(github_data,'language','forks_count')
process_show(github_data,feature_corr_data,'language')

"""（4）通过数据对象之间的相似性来填补缺失值"""

sample_corr_data = sample_corr_fill(github_data,'language',numeric)

process_show(github_data,sample_corr_data,'language')