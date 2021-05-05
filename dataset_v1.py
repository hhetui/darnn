
# encoding: utf-8
import os
import sys
import math
import json
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool


PWD = os.path.dirname(os.path.realpath(__file__))



def load_stock(s):
    df = pd.read_csv(os.path.join(PWD, 'HScode', s), index_col=1)#第一列作为index
    df.set_index(df.index.astype('str'), inplace=True)#index转为str类型
    #df.drop(['_id', 'code', 'updatetime'], axis=1, inplace=True)
    df.drop(['ts_code', 'pre_close', 'change','pct_chg'], axis=1, inplace=True)
    return df


def z_score(df):
    return (df - df.mean()) / df.std()


def stock_sample(input_):
    #d是X的最后一天，预测d后的第target天
    s, T, d = input_
    df = global_df[s]
    if d not in df.index:
        return
    iloc = list(df.index).index(d) + 1  # df.iloc是前开后闭的，所有要+1才能取到这个点
    if iloc < T:  # 数据量不够
        return
    x = df.iloc[iloc-T:iloc][x_column].copy()
    xz = np.array(z_score(x))
    if np.isnan(xz).any().any():
        return
    y = df.iloc[iloc-T:iloc][y_column].copy()
    yz = np.array(z_score(y))
    if np.isnan(yz).any():
        return
    t = 1 if df.iloc[iloc+target-1,:]['close'] > df.loc[d, 'close'] else 0
    return xz, yz, t, s, d


def sample_by_dates(T, dates):
    #T是时间窗 dates为 各个X结尾的时间点列表
    pool = Pool(22)
    datas = pool.map(stock_sample, [(f, T, d) for d in dates for f in files])
    pool.close()
    pool.join()
    datas = filter(lambda data: data is not None, datas)
    xs, ys, ts, stocks, days  = zip(*datas)
    return {'x': np.array(xs), 'y': np.array(ys), 't': np.array(ts),
        'stock': np.array(stocks), 'day': np.array(days)}


def generate_data_year(year):
    df = global_df['600000.SH.csv']
    ti = df['%s0101' % year:'%s1231' % year].index   # test
    data = sample_by_dates(T, ti)
    with open(os.path.join(DatasetPath, 'v1_T%s_yb%s_%s.pickle' % (T, target, year)), 'wb') as fp:
        pickle.dump(data, fp)


def generate_data_season(T):
    df = load_stock('600000.SH.csv')
    ti = df['20190101':'20190331'].index
    dataset = sample_by_dates(T, ti)
    with open(os.path.join(DatasetPath, 'v1_T%s_yb%s_123.pickle' % (T, target)), 'wb') as fp:
        pickle.dump(dataset, fp)
    ti = df['20190401':'20190631'].index
    dataset = sample_by_dates(T, ti)
    with open(os.path.join(DatasetPath, 'v1_T%s_yb%s_456.pickle' % (T, target)), 'wb') as fp:
        pickle.dump(dataset, fp)
    ti = df['20190701':'20190931'].index
    dataset = sample_by_dates(T, ti)
    with open(os.path.join(DatasetPath, 'v1_T%s_yb%s_789.pickle' % (T, target)), 'wb') as fp:
        pickle.dump(dataset, fp)
    ti = df['20191001':'20191231'].index
    dataset = sample_by_dates(T, ti)
    with open(os.path.join(DatasetPath, 'v1_T%s_yb%s_1012.pickle' % (T, target)), 'wb') as fp:
        pickle.dump(dataset, fp)


if __name__ == '__main__':
    global_df = {f: load_stock(f) for f in os.listdir('HScode')}
    x_column = ['open', 'high', 'low', 'amount', 'vol', 'close']
    y_column = 'close'
    T =10
    target =1
    files = os.listdir('HScode')
    DatasetPath = './v_10day'
    if not os.path.exists(DatasetPath):
        os.mkdir(DatasetPath)
    generate_data_season(T)
    for y in range(2018, 2009, -1):
        generate_data_year(y)
