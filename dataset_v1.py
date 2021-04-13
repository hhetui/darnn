#!/usr/bin/env python
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
DatasetPath = '/mnt/data1/ryan/dataset'


def load_stock(s):
    df = pd.read_csv(os.path.join(PWD, 'day_data_csv', s), index_col=0)
    df.set_index(df.index.astype('str'), inplace=True)
    df.drop(['_id', 'code', 'updatetime'], axis=1, inplace=True)
    return df


def z_score(df):
    return (df - df.mean()) / df.std()


def stock_sample(input_):
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
    t = 1 if df.iloc[iloc+target_day-1,:]['close'] > df.loc[d, 'close'] else 0
    # ci
    with open(os.path.join(PWD, '../price_network/ci', 'close_%s' % T, '%s.json' % s[:-4])) as fp:
        j = json.load(fp)
    if d not in j:
        return
    dci = j[d]
    ci = pd.Series([dci['%.4f' % p] for p in y])
    ciz = np.array(z_score(ci))
    if np.isnan(ciz).any():
        ciz = np.array(ci)
    return xz, yz, t, ciz, s, d


def sample_by_dates(T, dates):
    pool = Pool(22)
    datas = pool.map(stock_sample, [(f, T, d) for d in dates for f in files])
    pool.close()
    pool.join()

    datas = filter(lambda data: data is not None, datas)
    xs, ys, ts, cis, stocks, days  = zip(*datas)
    return {'x': np.array(xs), 'y': np.array(ys), 't': np.array(ts),
            'ci': np.array(cis), 'stock': np.array(stocks), 'day': np.array(days)}


def generate_data_year(year):
    df = global_df['999999.XSHG.csv']
    ti = df['%s0101' % year:'%s1231' % year].index   # test
    data = sample_by_dates(T, ti)
    with open(os.path.join(DatasetPath, 'v1_T%s_yb%s_%s.pickle' % (T, target_day, year)), 'wb') as fp:
        pickle.dump(data, fp)


def generate_data_season(T):
    df = load_stock('999999.XSHG.csv')
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
    global_df = {f: load_stock(f) for f in os.listdir('day_data_csv')}
    x_column = ['open', 'high', 'low', 'amount', 'vol', 'close']
    y_column = 'close'
    target_day = 1
    T =20
    target =1
    files = os.listdir(os.path.join(PWD, 'day_data_csv'))
    files = set(files) - set(['999999.XSHG.csv'])
    generate_data_season(20)
    for y in range(2018, 2009, -1):
        generate_data_year(20, y)
    # stock_sample('601800.XSHG.csv', 20, '20120625')
