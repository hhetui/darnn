import os
import pickle
import numpy as np
data_conf = {'train_list':[2010,2011,2012,2013,2014,2015,2016,2017,2018],'test_list':[123,456,789,1012],'T':20,'datapath':'../v_20day'}
def load_dataset(data_conf):
    train_years = data_conf['train_list']
    test_years = data_conf['test_list']
    def load_pickle(years,resfile):
        data_dic = None
        for y in years:
            with open(os.path.join(data_conf['datapath'],
                'v1_T'+str(data_conf['T'])+'_yb1_%s.pickle' % (y)), 'rb') as fp:
                dataset = pickle.load(fp)

            if data_dic is None:
                data_dic = {}
                data_dic['x'] = dataset['x']
                data_dic['y'] = dataset['y']
                data_dic['t'] = dataset['t']

            data_dic['x'] = np.append(data_dic['x'], dataset['x'], axis=0)#20*6
            data_dic['y'] = np.append(data_dic['y'], dataset['y'], axis=0)#20*1
            data_dic['t'] = np.append(data_dic['t'], dataset['t'], axis=0)#1
        with open(os.path.join(data_conf['datapath'],resfile), 'a+') as f:
             f.write('6'+'\n')
        print(len(data_dic['t']))
        for i in range(len(data_dic['t'])):
            if i%1000 == 0:
                print(i)
            x = data_dic['x'][i].reshape(1,-1)[0]
            s = str(data_dic['t'][i])
            for j in range(len(x)):
                s += ' '+str(x[j])
            #print(len(s.split(' ')))
            with open(os.path.join(data_conf['datapath'],resfile), 'a+') as f:
                 f.write(s+'\n')
        return 0
    dataset = {}
    load_pickle(train_years,'train.txt')
    load_pickle(test_years,'test.txt')

    return 0
if __name__  == '__main__':
    load_dataset(data_conf)
