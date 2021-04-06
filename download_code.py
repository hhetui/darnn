import os
import tushare as ts
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--start_date', type=str, default = '20100101')
args = parser.parse_args()


def updateHS300(start_date='20100101',datapath='HScode'):
    if not os.path.exists(datapath):
        os.mkdir(datapath)
    ts.set_token('41ed2d8cb1b884009c361e9bc4bb3885a27ebe2679d04de271cf87bc')
    pro=ts.pro_api()
    HS300=ts.get_hs300s()
    num=0
    for name in HS300['code']:
        num+=1
        print(num,'--OK!')
        filename=name+'.SZ'
        df=pro.daily(ts_code=filename,start_date=start_date)
        if len(df)==0:
            filename=name+'.SH'
            df=pro.daily(ts_code=filename,start_date=start_date)
        df=df.iloc[::-1]
        df.to_csv(os.path.join(datapath,filename+'.csv'),index=None)

if __name__ == "__main__":
    updateHS300(args.start_date)
