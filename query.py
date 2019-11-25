
import requests
import json
import csv
import datetime
import time
from collections import defaultdict
import sys
import os
import glob
import logging
import pandas as pd


json_query = '''
{
  "query": {
    "bool": {
      "filter": [
        {
          "terms": {
            "metricset.name": [
              "POP_tunnel_monitor"
            ]
          }
        },
        {
          "terms": {
            "metricset.module": [
              "business"
            ]
          }
        },
        {
          "range": {
            "timestamp": {
              "gte": 1563349200000,
               "lt": 1563349260000
            }
          }
        },
        {
          "nested": {
            "path": "pairs",
               "query": {
               "bool": {
                  "filter": [
                      {"term": {"pairs.name": "ns_id"}},
                      {"term": {"pairs.value.string": "$nsid"}}
                  ]
               }
            }
          }
        },
        {
          "nested": {
            "path": "pairs",
               "query": {
               "bool": {
                  "filter": [
                      {"term": {"pairs.name": "tun_oip"}},
                      {"term": {"pairs.value.string": "$tunoip"}}
                  ]
               }
            }
          }
        }
      ]
    }
  },
  "size": 10000
}
'''

url = 'http://rpc.dsp.chinanetcenter.com:10200/api/console/proxy?path=*metricelf*%2F_search&method=POST'
headers = { 
    'Content-Type': 'application/json; charset=utf-8',
    'Cookie': '__lcl=zh_CN; cluster=dashboard',
    'DNT': '1',
    'Host': 'rpc.dsp.chinanetcenter.com:10200',
    'kbn-version': '6.1.1',
    'Origin': 'http://rpc.dsp.chinanetcenter.com:10200',
    'Referer': 'http://rpc.dsp.chinanetcenter.com:10200/app/kibana',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36',
}

keys = [
    "timestamp",

    "pop_proname",  ##项目名称
    "pop_proid",  ##oms上项目ID
    "ns_id",  ##项目的enterpriseID

    "tun_uip",  ##underlay链路ip
    "tun_oip",  ##overlay链路ip
    "pop_ip",    ##POP服务器管理ip
    "tun_cost",   ##链路权重
    "tun_status",  ##链路状态，0表示正常，1表示异常
    "tun_cost_status",  ##链路是否主备线路，0表示备线路，1表示主线路

    "over_drop",  ##overlay非云网链路丢包率
    "over_dropth",   ##overlay非云网链路丢包率阈值
    "over_sl_drop",  ##overlay云网链路丢包率
    "over_sl_dropth",   ##overlay云网链路丢包率阈值
    "over_delay",  ##overlay链路时延
    "over_shakedelay", ##overlay链路抖动
    "over_mindelay",
    "over_maxdelay",
    "over_mdev",

    "under_drop",  ##underlay非云网链路丢包率
    "under_dropth",  ##underlay非云网链路丢包率阈值
    "under_sl_drop",  ##underlay云网链路丢包率
    "under_sl_dropth",  ##underlay云网链路丢包率阈值
    "under_delay",  ##underlay链路时延
    "under_shakedelay",   ##underlay链路抖动
    "under_mindelay",
    "under_maxdelay",
    "under_mdev",

    "tags",
    #"metricset_name"
]

key2index = {k : i for i, k in enumerate(keys)}

def request_data(sess, postdata):
    postdata = json.dumps(postdata)
    response = sess.post(url, data=postdata)
    res = response.text
    data = json.loads(res)
    samples = []
    for i, a in enumerate(data['hits']['hits']):
            pairs = a['_source']['pairs']
            pair_dict = defaultdict(str)
            pair_dict['timestamp'] = datetime.datetime.fromtimestamp(float(a['_source']['timestamp'])/1000)
            if len(a['_source']['tags']) > 0:
                pair_dict['tags'] = a['_source']['tags'][0]['value']
            if len(a['_source']['tags']) > 1:
                print('Warning:', a['_source']['tags'])
            pair_dict['metricset_name'] = a['_source']['metricset']['name']
            for p in pairs: 
                pair_dict[p['name']] = list(p['value'].values())[0]
            samples.append([pair_dict[key] for key in keys])
    return sorted(samples, key=lambda x:x[0]), data['hits']['total']

def get_list():
    postdata = json.loads(json_query)
    del postdata['query']['bool']['filter'][3:]
    timestamp_range = postdata['query']['bool']['filter'][2]['range']['timestamp']
    ta = datetime.datetime.now() - datetime.timedelta(hours=1)
    tb = ta + datetime.timedelta(minutes=2)
    timestamp_range['gte'] = ta.timestamp()*1000
    timestamp_range['lt'] = tb.timestamp()*1000
    logging.info('Request data in [{}, {})'.format(ta, tb))
    sess = requests.Session()
    sess.headers.update(headers)
    for _ in range(3):
        try:
            samples, tot = request_data(sess, postdata)
            logging.info('Hists: {} / {}'.format(len(samples), tot))
            break
        except Exception as e:
            logging.warning(e)
            logging.info('Retry')
            time.sleep(3)
    df = pd.DataFrame(samples, columns=keys)
    tunoip = {}
    proname = {}
    for nsid in df['ns_id'].unique():
        if nsid=='': continue
        t = df[df.ns_id==nsid]
        tunoip[nsid] = sorted(list(t.tun_oip.unique()))
        proname[nsid] = t.pop_proname.iloc[-1]
    return tunoip, proname


def query(nsid, tunoip, begin_time=None, end_time=None, data_dir='./data'):
    """ Query the ping data of specifical ns_id and tun_oip between specifical time range.

    Time is datetime type or a string with format '%Y-%m-%d %H:%M:%S', e.g. '2019-10-13 10:09:00'

    nsid: ns_id value.

    tunoip: tun_oip value.

    begin_time: The begining time to query. Default None means the time before 30 days.

    end_time: The ending time to query. Default None means the current time.
    """
    if type(begin_time) is str: begin_time = datetime.datetime.fromisoformat(begin_time)
    if type(end_time) is str: end_time = datetime.datetime.fromisoformat(end_time)
    nowtime = datetime.datetime.now()
    if end_time is None: end_time = nowtime
    if begin_time is None: begin_time = nowtime - datetime.timedelta(days=30)
    q_begin_time, q_end_time = begin_time, end_time
    if not os.path.exists(data_dir): os.mkdir(data_dir)
    nsid_dir = os.path.join(data_dir, nsid)
    if not os.path.exists(nsid_dir): os.mkdir(nsid_dir)
    file_prefix = tunoip.replace('>', '')
    # timestamp_path = os.path.join(nsid_dir, '%s.ts'%file_prefix)
    csv_path = os.path.join(nsid_dir, '%s.csv'%file_prefix)

    logging.info('Query: nsid={}, tunoip={}, timerange=[{}, {})'.format(nsid, tunoip, begin_time, end_time))
    if os.path.exists(csv_path):
        logging.info('Read data from %s'%csv_path)
        df = pd.read_csv(csv_path)
        begin_time = datetime.datetime.fromisoformat(df.iloc[-1]['timestamp']) + datetime.timedelta(minutes=1)
        logging.info('Time range: [{}, {}]'.format(df.iloc[0]['timestamp'], df.iloc[-1]['timestamp']))
    else:
        df = pd.DataFrame(columns=keys)
        begin_time = nowtime - datetime.timedelta(days=30)
    # if nowtime - end_time < datetime.timedelta(hours=1):
    #     tmp_end_time = end_time - datetime.timedelta(hours=1)
    # else:
    #     tmp_end_time = end_time
    tmp_end_time = end_time

    if q_end_time < begin_time:
        df = df.loc[q_begin_time:q_end_time]
        logging.info('Query: {} items'.format(len(df)))
        logging.info('done.')
        return 
    s = json_query.replace('$nsid', nsid).replace('$tunoip', tunoip)
    postdata = json.loads(s)
    timestamp_range = postdata['query']['bool']['filter'][2]['range']['timestamp']
    t_begin_time, t_end_time = int(begin_time.timestamp()), int(end_time.timestamp())
    t_tmp_end_time = int(tmp_end_time.timestamp())
    postdata['size'] = 10000
    delta_time = 3600*24*3 # 7 days
    sess = requests.Session()
    sess.headers.update(headers)
    def get_data(t, et):
        if t >= et: return []
        timestamp_range['gte'] = t*1000
        timestamp_range['lt'] = et*1000
        logging.info('Request data in [{}, {})'.format(datetime.datetime.fromtimestamp(t), datetime.datetime.fromtimestamp(et)))
        for _ in range(5):
            try:
                samples, tot = request_data(sess, postdata)
                logging.info('Hists: {} / {}'.format(len(samples), tot))
                return samples
            except Exception as e:
                logging.warning(e)
                logging.info('Retry')
                time.sleep(3)
    data = []
    for t in range(t_begin_time, t_tmp_end_time, delta_time):
        et = min(t+delta_time, t_tmp_end_time)
        data.extend(get_data(t, et))
    df = df.append(pd.DataFrame(data, columns=keys), ignore_index=True)
    # samples = get_data(t_tmp_end_time, t_end_time)
    logging.info('Total Hists: {}'.format(len(data)))
    if len(data)>0:
        logging.info('Save data to %s'%csv_path)
        df.to_csv(csv_path, index=False)
    # df = df.append(pd.DataFrame(samples, columns=keys), ignore_index=True)
    df.index = pd.to_datetime(df['timestamp'])
    df = df.loc[q_begin_time:q_end_time]
    logging.info('Query: {} items'.format(len(df)))
    logging.info('done.')
    df.fillna(0, inplace=True)
    return df


def cache():
    tunoip, proname = get_list()
    nsids = ['500850', '504608', '502902']
    for nsid in nsids:
        for lk in tunoip[nsid]:
            query(nsid, lk, data_dir='/home/monitor/data')

# data_dir = '/home/monitor/data'
# fn = os.path.join(data_dir, 'collect.log')
fn = '/dev/stdout'
logging.basicConfig(level=logging.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s', 
                datefmt='%Y-%m-%d %H:%M:%S')


if __name__ == "__main__":
    cache()
    nsid = '1'
    tunoip = '10.0.1.214->10.0.1.213'
    # ta = '2019-10-30 13:00:00'
    # tb = '2019-10-30 14:05:00'
    # ta = None
    # df = query(nsid, tunoip, data_dir='/home/monitor/data')
    # print(proname)
    # print(data['over_drop'])
    # print(data['over_drop'].astype(float))
    # a = df['tun_uip'].unique()
    # b = df['tun_oip'].unique()
    # print(b,a)