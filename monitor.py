from TSAD import hpsad
from TSAD import TDD

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import query
import datetime
import time
import os
import json
import sys
import logging

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
from urllib.parse import urlparse, parse_qs

class Monitor:
    def __init__(self, nsid, tunoip):
        self.nsid = nsid
        self.tunoip = tunoip

    def get_setting():
        setting = {}
        setting['hpsad'] = {
            'min_thr' : [10, '取值范围：0-100；最小阈值，小于其则判定为正常'],
            'max_thr' : [70, '取值范围：0-100；最大阈值，超过则判定为异常'],
            'alpha' : [0.5, '取值范围：0-1；数据指数平滑系数'],
            'p' : [0.5, '取值范围：0-1；计算统计量的指数平滑系数'], 
            'z' : ['auto', '取值范围：>0；标准差倍数，auto表示动态选取'],
        }
        setting['tdd'] = {
            'variable_1' : ['default_value', 'range: xxx; description ... '],
            'variable_2' : ['default_value', 'range: xxx; description ... '],
        }
        return setting
        
    def preprocess(self, column):
        def tofloat(x):
            try: 
                return float(x)
            except:
                return 0.0
        try:
            self.df[column] = self.df[column].astype(float)
        except Exception as e:
            logging.info('Catch data type exception in column %s'%column)
            self.df[column] = self.df[column].apply(tofloat)

    def update(self):
        self.df = query.query(self.nsid, self.tunoip, data_dir='/home/monitor/data')
        self.time = list(map(lambda x:x.strftime("%Y-%m-%d %H:%M:%S"), self.df.index))
        self.preprocess('over_drop')
        self.preprocess('under_drop')

    def _hpsad(self, column, setting, d=100.0):
        series = self.df[column]
        det = hpsad.Detector(period=720, p=setting['p'], z=setting['z'], min_thr=setting['min_thr']/d, max_thr=setting['max_thr']/d)
        smoothfun = det.smooth_ema(alpha=setting['alpha'])
        anom, score = det.detect(series.values/d, smoothfun)
        warn = [v if a else None for a,v in zip(anom, series.values)]
        return warn, score
        

    def hpsad_detect(self, setting):
        logging.info('Settings: %s'%str(setting))
        for k in setting:
            if k == 'z' and setting[k][0] in [None, 'null', 'auto']: 
                setting[k] = None
            else:
                setting[k] = float(setting[k][0])
        self.warn, self.score = self._hpsad('over_drop', setting, 100)
        self.warn2, self.score2 = self._hpsad('under_drop', setting, 100)

        # if figpath != None:
        #     plt.figure(figsize=(15,8))
        #     det.plot(title="{}@{}".format(self.nsid, self.tunoip), plotnum=3, xlim='tight')
        #     print('Save fig to %s'%figpath)
        #     plt.savefig(figpath, bbox_inches='tight')
        #     plt.close()

    def tdd_detect(self, setting):
        self.warn = TDD.run_TDD(self.df['over_drop'])
        self.warn2 = TDD.run_TDD(self.df['under_drop'])
        self.score = []
        self.score2 = []

    def get_data(self):
        self.score = []
        self.score2 = []
        # tmp = self.result
        # data = list(tmp['data'].values)
        # warn = list(tmp['warn'].values)
        # score = list(tmp['score'].values)
        self.data = list(self.df['over_drop'].values)
        self.data2 = list(self.df['under_drop'].values)
        assert len(self.time)==len(self.warn)
        result = {'Time':self.time, 'Data':self.data, 'Warn':self.warn, 'Score':self.score,
                  'Data2':self.data2, 'Warn2':self.warn2, 'Score2':self.score2,
                  'Delay':list(self.df['over_delay'].values),
                  'Underlay': list(self.df['tun_uip'].unique())
            }
        return result



class Handler(BaseHTTPRequestHandler):

    def _response(self, ext=''):
        tp = {'.js':'text/javascript', '.html':'text/html', '.css':'text/css', 
            '.csv':'text/plain', '.png':'image/png', '.jpg':'image/jpg', '.ico':'image/ico',
            'plain':'text/plain', '.json':'appplication/json'}
        content_type = tp[ext] if ext in tp else ext
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.end_headers()

    def _response404(self):
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        qdict = parse_qs(u.query)
        ext = os.path.splitext(path)[-1]
        local_path = os.path.join(WEBDIR, path[1:])
        if ext != '':
            if not os.path.exists(local_path):
                self._response404()
            else:
                self._response(ext)
                with open(local_path, 'rb') as f:
                    self.wfile.write(f.read())
        elif path == '/':
            self._response('.html')
            self.wfile.write(b'<meta http-equiv="Refresh" content="0; url=/monitor" />')
        elif path in '/monitor':
            # nsid = qdict['nsid'][0]
            # tunoip = qdict['tunoip'][0]
            self._response('text/html; charset=utf-8')
            with open(os.path.join(WEBDIR,'index.html')) as f:
                self.wfile.write(f.read().encode('utf-8'))
        elif path == '/search':
            self._response('text/html; charset=utf-8')
            nsip, nsname = query.get_list()
            nsid = qdict['nsid'][0]
            if nsid == 'all':
                html = "<html>%s</html>"%('<br>'.join(['%s %s'%(i,nsname[i]) for i in sorted(nsname.keys())]))
                self.wfile.write(html.encode('utf-8'))
            else:
                html = "<html>%s  %s<br><br>%s</html>"%(nsid, nsname[nsid], '<br>'.join(nsip[nsid]))
                self.wfile.write(html.encode('utf-8'))
        elif path == '/getsetting':
            self._response('plain; charset=utf-8')
            data = Monitor.get_setting()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        elif path =='/log':
            self._response('plain; charset=utf-8')
            with open('monitor.log', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode()
        # qdict = parse_qs(post_data)
        # print(post_data)
        u = urlparse(self.path)
        if u.path == '/select':
            qdict = json.loads(post_data)
            self._response('plain')
            nsid, tunip = qdict['nsid'], qdict['tunip']
            method = qdict['method']
            setting = qdict['setting']
            data = self.get_data(nsid, tunip, method, setting)
            self.wfile.write(data.encode('utf-8'))

        if u.path == '/getlist':
            qdict = parse_qs(post_data)
            nsip, nsname = query.get_list()
            data = {'nsip':nsip, 'nsname':nsname}
            self._response('plain')
            self.wfile.write(json.dumps(data).encode('utf-8'))

    def get_data(self, nsid, tunoip, method, setting):
        # nsid = '500052'
        # tunoip = '198.18.2.21->198.18.2.22'
        # nsid = '500850'
        # tunoip = '198.18.0.134->198.18.0.133'
        if nsid=='0' or tunoip=='0':
            data = {
                'Time': ["2019-10-07 10:%02d:00"%i for i in range(0,10)],
                'Data': [0,1,15,30,20,1,34,50,23,1],
                'Warn': [None,None,None,30,20,None,None,50,None,None],
                'Score': [0,0.01,0.15,0.30,0.20,0.01,0.34,0.50,0.23,0.01],
                'Data2': [0,10,10,3,10,9,30,10,30,1],
                'Warn2': [None,10,10,None,10,None,30,None,30,None],
                'Delay': [0,0.01,0.15,0.30,0.20,0.01,0.34,0.50,0.23,0.01],
            }
            data = json.dumps(data).replace('NaN', 'null').replace('nan','null').replace('None','null')
            return data

        m = Monitor(nsid, tunoip)
        m.update()
        logging.info('Start %s detecting ...'%method.upper())
        if method == 'hpsad':
            m.hpsad_detect(setting)
        elif method == 'tdd':
            m.tdd_detect(setting)
        else:
            logging.info('Wrong method: %s'%method)
            return 'Wrong method'
        logging.info('Detect done.')
        data = m.get_data()
        data = json.dumps(data).replace('NaN', 'null').replace('nan','null').replace('None','null')
        return data
        try:
            pass
        except Exception as e:
            logging.info('Exception: %s'%e)
            self._response404()
            self.wfile.write(str(e).encode('utf-8'))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """在一个新的线程中处理请求。"""

WEBDIR = 'web'

fn = '/dev/stdout'
logging.basicConfig(level=logging.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s', 
                datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    port = 8088 if len(sys.argv)<2 else int(sys.argv[1])
    host = '0.0.0.0'
    server = ThreadedHTTPServer((host, port), Handler)
    logging.info('Starting server litening on %s:%s'%(host,port))
    server.serve_forever()
    # exit()
    # nsid = '500850'
    # tunoip = '198.18.0.134->198.18.0.133'
    # m = Monitor(nsid, tunoip)
    # m.update()
    # # m.hpsad_detect()
    # m.tdd_detect()
    # d = m.get_data()
    # print(d)
    # return m.hpsad_detect()

