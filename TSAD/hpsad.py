import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
class Detector:
    '''
    A Class for implementing the Historical Peoridical Statistic based Anomaly Detection Method.

    Args:
        period: The period of the time series.
        p: The parameter of the exponential weighted moving average on the statistics. 
            Defaul is 0.5.
        z: The parameter in the threshold.
            Defaul is None, which means to use the dynamic value automatically.
        min_thr: The minimum gap to the anomaly threshold.
        max_thr: The maximum value of the data in the series to update the statistics.
            If the value exceeds it, the statistics will not be update in this time step.
    '''
    def __init__(self, period=720, p=0.5, z=None, min_thr=None, max_thr=None):
        self.period = period
        self.p = p
        self.z = z
        self.min_thr = min_thr if min_thr else 0.1
        self.max_thr = max_thr if max_thr else 0.6

    def smooth_wma(self, window_size, weights=None):
        if weights is None:
            weights = np.ones(window_size, dtype=float)/window_size
        elif weights in ['auto', 'default']:
            weights = (window_size-np.arange(window_size-1,-1,-1, dtype=float))*2/window_size/(window_size+1)
        def func(series):
            ss = np.empty_like(series, dtype=float)
            for i in range(len(ss)):
                if i < window_size:
                    ss[i] = np.dot(series[:i+1], weights[-i-1:]/np.sum(weights[-i-1:]))
                else:
                    ss[i] = np.dot(series[i-window_size+1:i+1], weights)
            return ss
        return func
    def smooth_ema(self, alpha=None, span=None):
        assert alpha or span, 'Args `alpha` and `span` one of which must be set.'
        if span:
            alpha = 2/(span+1)
        def func(series):
            ss = np.empty_like(series, dtype=float)
            for i in range(len(ss)):
                if i==0:
                    ss[i] = series[i]
                else:
                    ss[i] = alpha*series[i] + (1-alpha)*ss[i-1]
            return ss
        return func

    def _smooth(self, series, smooth_func):
        if smooth_func:
            series= smooth_func(series)
        return series

    def detect(self, series, smooth_func=None):
        '''
        To detect the anomaly points in a univariate time series.

        Args:
            series: The data of a time series.

            smooth_func: A smoothing function on the series.
                e.g. smooth_wma(), smooth_ema()
        '''
        def find_max_z(err_cnt, dz):
            min_z = 3
            m = [min_z, 0]
            for i in range(1, len(err_cnt)):
                if i*dz >= min_z:
                    d = err_cnt[i-1] - err_cnt[i]
                    if d>0 and d >= m[1]:
                        m = [i*dz, d]
            return m[0]
        self.mu_pre = np.zeros_like(series) 
        self.mu2_pre = np.zeros_like(series) 
        # self.mu_pre[0], self.mu2_pre[0] = 0, 0
        thr_pre = 0
        self.mu = np.zeros(self.period)
        self.mu2 = np.zeros(self.period)
        self.mu_pre[0] = series[0]
        self.mu2_pre[0] = series[0]**2
        self.mu_tmp = np.empty_like(series)
        self.std_tmp = np.empty_like(series)
        ss = self._smooth(series, smooth_func)
        score = np.empty_like(ss, dtype=float)
        threshold = np.zeros_like(score)
        dz, maxz = 0.5, 10
        err_cnt = np.zeros(int(maxz/dz)+1)
        zs = np.empty_like(ss)
        for t, x in enumerate(ss):
            k = t % self.period
            if t<self.period:
                score[t] = 0
                zs[t] = 0
            else:
                sigma = np.sqrt(self.mu2[k] - self.mu[k]**2)
                for i in range(len(err_cnt)-1, -1, -1):
                    tz = i*dz
                    eps = self.mu[k] + tz*sigma
                    if x>eps:
                        err_cnt[0:i+1] += 1
                        break
                zs[t] = find_max_z(err_cnt, dz)
                z = self.z if self.z else zs[t]
                eps = self.mu[k] + z*sigma
                # eps = max(eps, thr_pre)
                eps = min(eps, self.max_thr)
                score[t] = x - eps - self.min_thr #- (thr_pre-self.mu[k]-sigma)
                threshold[t] = eps
            self.mu_tmp[t] = self.mu[k]
            self.std_tmp[t] = np.sqrt(self.mu2[k]-self.mu[k]**2)
            if series[t] < self.max_thr:
                self.mu[k] = (1-self.p)*self.mu[k] + self.p*x
                self.mu2[k] = (1-self.p)*self.mu2[k] + self.p*x*x
                if t>0:
                    self.mu_pre[t] = (1-self.p)*self.mu_pre[t-1] + self.p*x
                    self.mu2_pre[t] = (1-self.p)*self.mu2_pre[t-1] + self.p*x*x
                    # if self.mu_pre[t]>10:
                    #     print(self.mu_pre[t])
                    tmp = self.mu2_pre[t]-self.mu_pre[t]**2
                    if tmp<0: 
                        # self.mu2_pre[t] -= tmp-0.1
                        tmp = 0
                    thr_pre = self.mu_pre[t]+np.sqrt(tmp)
            # self.mu[k] = (t*self.mu[k] + x)/(t+1)
            # self.mu2[k] = (t*self.mu2[k] + x*x)/(t+1)
        self.series = series
        self.ss = ss
        self.score = score
        self.threshold = threshold
        self.zs = zs

        anom = [s>0 for s in score]
        return anom, list(score)

    def plot(self, xlim=None, title=None, plotnum=None, ff=None):
        if xlim=='tight':
            xlim = (0, len(self.series))
        if plotnum is None: 
            plotnum = 6
        n_plots = plotnum
        plt.subplot(n_plots,1,1)
        if title: plt.title(title, fontfamily=ff)
        x,y, series_a, = [], [], []
        for i,s in enumerate(self.score):
            if s>0:
                x.append(i)
                y.append(s)
                series_a.append(self.series[i])
        ms = 5 if len(x)<100 else 2 if len(x)<500 else 1
        plt.plot(self.series, label='series')
        plt.plot(x, series_a, 'r.', ms=ms, label='anomaly (%s)'%len(x))
        # plt.plot([self.period, self.period], [0, max(self.series)], '--', color='gray')
        for ti in range(0, len(self.series), self.period):
            plt.plot([ti, ti], [max(self.series)/1.5, max(self.series)], '--', color='gray')
        plt.legend(loc='upper right')
        # plt.ylim(0,0.3)
        if xlim: plt.xlim(xlim)
        plotnum -= 1
        if plotnum==0: 
            return
        plt.xticks([])

        plt.subplot(n_plots,1,2)
        # plt.plot(self.score, label='score')
        plt.plot(self.ss, label='smoothed')
        plt.plot(self.threshold, label='threshold')
        plt.legend(loc='upper right')
        if xlim: plt.xlim(xlim)
        plotnum -= 1
        if plotnum==0: 
            return
        plt.xticks([])

        plt.subplot(n_plots,1,3)
        plt.plot([0,len(self.score)], [0,0], '--', color='gray')
        plt.plot(self.score, label='score')
        plt.plot(x,y, 'r.', ms=ms, label='anomaly (%s)'%(sum(self.score>0)))
        plt.ylim(bottom=max(-0.5, min(self.score)))
        plt.legend(loc='upper right')
        if xlim: plt.xlim(xlim)
        plotnum -= 1
        if plotnum==0: 
            return
        plt.xticks([])

        plt.subplot(n_plots,1,4)
        plt.plot(self.zs, label='z')
        plt.legend(loc='upper right')
        if xlim: plt.xlim(xlim)
        plotnum -= 1
        if plotnum==0: 
            return
        plt.xticks([])

        plt.subplot(n_plots,1,5)
        plt.plot(self.mu_tmp, label='period avg')
        # plt.plot(self.mu_tmp+self.std_tmp, 'r--', label='mean+std')
        plt.plot(self.std_tmp, label='period std')
        plt.legend(loc='upper right')
        if xlim: plt.xlim(xlim)
        plotnum -= 1
        if plotnum==0: 
            return
        plt.xticks([])

        plt.subplot(n_plots,1,6)
        plt.plot(self.mu_pre, label='EW avg')
        plt.plot(np.sqrt(max(0,self.mu2_pre-self.mu_pre**2)), label='EW std')
        plt.legend(loc='upper right')
        if xlim: plt.xlim(xlim)
        plotnum -= 1
        if plotnum==0: 
            return