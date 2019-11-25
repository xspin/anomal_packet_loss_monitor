import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

def ss_detect(series, timespan, thr_pred, threshold):
    score = np.zeros(len(series))
    pred = np.zeros(len(series), dtype=np.int)
    v = np.array(series)
    for k in range(timespan, len(v)):
        for i in range(k-timespan, k):
            score[k] += 1 if v[k]-v[i]>threshold else 0
        score[k] /= timespan
        if score[k] > thr_pred:
            pred[k] = 1
    return pred, score

def entropy_detect(series, timespan, threshold):
    score = np.zeros(len(series))
    pred = np.zeros(len(series), dtype=np.int)
    v = np.array(series)
    for k in range(timespan-1,  len(v)):
        count = defaultdict(int)
        for i in range(k-timespan+1, k+1):
            count[v[i]] += 1
        for c in count.values():
            score[k]  += - np.log(c/timespan) * c/timespan / np.log(100)
        if score[k] > threshold:
            pred[k] = 1
    return pred, score
        
class Entropy_clf:
    def __init__(self, threshold):
        self.threshold = threshold
    def entropy(self, sample):
        count = defaultdict(int)
        for x in sample:
            count[x] += 1
        return sum([-np.log(c/len(sample))*c/len(sample) for c in count.values()])

    def predict(self, samples):
        pred, score = [],  []
        for sample in samples:
            e = self.entropy(sample) / np.log(100)
            p = 0 if e <= self.threshold else 1
            pred.append(p)
            score.append(e)
        return np.asarray(pred), np.asarray(score)
            
class EmpiricalRule_clf:
    def __init__(self, threshold=3, alpha=0.6):
        self.threshold = threshold
        self.alpha = 0.9
    def detect(self, sample, avg, std):
        cnt = 0
        tot = 0
        for i, x in enumerate(sample):
            avg_ = sample[:i].mean()*self.alpha + (1-self.alpha)*avg
            std_ = sample[:i].std()*self.alpha + (1-self.alpha)*std
            if x > avg_ + self.threshold*std_:
                cnt += 1
        return sample.mean(), sample.std(), cnt
            
    def predict(self, samples):
        pred, score = [], []
        avg, std = 0, 0
        for sample in samples:
            avg, std, cnt = self.detect(sample, avg, std)
            p = 0 if cnt <= 2 else 1
            pred.append(p)
            score.append(cnt/len(sample))
        return np.asarray(pred), np.asarray(score)


        
import scipy.stats as ss

class Distance_to_history:
    def __init__(self, window_size, period=720, num_history=None, dist_func=ss.wasserstein_distance):
        self.period = period
        self.window_size = window_size
        self.num_history = num_history
        self.dist_func = dist_func
    def detect(self, sample):
        dist = [0]*self.window_size
        dist0 = [0]*self.window_size
        for i in range(self.window_size, len(sample)):
            u = sample[i:i+self.window_size]
            d, cnt = 0, 0
            for j in range(i-self.period, 0, -self.period):
                v = sample[j:j+self.window_size]
                d += self.dist_func(u, v)
                cnt += 1
                if self.num_history and cnt>=self.num_history: 
                    break
            cnt = max(cnt, 1)
            dist.append(d/cnt)
            dist0.append(self.dist_func(u, [0]))
        return np.asarray(dist), np.asarray(dist0)

class Compare_to_history:
    def __init__(self, window_size, period=720, num_history=None, alpha=1, weight_func=sum):
        self.period = period
        self.window_size = window_size
        self.num_history = num_history
        self.weight_func = weight_func
        self.alpha = alpha
    def detect(self, sample):
        score = [self.weight_func(sample[max(0,i-self.window_size):i+1]) for i in range(len(sample))]
        threshold = score[0:self.window_size]
        for i in range(self.window_size, len(score)):
            cnt = 0
            tmp = []
            for j in range(i-self.period, 0, -self.period):
                tmp.append(score[j])
                if self.num_history and cnt>=self.num_history: 
                    break
            threshold.append(np.mean(tmp)+self.alpha*np.std(tmp))
        return np.asarray(score), np.asarray(threshold)

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
        # self.window_size = window_size
        self.period = period
        self.p = p
        self.z = z
        # self.smooth_func = smooth_func
        self.min_thr = min_thr if min_thr else 0.02
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
        return score, threshold, zs, ss 

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
        plt.plot(np.sqrt(self.mu2_pre-self.mu_pre**2), label='EW std')
        plt.legend(loc='upper right')
        if xlim: plt.xlim(xlim)
        plotnum -= 1
        if plotnum==0: 
            return

def to_vg(series, h=False, scope=None, graph=False):
    n = len(series)
    if not scope:
        scope = n
    if graph:
        g = [[set(), set()] for _ in range(n)]
    degree = np.zeros([n,2], dtype=int) 
    for j in range(n-1,-1,-1):
        v = -np.inf if h else np.inf
        for i in range(j-1,max(-1,j-scope),-1):
            r = min(series[i], series[j]) if h else (series[j]-series[i])/(j-i)
            flag = v<r if h else v>r
            v = max(v, series[i]) if h else min(v, r)
            if flag:
                if series[i]<series[j]: # low -> high
                    if graph:
                        g[i][0].add(j)
                        g[j][1].add(i)
                    degree[i][0] += 1
                    degree[j][1] += 1
                else:
                    if graph:
                        g[i][1].add(j)
                        g[j][0].add(i)
                    degree[i][1] += 1
                    degree[j][0] += 1
    return (degree, g) if graph else degree

def to_hvg(series, scope=None):
    return to_vg(series, h=True, scope=scope)

if __name__ == "__main__":
    a = np.random.randint(0,5,10)