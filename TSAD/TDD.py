# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:18:42 2019

@author: zhongyh
"""
#from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def find_min(input_label): #寻找数量最小的标签
    if len(np.argwhere(input_label==0)) >len(np.argwhere(input_label==-1)):
        return -1
    else:
        return 0
    
def sort_ano(ano,n_clu):   #
    y = 0
    buf = ano
    ano = ano*n_clu
    while y<len(buf):
        o = 1
        while o<=n_clu:
            a_buf = ano[y]+o
            ano = np.append(ano,a_buf)
            o += 1
        y += 1
    ano = np.sort(ano)
    return ano

#def set_tho(value,enoma,th): #设定固定阈值，可不设
#    u = value[enoma-1]
#    u[u<th] = None
#    return u

def dynamic_th(data_in,nn,endn): #动态阈值1，高振福滤波，th1=max(max-avg,min-avg)
    jj = 0
    data_result = []
    while jj<endn-nn:
        data_input = data_in[jj:jj+nn]
        data_input = np.array(data_input).reshape(-1,1)
        data_buf =  max(data_input.max()-data_input.mean(),data_input.mean()-data_input.min())
        data_result = np.append(data_result,data_buf)
        jj += 1
    return data_result

def chain(data_input2): #动态阈值2，同比振幅：th2 此时刻的数值与过去n天相比较，取其中最大值
    longdata = len(data_input2)
    ii = 0   
    data_result = data_input2[:1440]

    while ii<longdata-2:
        data_max = np.max([data_input2[ii:ii+1],data_input2[ii+720:ii+720+1],data_input2[ii+1440:ii+1440+1]])
                            #data_input2[ii+2160:ii+2160+1],data_input2[ii+2880:ii+2880+1]])
        data_result = np.append(data_result,data_max)
        ii += 1
    return data_result.reshape(-1,1)

def all_war(data_a,ano,dy_t,cha_t,began_num): #源数据，异常点索引，阈值1值，阈值2值
    ano = ano[ano>began_num]    #挑出异常点
#    data_a = data_a.reshape(-1,1)
#    con1 = data_a[ano]-dy_t[ano-began_num]  #异常点的值-与异常点的值相同位置的 阈值1
#    con2 = data_a[ano]-cha_t[ano-began_num]  #异常点的值-与异常点的值相同位置的 阈值2
    pp = 0 
    ano_result = []
    while pp<len(ano)-1:
        
        try:
           con1 = data_a[ano[pp]]-dy_t[ano[pp]-began_num]  #异常点的值-与异常点的值相同位置的 阈值1  ano[ii]-began_num假如是负数呢
           con2 = data_a[ano[pp]]-cha_t[ano[pp]-began_num]  #异常点的值-与异常点的值相同位置的 阈值2
        except:
            break
        if con1>0 and con2>0:
            ano_result.append(ano[pp])
        pp += 1
    ano_result = np.array(ano_result).astype(int)
    return ano_result,data_a[ano_result]   
        
def run_TDD(data):
#    data = pd.read_csv('liudao.csv')
#data = pd.read_csv('198.18.1.18-198.18.1.17.csv') #异常
#data = pd.read_csv('198.18.0.102-198.18.0.101.csv')
    data_th = 10 #设置的固定阈值
    n = 60
    epsn = 10
    samp = 5
    dyth_nn = 2160
    chain_end = 2160
#    data_all = data['over_drop'].fillna(0).values.reshape(-1,1) #源数据
    data_all = data.fillna(0).values.reshape(-1,1)
    data_add = data_all
    if len(data_all)%n!=0: 
        data_add=np.append(data_all, [0]*(n-len(data_all)%n)) #加0后的数据
        
    #    while len(data_all)%n!=0:
    #        data_all = np.append(data_all,0)
    
    data_packets = data_add.reshape(-1,n)    #聚类预处理
    data_none = np.full(len(data_all),None)  #生成一个长度相同的none数组
    data_all2 = data_all.reshape(-1,)        #用来产生最终异常数据 list
    
    
    odeestimator = DBSCAN(eps=epsn,min_samples=samp,metric=euclidean,algorithm='auto')
    odeestimator.fit(data_packets)  #使用聚类数据
    label_pred_ode = odeestimator.labels_      
    label_1 = find_min(label_pred_ode) 
    anomaly = sort_ano(np.argwhere(label_pred_ode==label_1),n)
    
    dy_th = dynamic_th(data_all,dyth_nn,len(data_all)).reshape(-1,1) #动态阈值使用源数据
    cha_th = chain(data_all)
    war_dy,datawar = all_war(data_all,anomaly,dy_th,cha_th,dyth_nn)
    
    data_none[war_dy] = data_all2[war_dy]
    aa = np.where(data_all2==100)
    data_none[aa] = data_all2[aa]
    #data_none[data_none<10]=None
    data_none = list(data_none)
    data_none = data_none[:len(data_all)]
    j = 0
    while j<len(data_none):
        if data_none[j]!=None:
            if data_none[j]<data_th:
                data_none[j]=None
        j += 1
#    plt.title('out_packets')
#    plt.xlabel('Time series')
#    plt.ylabel('out_packets')
#    plt.scatter(np.arange(len(data_packets)*n),data_packets,s=1,c='b',label='Data')
#    #plt.scatter(anomaly,data_packets,s=5,c='red',marker="*",label='Waring')
#    #plt.scatter(war_dy,datawar,s=5,c='red',marker="*",label='Waring')
#    plt.scatter(np.arange(len(data_none)),data_none,s=1,c='red',marker="*",label='Waring')
#    #plt.plot(np.arange(len(dy_th))+dyth_nn,dy_th,c='r',label='Threshold')
#    #plt.plot(np.arange(len(cha_th))+chain_end,cha_th,c='k',label='Threshold2')
#    plt.vlines(dyth_nn,0,data_packets.max(),'r','dashed')
#    plt.show()
    
    return data_none

    
#if __name__ == '__main__': 
#    data = pd.read_csv('198.18.0.122-198.18.0.121.csv')
#    data2 = data['over_drop'].fillna(0).values.reshape(-1,1) #源数据
#    a = run_TDD(data)
#    print(a)