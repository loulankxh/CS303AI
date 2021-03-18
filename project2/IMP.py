# -*- coding: utf-8 -*-
# written by mark zeng 2018-11-14
# modified by Yao Zhao 2019-10-30
# re-modified by Yiming Chen 2020-11-04

import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np


core = 8

class Edge(object):
    def __init__(self, u1, v1, w1):
        self.u = u1
        self.v = v1
        self.w = w1
        self.next_e = -1


global n, m, head_reverse, edge_reverse, seed_num, time_limit
head_reverse = {}
edge_reverse = []
active = []


def get_RR(model, node):
    if model == 'IC':
        return IC_RR(node)
    else:
        return LT_RR(node)


def IC_RR(node):
    head1 = head_reverse.copy()
    edge1 = edge_reverse[:]
    active_set1 = [node]
    seeds = active_set1[:]
    active1 = active[:]
    active1[node] = 1
    len1 = len(active_set1)
    ans = len1
    while len1 != 0:
        new_active = []
        for ele in active_set1:
            u = ele
            if u in head1:
                es = edge1[head1[u]]
                v = es.v
                w = es.w
                if active1[v] == 0:
                    ran = np.random.random()
                    if ran <= w:
                        active1[v] = 1
                        new_active.append(v)
                        seeds.append(v)
                while es.next_e != -1:
                    es = edge1[es.next_e]
                    v = es.v
                    w = es.w
                    if active1[v] == 0:
                        ran = np.random.random()
                        if ran <= w:
                            active1[v] = 1
                            new_active.append(v)
                            seeds.append(v)
        len1 = len(new_active)
        ans += len1
        active_set1 = new_active
    return ans, seeds


def LT_RR(node):
    head1 = head_reverse.copy()
    edge1 = edge_reverse[:]
    active_set1 = node
    seeds = [node]
    active1 = active[:]
    active1[node] = 1
    while active_set1 > 0:
        new_active = []
        u = active_set1
        if u in head1:
            es = edge1[head1[u]]
            v = es.v
            if active1[v] == 0:
                new_active.append(v)
            while es.next_e != -1:
                es = edge1[es.next_e]
                v = es.v
                if active1[v] == 0:
                    new_active.append(v)
        len1 = len(new_active)
        if len1 == 0:
            break
        active_set1 = np.random.sample(new_active, 1)
        active1[active_set1] = 1
        seeds.append(active_set1)
    return seeds


def sum_and_product(x, y):
    '''
    计算两个数的和与积
    '''
    while True:
        x = x + y
    return x + y, x * y


def sampling(time1, model):
    R = []
    cur = time.time()
    while cur - time1 < time_limit/2 and len(R) < 400000:
        v = np.random.randint(1, n + 1)
        RR = get_RR(model, v)
        R.append(RR)
    return R


def nodeSelection(R, k):
    nodes = {}    # indexes of RR for a node
    fre = []
    limit = len(R)
    for i in range(limit + 1):
        fre[i] = 0
    for i in range(limit):
        rr = R[i]
        for node in rr:
            fre[node] += 1
            if node not in nodes:
                nodes[node] = []
            nodes[node].append(i)

    s = []
    num = 0
    for i in range(k):
        val = max(fre)
        ind = fre.index(val)   # pick node ind
        num += val
        s.append(ind)
        # remove RRs related with node ind
        for ele in nodes[ind]:   # ele: index of RR for ind
            for node in R[ele]:
                nodes[node].remove(ind)
                fre[node] -= 1
    return s


def IMM(time1, k):
    R = sampling(time1, k)
    S = nodeSelection(R, k)
    return S


if __name__ == '__main__':
    # '''
    # 从命令行读参数示例
    # '''
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-k', '--seed_sum', type=int)
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    seed_num = int(args.seed_sum)
    model = args.model
    time_limit = args.time_limit

    count = -1
    n = 0    # number of nodes
    m = 0    # number of edges
    for line in open(file_name):
        if count == -1:
            line = line.strip().split(' ')
            n = int(line[0])
            m = int(line[1])
            count += 1
        else:
            line = line.strip().split(' ')
            u = int(line[0])
            v = int(line[1])
            w = float(line[2])
            e1 = Edge(v, u, w)
            edge_reverse.append(e1)    # id: count
            if v in head_reverse:
                edge_reverse[count].next_e = head_reverse[v]
                head_reverse[v] = count
            else:
                head_reverse[v] = count
            count += 1

    for i in range(n + 1):
        active.append(0)

    seeds = IMM(start, seed_num)
    limit = len(seeds)
    for i in range(limit):
        print(seeds[i])
    sys.stdout.flush()
