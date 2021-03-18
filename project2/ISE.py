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


head = {}
edge = []
active_set = []
active = []
weight_total = []


def IC(head, edge, active_set, active):
    head1 = head.copy()
    edge1 = edge[:]
    active_set1 = active_set[:]
    active1 = active[:]
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
                while es.next_e != -1:
                    es = edge1[es.next_e]
                    v = es.v
                    w = es.w
                    if active1[v] == 0:
                        ran = np.random.random()
                        if ran <= w:
                            active1[v] = 1
                            new_active.append(v)
        len1 = len(new_active)
        ans += len1
        active_set1 = new_active
    return ans


def LT(n, head, edge, active_set, active, weight_total):
    head1 = head.copy()
    edge1 = edge[:]
    active_set1 = active_set[:]
    active1 = active[:]
    weight_total1 = weight_total[:]
    thresh = []
    for i in range(n + 1):
        thresh.append(np.random.random())
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
                    weight_total1[v] += w
                    if weight_total1[v] >= thresh[v]:
                        new_active.append(v)
                        active1[v] = 1
                while es.next_e != -1:
                    es = edge1[es.next_e]
                    v = es.v
                    w = es.w
                    if active1[v] == 0:
                        weight_total1[v] += w
                        if weight_total1[v] >= thresh[v]:
                            new_active.append(v)
                            active1[v] = 1
        len1 = len(new_active)
        active_set1 = new_active
        ans += len1
    return ans


def sum_and_product(x, y):
    '''
    计算两个数的和与积
    '''
    while True:
        x = x + y
    return x + y, x * y


if __name__ == '__main__':
    # '''
    # 从命令行读参数示例
    # '''
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-s', '--seed', type=str, default='seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    seed = args.seed
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
            e = Edge(u, v, w)
            edge.append(e)    # id: count
            if u in head:
                edge[count].next_e = head[u]
                head[u] = count
            else:
                head[u] = count
            count += 1

    for i in range(n + 1):
        active.append(0)
        weight_total.append(0)

    for line in open(seed):
        line = int(line.strip())
        active[line] = 1
        active_set.append(line)

    if model == 'IC':
        summ = 0
        start1 = time.time()
        for i in range(10):
            ans1 = IC(head, edge, active_set, active)
            summ += ans1
        end = time.time()
        gap = (end - start1) / 10
        N = int(((time_limit - 1)*0.8 - (end - start) - gap * 10) / gap)
        if N < 0:
            N = 0
        for i in range(N):
            ans1 = IC(head, edge, active_set, active)
            summ += ans1
        summ = (summ/(N + 10))
        print(summ)
    if model == 'LT':
        summ = 0
        start1 = time.time()
        for i in range(10):
            ans1 = LT(n, head, edge, active_set, active, weight_total)
            summ += ans1
        end = time.time()
        gap = (end - start1) / 10
        N = int(((time_limit - 1)*0.8 - (end - start) - gap * 10) / gap)
        if N < 0:
            N = 0
        for i in range(N):
            ans1 = LT(n, head, edge, active_set, active, weight_total)
            summ += ans1
        summ = (summ / (N + 10))
        print(summ)

    sys.stdout.flush()
