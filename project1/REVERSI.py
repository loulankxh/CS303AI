import numpy as np
import random
import time
import datetime

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

eval1 = np.array(([[1000, -800, 150, 15, 15, 150, -800, 1000],
                   [-800, -800, 10, 1, 1, 10, -800, -800],
                   [150, 10, 5, 0, 0, 5, 10, 150],
                   [15, 1, 0, 0, 0, 0, 1, 15],
                   [15, 1, 0, 0, 0, 0, 1, 15],
                   [150, 10, 5, 0, 0, 5, 10, 150],
                   [-800, -800, 10, 1, 1, 10, -800, -800],
                   [1000, -800, 150, 15, 15, 150, -800, 1000]]))
eval2 = np.ones((8, 8), dtype=np.int)


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    def go(self, chessboard_go):
        self.candidate_list.clear()
        self.check(chessboard_go)

    def check(self, chessboard_check):
        self.candidate_list = self.get_list(chessboard_check, self.color)
        self.minimax_alphabeta(chessboard_check)

    def get_list(self, chessboard_list, color_ge):
        a_list = []
        direction = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if chessboard_list[i][j] == COLOR_NONE:
                    flag = 0
                    count = 0
                    while count < 8 and flag == 0:
                        x = i
                        y = j  # 不能放在上一层
                        x += direction[count][0]
                        y += direction[count][1]
                        while -1 < x < 8 and -1 < y < 8 and chessboard_list[x][y] == -color_ge:
                            x += direction[count][0]
                            y += direction[count][1]
                        if -1 < x < 8 and -1 < y < 8:
                            if chessboard_list[x][y] == color_ge and chessboard_list[x - direction[count][0]][y - direction[count][1]] == -color_ge:  # -self.color != (!=self.color)
                                flag = 1
                        count += 1
                    if flag == 1:
                        a_list.append((i, j))
        return a_list

    def minimax_alphabeta(self, chessboard_ab):
        def max_value(chessboard_max, alpha, beta, depth):     # self.color
            a_list = self.get_list(chessboard_max, self.color)
            node = (-1, -1)
            if self.terminal(depth, a_list):
                return self.utility(chessboard_max), node
            v = -1e10
            for a in a_list:
                val, ns = min_value(self.result(chessboard_max, a, self.color), alpha, beta, depth - 1)
                v = max(v, val)
                if v >= beta:
                    return v, node
                if v > alpha:
                    alpha = v
                    node = a
            return v, node

        def min_value(chessboard_min, alpha, beta, depth):        # -self.color
            a_list = self.get_list(chessboard_min, -self.color)
            node = (-1, -1)
            if self.terminal(depth, a_list):
                return self.utility(chessboard_min), node
            v = 1e10
            for a in a_list:
                val, ns = max_value(self.result(chessboard_min, a, -self.color), alpha, beta, depth - 1)
                v = min(v, val)
                if v <= alpha:
                    return v, node
                if v < beta:
                    beta = v
                    node = a
            return v, node

        self.candidate_list.sort(key=lambda x: eval1[x[0]][x[1]], reverse=True)
        val, ns = max_value(chessboard_ab, -1e10, 1e10, 3)
        if ns != (-1, -1):
            self.candidate_list.append(ns)

    def terminal(self, depth, a_list):
        if depth == 0:
            return True
        if len(a_list) == 0:
            return True
        return False


    def result(self, chessboard_re, dot, color_re):  # dot: (i, j)
        # new_board = chessboard_re
        new_board = np.zeros((self.chessboard_size, self.chessboard_size), dtype=np.int)
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                new_board[i][j] = chessboard_re[i][j]
        i = dot[0]
        j = dot[1]
        new_board[i][j] = color_re
        direction = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
        for k in range(8):
            x = i
            y = j
            x += direction[k][0]
            y += direction[k][1]
            while -1 < x < 8 and -1 < y < 8 and new_board[x][y] == -color_re:
                x += direction[k][0]
                y += direction[k][1]
            if -1 < x < 8 and -1 < y < 8:
                if new_board[x][y] == color_re and new_board[x - direction[k][0]][y - direction[k][1]] == -color_re:
                    while x != i or y != j:
                        new_board[x][y] = color_re
                        x -= direction[k][0]
                        y -= direction[k][1]
        return new_board

    def cal_stable(self, chessboard):
        num = 0
        visit = np.zeros((8, 8), dtype=np.int)
        if chessboard[0][0] != COLOR_NONE:
            count = 0
            x = 0
            pre = self.chessboard_size
            rec_edge = 0
            while -1 < x < 8 and chessboard[x][0] == chessboard[0][0]:
                y = 0
                while -1 < y < 8 and y < pre and chessboard[x][y] == chessboard[0][0]:
                    if visit[x][y] == 0:
                        visit[x][y] = 1
                        count += 1
                    y += 1
                if rec_edge == 1:   # 上层与上上层是一个矩形边而非梯形
                    if y < pre - 1 and -1 < x - 2 < 8 and -1 < pre < 8:
                        if chessboard[x - 2][pre] == COLOR_NONE:
                            count -= 1  # 上层最后一个子其实不是稳定子
                            visit[x - 1][pre - 1] = 0
                    rec_edge = 0
                if y == pre and x != 0:
                    rec_edge = 1
                pre = y  # 衡量上一层的y的最终范围， 要符合梯形
                x += 1
            if chessboard[0][0] == self.color:
                num += count
            else:
                num -= count

        if chessboard[7][7] != COLOR_NONE:
            count = 0
            x = 7
            pre = -1
            rec_edge = 0
            while -1 < x < 8 and chessboard[x][7] == chessboard[7][7]:
                y = 7
                while -1 < y < 8 and y > pre and chessboard[x][y] == chessboard[7][7]:
                    if visit[x][y] == 0:
                        visit[x][y] = 1
                        count += 1
                    y -= 1
                if rec_edge == 1:
                    if y > pre + 1 and -1 < x + 2 < 8 and -1 < pre < 8:
                        if chessboard[x + 2][pre] == COLOR_NONE:
                            count -= 1
                            visit[x + 1][pre + 1] = 0
                    rec_edge = 0
                if y == pre and x != 7:
                    rec_edge = 1
                pre = y
                x -= 1
            if chessboard[7][7] == self.color:
                num += count
            else:
                num -= count

        if chessboard[0][7] != COLOR_NONE:
            count = 0
            x = 0
            pre = -1
            rec_edge = 0
            while -1 < x < 8 and chessboard[x][7] == chessboard[0][7]:
                y = 7
                while -1 < y < 8 and y > pre and chessboard[x][y] == chessboard[0][7]:
                    if visit[x][y] == 0:
                        visit[x][y] = 1
                        count += 1
                    y -= 1
                if rec_edge == 1:
                    if y > pre + 1 and -1 < x - 2 < 8 and -1 < pre < 8:
                        if chessboard[x - 2][pre] == COLOR_NONE:
                            count -= 1
                            visit[x - 1][pre + 1] = 0
                    rec_edge = 0
                if y == pre and x != 0:
                    rec_edge = 1
                pre = y
                x += 1
            if chessboard[0][7] == self.color:
                num += count
            else:
                num -= count

        if chessboard[7][0] != COLOR_NONE:
            count = 0
            x = 7
            pre = 8
            rec_edge = 0
            while -1 < x < 8 and chessboard[x][0] == chessboard[7][0]:
                y = 0
                while -1 < y < 8 and y < pre and chessboard[x][y] == chessboard[7][0]:
                    if visit[x][y] == 0 :
                        visit[x][y] = 1
                        count += 1
                    y += 1
                if rec_edge == 1:
                    if y < pre - 1 and - 1 < x + 2 < 8 and -1 < pre < 8:
                        if chessboard[x + 2][pre] == COLOR_NONE:
                            count -= 1
                            visit[x + 1][pre - 1] = 0
                pre = y
                x -= 1
            if chessboard[7][0] == self.color:
                num += count
            else:
                num -= count
        return num

    def utility(self, chessboard_ul):
        val1 = np.sum(np.multiply(chessboard_ul, eval1))   # 位置权值   白 - 黑
        val2 = np.sum(np.multiply(chessboard_ul, eval2))   # 棋子个数   白 - 黑
        count = np.sum(np.multiply(chessboard_ul, chessboard_ul))
        list1 = self.get_list(chessboard_ul, self.color)
        list2 = self.get_list(chessboard_ul, -self.color)
        val3 = (len(list1) - len(list2))   # 行动力  self - (!self)
        num = self.cal_stable(chessboard_ul)  # 稳定子

        if count < 15:
            val = (val1 + val2 * 0.5) * self.color + val3 * 15 + num * 1000
        elif count < 35:
            val = (val1 * 0.8 + val2) * self.color + val3 * 60 + num * 1000
        elif count < 45:
            val = (val1 * 0.5 + val2 * 2) * self.color + val3 * 50 + num * 1000
        else:
            val = (val1 * 0.3 + val2 * 10) * self.color + val3 * 50 + num * 1000
        return val






if __name__ == "__main__":
    size = 8
    color = COLOR_BLACK
    chessboard = np.zeros((size, size), dtype=np.int)
    chessboard[3][4], chessboard[4][3], chessboard[3][3], chessboard[4][4] = -1, -1, 1, 1
    ai = AI(size, color, 10000000)
    for turn in range(4, 64):
        start = datetime.datetime.now()
        ai.go(chessboard)
        lens = len(ai.candidate_list)
        print(turn, ai.color)
        print(ai.candidate_list)
        # print(turn, Limit, NumOfFind)
        if lens > 0:
            chessboard = ai.result(chessboard, ai.candidate_list[lens - 1], ai.color)
        # if lens != 0:
        #     do(chessboard, ai.color, ai.candidate_list[lens-1][0], ai.candidate_list[lens-1][1])
        ai.color = -ai.color
        end = datetime.datetime.now()
        print(end-start)
