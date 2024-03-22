import numpy as np
import requests as rq
import torch


def dop1():
    w = rq.get("https://xn--e1alhsoq4c.xn--p1ai/base/Mail.txt")
    s = w.text
    l1 = list(map(str, s.split('\n')))
    f1 = open("text1_mail_log.txt", 'w')
    f2 = open("text1_mail_pass.txt", 'w')
    print(len(l1))
    for i in l1:
        l2 = list(map(str, i.split(':')))
        f1.write(l2[0])
        f1.write('\n')
        f2.write((l2[-1]))
    f1.close()
    f2.close()


def dop2():
    f = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t = torch.ones([f.shape[0], 1])
    print(t)


def dop3():
    print(int("101", 2))


if __name__ == '__main__':
    dop3()
