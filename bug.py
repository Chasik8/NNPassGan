import torch
import numpy as np
from model import Net_rand, Net_detection
import torch.nn as nn
import time
from tqdm import tqdm


def kol_weight(N, name):
    n = 10
    kol = [0 for i in range(n)]
    for i in N.children():
        if isinstance(i, nn.Linear):
            for j in tqdm(i.weight):
                for k in j:
                    for p in range(n):
                        if 10 ** (-(p + 1)) < abs(k) < 10 ** (-p):
                            kol[p] += 1
                            break
    f = open(f"{name}weight.txt", 'w')
    for i in range(len(kol)):
        f.write(f"{i} {str(i)}")
        f.write('\n')
    f.close()
    print(name, *kol)


def Run():
    ff = open('conf_model.txt', 'r')
    k_model = int(ff.read()) - 1
    # k_model=0
    ff.close()
    dev = torch.device("cuda:0")
    G = Net_rand()
    D = Net_detection()
    PATH = f"models\Gmodel{str(k_model)}.pth"
    G.load_state_dict(torch.load(PATH))
    G.eval()
    PATH = f"models\Dmodel{str(k_model)}.pth"
    D.load_state_dict(torch.load(PATH))
    D.eval()
    kol_weight(G, 'G')
    kol_weight(D, 'D')
    # G.to(dev)
    # D.to(dev)
    # for layer in G.children():
    #     if isinstance(layer, nn.Linear):
    #         # for j in layer.state_dict()['weight']:
    #         #     print(j)
    #         #     break
    #         print(layer.state_dict()['bias'])


def main():
    tim = time.time()
    Run()
    print(time.time() - tim)


if __name__ == '__main__':
    main()
