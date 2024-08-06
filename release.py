import torch
import numpy as np
from model import Net_rand, Net_detection
import torch.nn as nn
import time


def Trainx(kol):
    batch = 1
    train_x = torch.from_numpy(np.random.uniform(0, 1, size=(batch, kol)).astype(np.float32))
    return train_x


def Run():
    # можно переделать сделать выход не ввидде asci а в виде конкретных символов
    ff = open('conf_model.txt', 'r')
    k_model = int(ff.read()) - 1
    # k_model = 4
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
    G.to(dev)
    D.to(dev)
    f = open("Output.txt", 'w')
    f_log = open("Output_log.txt", 'w')
    kol_passsord = 10
    k = 0
    for i in range(kol_passsord):
        h = True
        while h:
            x = Trainx(G.inp())[0]
            # x=torch.from_numpy(np.array([0 for i in range(int(G.inp()))]).astype(np.float32))
            x = x.to(dev)
            out = G(x)
            prow = D(out)
            if k % 1e3 == 0:
                print(k, prow.detach().cpu().numpy()[0])
            k += 1
            # print(prow.detach().cpu().numpy()[0])
            # if prow.detach().cpu().numpy()[0] >= 0.5:
            if True:
                print(i)
                h = False
                out_np = out.detach().cpu().numpy()
                s = ''
                s_log = ''
                sym = ""
                for j in range(len(out_np)):
                    # print(out_np[j])
                    if out_np[j] >= 0.5:
                        sym += '1'
                    else:
                        sym += '0'
                    if j % 8 == 7:
                        dig = int(sym, 2)
                        if dig != 0:
                            s += chr(dig)
                        s_log += chr(dig)
                        sym = ''
                # for j in out_np:
                #     jj = int(j * 256)
                #     if jj < 0:
                #         s += chr(0)
                #     elif jj > 255:
                #         s += chr(255)
                #     else:
                #         s += chr(jj)
                f_log.write(s_log)
                f_log.write('\n')
                f.write(s)
                f.write('\n')
    f.close()
    f_log.close()


def main():
    tim = time.time()
    Run()
    print(time.time() - tim)


if __name__ == '__main__':
    main()
