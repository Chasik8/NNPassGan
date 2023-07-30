import torch
import numpy as np
from model import Net_rand, Net_detection
import torch.nn as nn
import time


def Trainx(kol):
    batch = 1
    train_x = torch.from_numpy(np.random.normal(0, 1, size=(batch, kol)).astype(np.float32))
    return train_x


def Run():
    k_model = 0
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
    kol_passsord = 1
    for i in range(kol_passsord):
        h = True
        while h:
            x = Trainx(G.inp())
            x = x.to(dev)
            out = G(x[0])
            prow = D(out)
            if prow.detach().cpu().numpy()[0] >= 0.5:
                h = False
                out_np = out.detach().cpu().numpy()
                s = ''
                for j in out_np:
                    jj = int(j * 256)
                    if jj < 0:
                        s += chr(0)
                    elif jj > 255:
                        s += chr(255)
                    else:
                        s += chr(jj)

                f.write(s)
                f.write('\n')
    f.close()


def main():
    tim = time.time()
    Run()
    print(time.time() - tim)


if __name__ == '__main__':
    main()
