import torch
import numpy as np
from model_batch import Net_rand, Net_detection
import torch.nn as nn
import time
from tqdm import tqdm

def Trainx(kol):
    batch = 1000
    train_x = torch.from_numpy(np.random.normal(0, 1, size=(batch, kol)).astype(np.float32))
    return train_x


def Trainy(kol):
    f = open("10_million_password_list_top_1000000.txt.txt", 'r')
    l = list(map(str, f.read().split()))
    l = l[:1000]
    f.close()
    train_y = []
    for i in range(len(l)):
        if l[i][-1] == '\n':
            l[i] = l[i][:-1]
        p = []
        for j in l[i]:
            p.append(ord(j) / 256)
        if kol > len(l[i]):
            for j in range(kol - len(l[i])):
                p.append(-10)
        if (kol < len(l[i])):
            print(f"len password!: {len(l[i])}")
            exit()
        train_y.append(p)
    train_y = np.array(train_y)
    train_y = train_y.astype(np.float32)
    train_y = torch.from_numpy(train_y)
    return train_y


def save(G, D,kol_model):
    torch.save(G.state_dict(), f"D:\\Project\\Python\\Neroset_PassGan\\models\\Gmodel{str(kol_model)}.pth")
    torch.save(D.state_dict(), f"D:\\Project\\Python\\Neroset_PassGan\\models\\Dmodel{str(kol_model)}.pth")


def Run():
    k_model = 0
    train_dop = False
    batch=100
    try:
        ff = open('conf_model.txt', 'r')
        k_model = int(ff.read())
        ff.close()
        ff = open('conf_model.txt', 'w')
        ff.write(str(k_model + 1))
        ff.close()
    except:
        ff = open('conf_model.txt', 'w')
        ff.write(str(1))
        ff.close()
    dev = torch.device("cuda:0")
    G = Net_rand(batch)
    D = Net_detection()
    if train_dop:
        PATH = f"models\Gmodel{str(k_model - 1)}.pth"
        G.load_state_dict(torch.load(PATH))
        G.eval()
        PATH = f"models\Dmodel{str(k_model - 1)}.pth"
        D.load_state_dict(torch.load(PATH))
        D.eval()
    G.to(dev)
    D.to(dev)
    Dcriterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    Gcriterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # optimizer = torch.optim.Adam(net.parameters(), lr=net.learning_rate)
    Goptimizer = torch.optim.Adam(G.parameters())
    Doptimizer = torch.optim.Adam(D.parameters())
    y_train = Trainy(G.out())
    y_train = y_train.to(dev)
    # Gdop = torch.ones([np.ones(G.out()).shape[0], 1])
    # Gdop = Gdop.to(dev)
    Ddopfalse = torch.tensor(np.array([0]).astype(np.float32))
    Ddopfalse = Ddopfalse.to(dev)
    Ddoptrue = torch.tensor(np.array([1]).astype(np.float32))
    Ddoptrue = Ddoptrue.to(dev)
    Gdoptrue = torch.tensor(np.array([1]*G.out()).astype(np.float32))
    Gdoptrue =Gdoptrue.to(dev)
    loss_max = 1000000000000000000000000
    epoch_kol = 1
    if train_dop:
        ft = open(f"floss_dir\\floss_max.txt", 'r')
        loss_max = float(ft.read())
        ft.close()
    # ----------------------------------------------------------------------
    for epoch in range(epoch_kol):
        # for i, (images, labels) in enumerate(train_loader):  # Загрузка партии изображений с индексом, данными,
        # классом
        x_train = Trainx(G.inp())
        x_train = x_train.to(dev)
        Dloss_train = []
        Depoch_kol = 100
        for i in range(0,len(x_train),batch):
            Dloss_train.append(G(x_train[i:i+batch]))
        print("Train")
        Dsr_loss = float(0)
        for Depoch in range(Depoch_kol):
            Dsr_loss = float(0)
            for i in (range(len(x_train))):

                # Goutputs = G(x_train[i])
                Doutputs = D(Dloss_train[i])
                Dloss = Dcriterion(Doutputs, Ddopfalse)
                Dsr_loss += float(Dloss.item())
                # -----------------------
                Doptimizer.zero_grad()
                Dloss.backward(retain_graph=True)
                Doptimizer.step()
                # ----------------------------
                # очень самнительно Gdop. Мы считаем ошибку, как будто выход G должен состоять из 1
                # Glossone=Gcriterion(Goutputs,Gdop)
                # Gloss.append(Glossone)
            # for i in range(len(Gloss)):
            #     Goptimizer.zero_grad()
            #     Gloss[i].backward()
            #     Goptimizer.step()
            for i in (range(len(y_train))):

                Doutputs = D(y_train[i])
                Dloss = Dcriterion(Doutputs, Ddoptrue)
                Dsr_loss += float(Dloss.item())
                # -----------------------
                Doptimizer.zero_grad()
                Dloss.backward()
                Doptimizer.step()
                # ----------------------------
            # print(Dsr_loss / (len(x_train) + len(y_train)))
        print("Gloss")
        for i in (range(0, len(Dloss_train), batch)):

            # Goutputs = G(x_train[i])
            # Doutputs = D(Dloss_train[i])
            Gloss = Gcriterion(Dloss_train[i], Gdoptrue)
            # -----------------------
            Goptimizer.zero_grad()
            Gloss.backward(retain_graph=True)
            # torch.clip_grad_norm_(value_model.parameters(), clip_grad_norm)
            Goptimizer.step()
            # ----------------------------
        # print(loss.data.text)
        print(Dsr_loss/(len(x_train)+len(y_train))/Depoch_kol)
        if Dsr_loss / (len(x_train)+len(y_train))/Depoch_kol < loss_max:
            loss_max = Dsr_loss / len(x_train)
            torch.save(G.state_dict(), fr"models\Gmodel{k_model}_max.pth")
            torch.save(D.state_dict(), fr"models\Dmodel{k_model}_max.pth")
            floss_max = open("floss_dir\\floss_max.txt", 'w')
            floss_max.write(str(Dsr_loss / (len(x_train)+len(y_train))))
            floss_max.close()
        if epoch % 10 == 0:
            torch.save(G.state_dict(), fr"models\Gmodel{k_model}.pth")
            torch.save(D.state_dict(), fr"models\Dmodel{k_model}.pth")
    torch.save(G.state_dict(), fr"models\Gmodel{str(k_model)}.pth")
    torch.save(D.state_dict(), fr"models\Dmodel{str(k_model)}.pth")
    # save(G, D, k_model)
def print_hi(name):
    tim = time.time()
    Run()
    print(time.time() - tim)


if __name__ == '__main__':
    print_hi('PyCharm')
