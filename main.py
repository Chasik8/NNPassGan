import torch
import numpy as np
from model import Net_rand, Net_detection
import torch.nn as nn
import time
from tqdm import tqdm

def Trainx(kol):
    batch = 1
    train_x = torch.from_numpy(np.random.normal(0, 1, size=(batch, kol)).astype(np.float32))
    return train_x


def Trainy(kol):
    f = open("10_million_password_list_top_1000000.txt.txt", 'r')
    l = list(map(str, f.read().split()))
    l = l[:10]
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


def save(G, D):
    torch.save(G.state_dict(), "D:\\Project\\Python\\Neroset_PassGan\\Gmodel1.pth")
    torch.save(D.state_dict(), "D:\\Project\\Python\\Neroset_PassGan\\Dmodel1.pth")


def Run():
    k_model = 0
    train_dop = False
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
    G = Net_rand()
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
    loss_max = 1000000000000000000000000
    epoch_kol = 2
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
        Depoch_kol = 10
        for i in range(len(x_train)):
            Dloss_train.append(G(x_train[i]))
        print("Train")
        Dsr_loss = 0
        for Depoch in range(Depoch_kol):
            Dloss = 0
            for i in tqdm(range(len(x_train))):
                # images = Variable(images.view(-1, 28 * 28))  # Конвертация тензора в переменную: изменяем изображение с
                # вектора, размером 784 на матрицу 28 x 28 labels = Variable(labels)
                # Goptimizer.zero_grad()
                Doptimizer.zero_grad()
                # Goutputs = G(x_train[i])
                Doutputs = D(Dloss_train[i])
                Dloss = Dcriterion(Dloss_train[i], Ddopfalse)
                # -----------------------
                Dloss.backward(retain_graph=True)
                Doptimizer.step()
                # ----------------------------
                Dsr_loss += Dloss.item()
                # очень самнительно Gdop. Мы считаем ошибку, как будто выход G должен состоять из 1
                # Glossone=Gcriterion(Goutputs,Gdop)
                # Gloss.append(Glossone)
            # for i in range(len(Gloss)):
            #     Goptimizer.zero_grad()
            #     Gloss[i].backward()
            #     Goptimizer.step()
            for i in tqdm(range(len(y_train))):
                Doptimizer.zero_grad()
                Doutputs = D(y_train[i])
                Dloss = Dcriterion(Doutputs, Ddoptrue)
                # -----------------------
                Dloss.backward(retain_graph=True)
                Doptimizer.step()
                # ----------------------------
                Dsr_loss += Dloss.item()
        print("Gloss")
        for i in tqdm(range(len(x_train))):
            Goptimizer.zero_grad()
            # Goutputs = G(x_train[i])
            Doutputs = D(Dloss_train[i])
            Gloss = Gcriterion(Dloss_train[i], Ddoptrue)
            # -----------------------
            Gloss.backward()
            Goptimizer.step()
            # ----------------------------
        # print(loss.data.text)
        print(Dsr_loss/(len(x_train)+len(y_train)))
        if Dsr_loss / (len(x_train)+len(y_train)) < loss_max:
            loss_max = Dsr_loss / len(x_train)
            torch.save(G.state_dict(), fr"models\Gmodel{k_model}_max.pth")
            torch.save(D.state_dict(), fr"models\Dmodel{k_model}_max.pth")
            floss_max = open("floss_dir\\floss_max.txt", 'w')
            floss_max.write(str(Dsr_loss / (len(x_train)+len(y_train))))
            floss_max.close()
        if epoch % 10 == 0:
            torch.save(G.state_dict(), fr"models\Gmodel{k_model}.pth")
            torch.save(D.state_dict(), fr"models\Dmodel{k_model}.pth")
    torch.save(G.state_dict(), fr"models\Gmodel{k_model}.pth")
    torch.save(D.state_dict(), fr"models\Dmodel{k_model}.pth")
    save(G, D)

def print_hi(name):
    tim = time.time()
    Run()
    print(time.time() - tim)


if __name__ == '__main__':
    print_hi('PyCharm')
