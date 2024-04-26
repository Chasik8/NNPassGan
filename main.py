import torch
import numpy as np
from model import Net_rand, Net_detection
import torch.nn as nn
import time
from tqdm import tqdm
from torch.autograd import Variable
from keyboard import is_pressed

def Trainx(kol, batcht):
    batch = batcht
    train_x = torch.from_numpy(np.random.uniform(0, 1, size=(batch, kol)).astype(np.float32))
    return train_x


def Trainy(kol, batch):
    f = open("10_million_password_list_top_1000000.txt.txt", 'r')
    l = list(map(str, f.read().split()))
    l = l[:batch]
    f.close()
    train_y = []
    for i in range(len(l)):
        if l[i][-1] == '\n':
            l[i] = l[i][:-1]
        p = []
        for j in l[i]:
            s = str(bin(ord(j)))[2:]
            s = '0' * (8 - len(s)) + s
            for o in s:
                p.append(int(o))
        # for j in l[i]:
        #     p.append(ord(j) / 256)
        # if kol > len(l[i]):
        #     for j in range(kol - len(l[i])):
        #         p.append(-10)

        if (kol // 8 < len(l[i])):
            print(f"len password!: {len(l[i])}")
            exit()
        p += [0] * (kol - len(p))
        train_y.append(p)
    train_y = np.array(train_y)
    # train_y = np.transpose(train_y)
    train_y = train_y.astype(np.float32)
    train_y = torch.from_numpy(train_y)
    return train_y


def save(G, D, k_model):
    torch.save(G.state_dict(), fr"models\Gmodel{k_model}.pth")
    torch.save(D.state_dict(), fr"models\Dmodel{k_model}.pth")
    print("Save model")


def Run_hab():
    k_model = 0
    train_dop = True
    # не никакого смысла в проходе больше одного 1 ошибка на следующих 0
    Gk_old = 1
    Gk = Gk_old
    Gk_dop = 5
    Dk = 1
    epoch_kol = int(1e5)
    batch = 100000
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
    # Dcriterion = nn.BCELoss()
    # Gcriterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=net.learning_rate)
    Goptimizer = torch.optim.Adam(G.parameters())
    Doptimizer = torch.optim.Adam(D.parameters())
    y_train = Trainy(G.out(), batch)
    y_train = y_train.to(dev)
    # Gdop = torch.ones([np.ones(G.out()).shape[0], 1])
    # Gdop = Gdop.to(dev)
    Ddopfalse = torch.tensor(np.array([0]).astype(np.float32))
    Ddopfalse = Ddopfalse.to(dev)
    Ddoptrue = torch.tensor(np.array([1]).astype(np.float32))
    Ddoptrue = Ddoptrue.to(dev)
    Gdoptrue = torch.tensor(np.array([1] * G.out()).astype(np.float32))
    Gdoptrue = Gdoptrue.to(dev)
    loss_max = 1000000000000000000000000
    if train_dop:
        ft = open(f"floss_dir\\floss_max.txt", 'r')
        loss_max = float(ft.read())
        ft.close()
    # ---------------------------------------------------------------------------------------------------
    Dsr_loss = 1
    tim = time.time()
    for epoch in range(epoch_kol):
        Gsr_loss = 0
        # if Dsr_loss > 1e-05:
        Dsr_loss = 0
        for k in range(Dk):
            x_train_set = Trainx(G.inp(), batch)
            x_train_set = x_train_set.to(dev)
            real_outputs = D(y_train)
            real_label = torch.ones(batch, 1).to(dev)
            fake_inputs = G(x_train_set)
            fake_outputs = D(fake_inputs)
            fake_label = torch.zeros(batch, 1).to(dev)
            outputs = torch.cat((real_outputs, fake_outputs), 0)
            targets = torch.cat((real_label, fake_label), 0)
            D_loss = Dcriterion(outputs, targets)
            Doptimizer.zero_grad()
            D_loss.backward()
            Doptimizer.step()
            Dsr_loss += float(D_loss.item())
            # for imgs, x_train in zip(y_train, x_train_set):
            #     real_outputs = D(imgs)
            #     real_label = torch.ones(1).to(dev)
            #     fake_inputs = G(x_train)
            #     fake_outputs = D(fake_inputs)
            #     fake_label = torch.zeros(1).to(dev)
            #     outputs = torch.cat((real_outputs, fake_outputs), 0)
            #     targets = torch.cat((real_label, fake_label), 0)
            #     D_loss = Dcriterion(outputs, targets)
            #     Doptimizer.zero_grad()
            #     D_loss.backward()
            #     Doptimizer.step()
            #     Dsr_loss += float(D_loss.item())
        if Dk != 0:
            Dsr_loss /= batch * Dk
        else:
            Dsr_loss = 1e10
        for k in range(Gk):
            x_train_set = Trainx(G.inp(), batch)
            x_train_set = x_train_set.to(dev)
            fake_inputs = G(x_train_set)
            fake_outputs = D(fake_inputs)
            fake_targets = torch.ones(batch, 1).to(dev)
            G_loss = Gcriterion(fake_outputs, fake_targets)
            Goptimizer.zero_grad()
            G_loss.backward()
            Goptimizer.step()
            Gsr_loss += float(G_loss.item())
            # for x_train in x_train_set:
            #     fake_inputs = G(x_train)
            #     fake_outputs = D(fake_inputs)
            #     fake_targets = torch.ones(1).to(dev)
            #     G_loss = Gcriterion(fake_outputs, fake_targets)
            #     Goptimizer.zero_grad()
            #     G_loss.backward()
            #     Goptimizer.step()
            #     Gsr_loss += float(G_loss.item())
        print(epoch)
        if Gk != 0:
            Gsr_loss /= batch * Gk
        else:
            Gsr_loss = 1e10
        if Gsr_loss >= 0.95:
            Gk = Gk_dop
        else:
            Gk = Gk_old
        print("D:", Dsr_loss)
        print("G:", Gsr_loss)
        if 10 ** (-4) >= Dsr_loss or Dk == 0:
            if Gsr_loss < loss_max:
                loss_max = Gsr_loss
                torch.save(G.state_dict(), fr"models\Gmodel{k_model}_max.pth")
                torch.save(D.state_dict(), fr"models\Dmodel{k_model}_max.pth")
                floss_max = open("floss_dir\\floss_max.txt", 'w')
                floss_max.write(str(Dsr_loss))
                floss_max.close()
        if epoch % 1 == 0:
            torch.save(G.state_dict(), fr"models\Gmodel{k_model}.pth")
            torch.save(D.state_dict(), fr"models\Dmodel{k_model}.pth")
            print("Save model")
        # torch.save(G.state_dict(), fr"models\Gmodel{str(k_model)}.pth")
        # torch.save(D.state_dict(), fr"models\Dmodel{str(k_model)}.pth")
        if is_pressed('s'):
            torch.save(G.state_dict(), fr"models\Gmodel{k_model}.pth")
            torch.save(D.state_dict(), fr"models\Dmodel{k_model}.pth")
            print("Save model")
        print(time.time() - tim)


def print_hi(name):
    tim = time.time()

    Run_hab()
    print(time.time() - tim)


if __name__ == '__main__':
    print_hi('PyCharm')
# flatten
# >>> t = torch.tensor([[[1, 2],
# ...                    [3, 4]],
# ...                   [[5, 6],
# ...                    [7, 8]]])
# >>> torch.flatten(t)
# tensor([1, 2, 3, 4, 5, 6, 7, 8])
# >>> torch.flatten(t, start_dim=1)
# tensor([[1, 2, 3, 4],
#         [5, 6, 7, 8]])
