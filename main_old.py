class BinomialDevianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.L = nn.L1Loss()
        self.L = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    def forward(self, x, D):
        a = D(x)
        a = self.L(torch.ones(1).to("cuda:0"), a)
        a = Variable(a, requires_grad=True).to("cuda:0")
        return a
def Run():
    k_model = 0
    train_dop = False
    epoch_kol = 1
    batch = 1000
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
    Gcriterion = BinomialDevianceLoss()
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
    # ----------------------------------------------------------------------
    tim = time.time()
    for epoch in range(epoch_kol):
        # for i, (images, labels) in enumerate(train_loader):  # Загрузка партии изображений с индексом, данными,
        # классом
        x_train = Trainx(G.inp(), batch)
        x_train = x_train.to(dev)
        Dloss_train = []
        Depoch_kol = 1
        Gepoch_kol = 1
        train_D = False
        for i in range(len(x_train)):
            Dloss_train.append(G(x_train[i]))
        print(epoch)
        print("Train")
        Dsr_loss = float(0)
        Gsr_loss=float(0)
        for Depoch in range(Depoch_kol):
            Dsr_loss = float(0)
            if train_D:
                for i in (range(len(x_train))):
                    # Goutputs = G(x_train[i])
                    Doutputs = D(Dloss_train[i])
                    Dloss = Dcriterion(Doutputs, Ddopfalse)
                    Dsr_loss += float(Dloss.item())
                    # -----------------------
                    Doptimizer.zero_grad()
                    Dloss.backward(retain_graph=True)
                    Doptimizer.step()
                for i in (range(len(y_train))):
                    Doutputs = D(y_train[i])
                    Dloss = Dcriterion(Doutputs, Ddoptrue)
                    Dsr_loss += float(Dloss.item())
                    # -----------------------
                    Doptimizer.zero_grad()
                    Dloss.backward()
                    Doptimizer.step()
                    # ----------------------------
            else:
                Dsr_loss = 10 ** -5
            # print(Dsr_loss / (len(x_train) + len(y_train)))
        print("Gloss")
        for Gepoch in range(Gepoch_kol):
            Gsr_loss = float(0)
            for i in (range(len(x_train))):
                Gloss = Gcriterion(G(x_train[i]), D)
                # -----------------------
                Goptimizer.zero_grad()
                Gloss.backward(retain_graph=True)
                Goptimizer.step()
                Gsr_loss += float(Gloss.item())
                # ----------------------------
        # print(loss.data.text)
        print("D:", Dsr_loss / (len(x_train) + len(y_train)) / Depoch_kol)
        print("G:", Gsr_loss / (len(Dloss_train)))
        print(time.time() - tim)
        # if Dsr_loss / (len(x_train) + len(y_train)) / Depoch_kol < loss_max:
        if Gsr_loss / len(x_train) < loss_max and 10 ** (-4) >= Dsr_loss:
            loss_max = Gsr_loss / len(x_train)
            torch.save(G.state_dict(), fr"models\Gmodel{k_model}_max.pth")
            torch.save(D.state_dict(), fr"models\Dmodel{k_model}_max.pth")
            floss_max = open("floss_dir\\floss_max.txt", 'w')
            floss_max.write(str(Dsr_loss / (len(x_train) + len(y_train))))
            floss_max.close()
        if epoch % 1 == 0:
            torch.save(G.state_dict(), fr"models\Gmodel{k_model}.pth")
            torch.save(D.state_dict(), fr"models\Dmodel{k_model}.pth")
    torch.save(G.state_dict(), fr"models\Gmodel{str(k_model)}.pth")
    torch.save(D.state_dict(), fr"models\Dmodel{str(k_model)}.pth")
    # save(G, D, k_model)