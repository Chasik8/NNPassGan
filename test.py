
def test_rel1():
    kol=320
    batch=10
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
    for yy in train_y:
        y=yy
        s = ''
        s_log = ''
        sym = ""
        for j in range(len(y)):
            # print(out_np[j])
            if y[j] >= 0.5:
                sym += '1'
            else:
                sym += '0'
            if j % 8 == 7:
                dig = int(sym, 2)
                if dig != 0:
                    s += chr(dig)
                s_log += chr(dig)
                sym = ''
        print(s)

if __name__ == '__main__':
    test_rel1()
