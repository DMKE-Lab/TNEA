import utils
import numpy as np
import sys

def time_pre(dataset_str):
    names = [['triples_1', 'triples_2'], ['ref_ent_ids']]
    for fns in names:
        for i in range(len(fns)):
            fns[i] = 'data/' + dataset_str + '/' + fns[i]
    Ts, ill = names
    ill = ill[0]
    ILL = utils.loadfile(ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = ILL[:illL // 10 * 3]
    KG1 = utils.loadfile(Ts[0], 3)
    KG2 = utils.loadfile(Ts[1], 3)
    process_time(train, KG1, KG2, dataset_str, 'triples_1_tem.npy', 'triples_2_tem.npy')


def process_time(train_list, KG1, KG2, dateset_str, temFile_1, temFile_2):
    import time
    start = time.time()

    path = 'data/' + dateset_str + '/'
    tem1 = np.load( path + temFile_1).tolist()
    tem2 = np.load( path + temFile_2).tolist()
    assert len(KG1) + len(KG2) == len(tem1) + len(tem2)

    train = set(train_list)
    left = [x[0] for x in train]
    right = [x[1] for x in train]

    timeSource = dict()
    for rg_1 in range(len(KG1)):
        id_1 = KG1[rg_1][0]

        if id_1 not in left:
            continue
        for rg_2 in range(len(KG2)):
            id_2 = KG2[rg_2][0]
            if (id_1, id_2) not in train and (KG1[rg_1][2], KG2[rg_2][2]) in train:
                if id_1 not in timeSource:
                    timeSource[id_1] = {}
                if id_2 not in timeSource[id_1]:
                    timeSource[id_1][id_2] = rg_1
                else:
                    timeSource[id_1][id_2] *= 100000
                    timeSource[id_1][id_2] += rg_1
    #{id1:{id2:time, id2:time}}


    if True:
        print('rg_1', time.time() - start, len(timeSource))
        np.save(path + 'timeSource.npy', timeSource)
        print(len(timeSource))
        print(sys.getsizeof(timeSource))
        for i in range(20):
            test = timeSource.popitem()
            if len(test[1])<20:
                print(test[0], test[1], '\n\n')
            else:
                print('\n\n')
        exit(0)

    start = time.time()
    for rg_2 in range(len(KG2)):
        id_2 = KG2[rg_2][0]
        if rg_2 % 10000 == 0:
            print('rg_2', rg_2, time.time() - start)
        if id_2 not in right:
            continue
        for rg_1 in range(len(KG1)):
            id_1 = KG1[rg_1][0]
            if (id_1, id_2) not in train and (KG1[rg_1][2], KG2[rg_2][2]) in train:
                timeSource[(id_2, id_1)] = timeSource.get((id_2, id_1), []) + [rg_2]



time_pre('zh_en')
'''
rg_1 5000 152.664311170578 174845
174845
(2521, 36463) [4999]
(2521, 12645) [4999]
(2521, 15792) [4999]
(2521, 38185) [4999]
(2521, 35060) [4999, 4999]
(2521, 19480) [4999]
(2521, 20634) [4999, 4999]
(2521, 16165) [4999]
(2521, 35766) [4999]
(5840, 14076) [4998]


7281 {15588: 4953, 13526: 4953, 38302: 4953, 11589: 4953, 38544: 4953, 13757: 4953, 14548: 4953, 34500: 4953, 37523: 4953, 36535: 4953, 20796: 4953, 14115: 4953, 17676: 4953, 17894: 4953}

len = 25xx
size = 7xxkb

23662 {35091: 69197, 10739: 69197, 36489: 69197, 15405: 69197, 13785: 69197, 36092: 69197, 16497: 69197, 14512: 69197, 14336: 69197, 15714: 69197, 13416: 69197, 18684: 69197, 11849: 69197, 15385: 69197, 16584: 69197, 17145: 69197, 38181: 69197, 16559: 69197, 16955: 69197, 17393: 69197, 20387: 69197, 14547: 69197, 16074: 69197, 14418: 69197, 16295: 69197, 16411: 69197, 35494: 69197, 16138: 69197, 17266: 69197, 36179: 69197, 12177: 69197, 16880: 69197, 14627: 69197, 37743: 69197, 37881: 69197, 20845: 69197, 13668: 69197, 15285: 69197, 15828: 69197, 19760: 69197, 16494: 6919769197, 20837: 69197, 18474: 69197, 15585: 69197, 34852: 69197, 19676: 69197, 11355: 69197, 36754: 69197, 19555: 69197, 17198: 6919769197, 16092: 69197, 20988: 69197, 17125: 69197, 15985: 69197, 35979: 6919769197, 18602: 69197, 11834: 69197, 34789: 69197, 17552: 69197, 37433: 69197, 12348: 69197, 19800: 69197, 15173: 69197, 17543: 69197, 16113: 69197, 12688: 69197, 11715: 69197, 38495: 69197, 19309: 69197, 20533: 69197, 20392: 69197, 37386: 69197, 19704: 69197, 38024: 69197}

'''