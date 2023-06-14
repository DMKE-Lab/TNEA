import numpy as np
import sys
import os

#dataFormat   [e, r, e, t]   t: 2010-10-10

# tem_dict
tem_dict = {
    '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
    '01m': 10, '02m': 11, '03m': 12, '04m': 13, '05m': 14, '06m': 15, '07m': 16, '08m': 17, '09m': 18, '10m': 19, '11m': 20, '12m': 21,
    '0d': 22, '1d': 23, '2d': 24, '3d': 25, '4d': 26, '5d': 27, '6d': 28, '7d': 29, '8d': 30, '9d': 31,
}

count = 0
dataset = sys.argv[1]
os.makedirs(dataset + '_TA', exist_ok=True)
path = './data/' + dataset


def preprocess(data_part):
    data_path = path + '/' + data_part
    tem_write_path = path + '/' + data_part + '_tem.npy'
    tem = []
    with open(data_path) as fp:
        for i,line in enumerate(fp):
            global count
            count += 1
            info = line.strip().split("\t")

            year, month, day = info[3].split("-")
            tem_id_list = []
            for j in range(len(year)):
                token = year[j]+'y'
                tem_id_list.append(tem_dict[token])
            #2014 --->  2,0,1,4

            for j in range(1):
                token = month+'m'
                tem_id_list.append(tem_dict[token])
            #10  ---> 19

            for j in range(len(day)):
                token = day[j]+'d'
                tem_id_list.append(tem_dict[token])
            #18  ---> 23,30

            tem.append(tem_id_list)
    np_tem = np.array(tem)
    np.save(tem_write_path, np_tem)


preprocess('triples_1')
preprocess('triples_2')