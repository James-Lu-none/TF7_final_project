import os
import numpy as np
import pickle
from common import check_win
from pathlib import Path
from data_set_trans import *

root_path = os.getcwd()
print(root_path)

data_dir = os.path.join(root_path,'playok_data_collector/data')
data_files = os.listdir(data_dir)
all_data = np.zeros((15, 15))

for f in data_files:
    print("reading file {}".format(f))
    f = "{}/{}".format(data_dir, f)
    with open(f, 'rb') as f:
        data = pickle.load(f)

    array_data = np.array(data)

    for game in array_data:
        all_data = np.append(all_data, game, axis=0)

all_data = all_data.reshape(int(all_data.shape[0]/15), 15, 15)
all_data = all_data[1:]
print(all_data.shape)

all_data_new = np.zeros((15, 15))
all_data_shit = np.zeros((15, 15))
b=0
print(all_data.shape[0])
for _ in range(all_data.shape[0]):
    lastmove_where = np.where(all_data[_]==np.max(all_data[_]))
    lastmove_number = np.max(all_data[_])
    if check_win(all_data[_], lastmove_where[0][0], lastmove_where[1][0], lastmove_number):
        all_data_new = np.append(all_data_new, all_data[_], axis=0)
    else:
        all_data_shit = np.append(all_data_shit, all_data[_], axis=0)
    
all_data_new = all_data_new.reshape(int(all_data_new.shape[0]/15), 15, 15)
all_data_new = all_data_new[1:]
all_data_shit = all_data_shit.reshape(int(all_data_shit.shape[0]/15), 15, 15)
all_data_shit = all_data_shit[1:]

print(all_data_new.shape)
print(all_data_shit.shape)

tmp_dir = os.path.join(root_path,'data_transform/tmp')
for i in [1,2,3,4]:
    last_n=i
    # -------------------------------
    data_final = []
    for idx,game_state in enumerate(np.array(all_data_new)):
        print(idx)
        data_final.append(data_trans(game_state,last_n))
    # -------------------------------
    save_trans_data(tmp_dir,data_final,'trans_from_playok',last_n)
# csv_file = 'Gomoku_Data.csv'
# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(data_final)

# #[第幾盤][第幾手][0]  [特徵0~3][x][y] 局勢
# #[第幾盤][第幾手][1] 機率
# #[第幾盤][第幾手][2] 誰贏
# print("/n/n/ndata:")
# print(all_data[9])
# print(data_final[9])