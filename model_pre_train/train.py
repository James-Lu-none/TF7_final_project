import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import time
import matplotlib.pyplot as plt
from keras.api.layers import *
from keras.api.models import Model, Sequential
from keras.api.regularizers import l2
from keras.api.optimizers import Adam
import keras.api.backend as K

import random
import numpy as np
import pickle
import json
import os

from module_design import model_structure

root_path = os.getcwd()
print(root_path)

board_size = 15
last_n_feature=1
l2_const = 1e-4
batch_size=512
epochs=1
init_lr=1e-4
is_load_model=False
load_model_dir=None# xx_xx_xxxxxx

sample_size=1 # number of games choose from data
take_last_n_move=5 # number of last steps used in each game

data_dir = os.path.join(root_path,f'data_transform/tmp/training_data_v{last_n_feature}')
data_files = os.listdir(data_dir)
data = []
for file in data_files:
    print(f"loading file {file}...")
    file = os.path.join(data_dir,file)
    with open(file, 'rb') as f:
        data += pickle.load(f)
    print(f"file {file} loaded")
print(f"total of {len(data)} game(s) loaded")


status = []
probs = []
wins = []
result = []
moves = []
data=random.sample(data,sample_size)
print(f"total of {len(data)} game(s) used")

for game in data:
    for move_n in game:
        moves.append(move_n)
random.shuffle(moves)
# print(moves[0])
for i in moves:
    status.append(i[0])
    probs.append(i[1])
    wins.append(i[2])

train_size = int(0.8*len(status))
print("total status:{}".format(len(status)))

x_train = status[:train_size]
x_test = status[train_size:]

y_train_probs = probs[:train_size]
y_test_probs = probs[train_size:]

y_train_wins = wins[:train_size]
y_test_wins = wins[train_size:]

print(len(x_train))
print(len(x_test))

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

y_train_probs = np.asarray(y_train_probs)
y_train_probs = np.reshape(y_train_probs, (y_train_probs.shape[0], -1))
y_train_wins = np.asarray(y_train_wins)

y_test_probs = np.asarray(y_test_probs)
y_test_probs = np.reshape(y_test_probs, (y_test_probs.shape[0], -1))
y_test_wins = np.asarray(y_test_wins)


if(is_load_model):
    model=load_model(os.path.join(root_path,'model_record',load_model_dir,'model.h5'))
else:
    model=model_structure(board_size,last_n_feature,l2_const)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=init_lr,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = Adam(
    learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999)

model.compile(
    optimizer=optimizer,
    loss={'policy_net': 'categorical_crossentropy', 'value_net': 'mean_squared_error'},
    metrics={
        'policy_net': ['accuracy'], 
        'value_net': ['mae', 'mean_squared_error']
    })

# print(model.summary())


try:
    start_time=time.process_time()
    history = model.fit(x_train,
                    [y_train_probs, y_train_wins],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    verbose=1,
                    validation_data=(x_test, [y_test_probs, y_test_wins]))
    precess_time=time.process_time()-start_time
except KeyboardInterrupt as e:
    print(e)

result=history
print(result.history.keys())

policy_net_accuracy = history.history['policy_net_accuracy']
val_policy_net_accuracy = history.history['val_policy_net_accuracy']
value_net_mae = history.history['value_net_mae']
val_value_net_mae = history.history['val_value_net_mae']
value_net_mse = history.history['value_net_mean_squared_error']
val_value_net_mse = history.history['val_value_net_mean_squared_error']

# Plotting Policy Net Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(2, 2, 1)
plt.plot(policy_net_accuracy, label='Policy Net Training Accuracy')
plt.plot(val_policy_net_accuracy, label='Policy Net Validation Accuracy')
plt.title('Policy Net Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting Value Net MAE
plt.subplot(2, 2, 2)
plt.plot(value_net_mae, label='Value Net Training MAE')
plt.plot(val_value_net_mae, label='Value Net Validation MAE')
plt.title('Value Net Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

# Plotting Value Net MSE
plt.subplot(2, 2, 3)
plt.plot(value_net_mse, label='Value Net Training MSE')
plt.plot(val_value_net_mse, label='Value Net Validation MSE')
plt.title('Value Net Training and Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()

plt.tight_layout()
plt.show()


score = model.evaluate(
    x_test, [y_test_probs, y_test_wins], verbose=1)

# print("Testing Accuracy = %.2f %%    loss = %f" % (accuracy * 100, loss))
fig=plt.gcf()
plt.show()
print(f"policy val acc = {score[3]}")
print(f"policy val loss = {score[1]}")
print(f"value val acc = {score[4]}")
print(f"value val loss = {score[2]}")


t = time.localtime()
folder_name = f"{t.tm_mon:0>2}_{t.tm_mday:0>2}_{t.tm_hour:0>2}{t.tm_min:0>2}{t.tm_sec:0>2}"
new_folder_dir=os.path.join(root_path,'model_record',folder_name)
os.makedirs(new_folder_dir, exist_ok=True)


def convert_seconds(seconds):
    seconds=int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:0>3}h {minutes:0>2}m {seconds:0>2}s"

fig.savefig(os.path.join(new_folder_dir,'history.png'))
model.save(os.path.join(new_folder_dir,'model.h5'))

with open(os.path.join(new_folder_dir,'result.txt'), 'w') as f:
    f.write(f"load model = {is_load_model}\n")
    f.write(f"load model dir = {load_model_dir}\n")
    f.write(f"epochs = {epochs}\n")
    f.write(f"batch_size = {batch_size}\n")
    f.write(f"sample_size = {sample_size}\n")
    f.write(f"take_last_n_move = {take_last_n_move}\n")
    f.write(f"input_shape = {input_shape}\n")
    f.write(f"last_n_feature = {last_n_feature}\n")
    f.write(f"init lr = {init_lr}\n")
    f.write(f"l2 const = {l2_const}\n")
    f.write(f"policy val acc = {score[3]}\n")
    f.write(f"policy val loss = {score[1]}\n")
    f.write(f"value val acc = {score[4]}\n")
    f.write(f"value val loss = {score[2]}\n")
    f.write(f"precess time = {convert_seconds(precess_time)}\n")

python_file_name="manual_data_training.ipynb"
with open(python_file_name) as f:
    code = f.read()
    f.close()
json_code = json.loads(code)
model_structure=json_code['cells'][4]
with open(os.path.join(new_folder_dir,"model_structure.txt"), mode="w") as f:
    f.write("".join(model_structure['source']))
    f.close()