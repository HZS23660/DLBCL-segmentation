import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = 'times new roman'
plt.rc('font', family='Times New Roman')
csv_dir = "/home/tju/Documents/HZS/Code_huang/model/PET_lymphoma_3D/WSS_l0.5_a0.2_60_labeled_128 64 64/vnet_att/loss_all.csv"

pdframe = pd.read_csv(csv_dir)

SS = pdframe["SS"].tolist()
DS = pdframe["DS"].tolist()
WS = pdframe["WS"].tolist()
all_cosloss = pdframe["all_cosloss"].tolist()
dice = pdframe["dice"].tolist()
ce = pdframe["ce"].tolist()
Sup = pdframe["Sup"].tolist()

total = 5400
inter = 30 #540
n = len(SS) // inter

SS = SS[:total:inter]
DS = DS[:total:inter]
WS = WS[:total:inter]
all_cosloss = all_cosloss[:total:inter]
dice = dice[:total:inter]
ce = ce[:total:inter]
Sup = Sup[:total:inter]

x = np.arange(0, len(SS))
# print(x)

plt.figure(dpi=300)
plt.plot(x, SS, color='r', linewidth=0.5, linestyle='--', label='SS')
plt.plot(x, WS, color='b', linewidth=0.5, linestyle='--', label='WS')
plt.plot(x, DS, color='g', linewidth=0.5, linestyle='--', label='DS')
plt.plot(x, all_cosloss, color='k', linewidth=0.5, linestyle='--', label='all_cosloss')
plt.plot(x, ce, color='deepskyblue', linewidth=0.5, linestyle='--', label='CE_loss')
plt.plot(x, dice, color='brown', linewidth=0.5, linestyle='--', label='DICE_loss')
plt.plot(x, Sup, color='lightpink', linewidth=0.5, linestyle='--', label='Sup_loss')
plt.legend(loc="upper right")

plt.show()
