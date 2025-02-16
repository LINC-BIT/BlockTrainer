# from dianzixuebao.scaling_law.vit_b_16.img_cls.fid.main import main
# main()



import matplotlib.pyplot as plt
from vis.util import *
import pandas as pd

set_figure_settings(fig_wh_ratio=3, font_size=36, font_family=None)

# data = pd.read_csv('dianzixuebao/scaling_law/vit_b_16/img_cls/synsign.csv')
data = pd.read_csv('dianzixuebao/scaling_law/vit_b_16/img_cls/imagenet.csv')

X, Y = data['batch_size'], data['distance']

plt.plot(X, Y, marker='o', color=BLACK, markerfacecolor='white', 
         markeredgecolor='black', linewidth=4,
         markersize=14, markeredgewidth=4)
plt.xlabel('Batch size')
plt.ylabel('Calculated\ndiscrepancy')
plt.xlim(-20, max(X) + 20)
plt.plot([-20, max(X) + 20], [270] * 2, color=BLUE, linestyle='--', linewidth=4) # 163 for synsign
plt.tight_layout()
plt.grid()
# plt.savefig('dianzixuebao/scaling_law/vit_b_16/img_cls/synsign.png', dpi=300)
plt.savefig('dianzixuebao/scaling_law/vit_b_16/img_cls/imagenet.png', dpi=300)
plt.clf()
