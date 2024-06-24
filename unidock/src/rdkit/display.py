import pandas as pd
import matplotlib.pyplot as plt

# 读取优化日志文件
data = pd.read_csv("optimization_log.csv")

# 绘制每次迭代的torsion角度和能量
fig, ax1 = plt.subplots()

ax1.set_xlabel('Iteration')
ax1.set_ylabel('Torsion Angles (rad)')
ax1.plot(data['Iteration'], data['Angle1'], label='Angle 1', color='tab:blue')
ax1.plot(data['Iteration'], data['Angle2'], label='Angle 2', color='tab:orange')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel('Energy')
ax2.plot(data['Iteration'], data['Energy'], label='Energy', color='tab:red')
ax2.tick_params(axis='y')

fig.tight_layout()
plt.title('Optimization Progress')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
