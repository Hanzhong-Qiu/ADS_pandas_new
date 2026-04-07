"""
步骤 4：VAR模型滞后阶数选择

用信息准则（AIC / BIC）从数据中客观地选出最优滞后天数，
供步骤5（Granger检验）和步骤6（VAR建模）使用。

前置条件：运行过 step2_adf.py，生成了 figures/step2_differenced_data.csv
依赖：pip install pandas numpy matplotlib statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import os

os.makedirs('figures', exist_ok=True)

# =============================================
# 加载差分数据
# =============================================
df = pd.read_csv('/home/qqq/ADS_pandas_new/graphs/modelling/step2_differenced_data.csv', index_col='date', parse_dates=True)
print(f"差分数据: {len(df)} 天")

# 核心双变量：情感均值 + 政策严格度
# 先用这两个做主分析，结果最清晰
core_vars = ['sentiment_mean', 'StringencyIndex_Average']
df_core = df[core_vars].dropna()
print(f"用于VAR的变量: {core_vars}")
print(f"样本量: {len(df_core)}")

# =============================================
# 逐滞后拟合VAR，记录AIC和BIC
# =============================================
MAX_LAG = 21  # 最多测试21天

model = VAR(df_core)

# 方法一：用statsmodels内置的select_order（它会打印一张完整的表）
print("\n" + "=" * 60)
print("VAR滞后阶数选择")
print("=" * 60)

selection = model.select_order(maxlags=MAX_LAG)
print(selection.summary())

aic_best = selection.aic
bic_best = selection.bic

print(f"\nAIC 最优滞后: {aic_best}")
print(f"BIC 最优滞后: {bic_best}")

# 方法二：手动记录每个滞后的AIC/BIC，用来画图
aic_vals = []
bic_vals = []

for lag in range(1, MAX_LAG + 1):
    try:
        result = model.fit(lag)
        aic_vals.append(result.aic)
        bic_vals.append(result.bic)
    except Exception:
        aic_vals.append(np.nan)
        bic_vals.append(np.nan)

# =============================================
# 选择最终使用的滞后
# =============================================
# BIC更保守（倾向短滞后，防止过拟合），通常更可靠
# 但如果AIC和BIC差距很大，说明数据信号较弱
CHOSEN_LAG = bic_best

print(f"\n决策:")
if aic_best == bic_best:
    print(f"  AIC和BIC一致，选择 lag = {CHOSEN_LAG} ✅")
else:
    print(f"  AIC ({aic_best}) 和 BIC ({bic_best}) 不一致。")
    print(f"  选择BIC的结果 lag = {bic_best}（更保守，防止过拟合）。")
    print(f"  后续可用 AIC 的 lag = {aic_best} 做稳健性检验。")
    CHOSEN_LAG = bic_best

# 保存选择结果供后续步骤读取
with open('figures/step4_chosen_lag.txt', 'w') as f:
    f.write(str(CHOSEN_LAG))
print(f"\n选定滞后 = {CHOSEN_LAG}，已保存到 figures/step4_chosen_lag.txt")

# =============================================
# 绘图
# =============================================
lags_range = list(range(1, MAX_LAG + 1))

fig, ax1 = plt.subplots(figsize=(10, 5))

# AIC 线
color_aic = '#2196F3'
ax1.plot(lags_range, aic_vals, 'o-', color=color_aic, label='AIC', markersize=5)
ax1.set_xlabel('Lag Order (days)', fontsize=11)
ax1.set_ylabel('AIC', color=color_aic, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_aic)

# BIC 线（共享x轴，独立y轴）
ax2 = ax1.twinx()
color_bic = '#E91E63'
ax2.plot(lags_range, bic_vals, 's-', color=color_bic, label='BIC', markersize=5)
ax2.set_ylabel('BIC', color=color_bic, fontsize=11)
ax2.tick_params(axis='y', labelcolor=color_bic)

# 标注最优点
ax1.axvline(x=aic_best, color=color_aic, linestyle=':', alpha=0.5)
ax1.axvline(x=bic_best, color=color_bic, linestyle=':', alpha=0.5)

# 标注选中的滞后
ax1.axvline(x=CHOSEN_LAG, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax1.annotate(f'Chosen = {CHOSEN_LAG}',
             xy=(CHOSEN_LAG, min(aic_vals)),
             xytext=(CHOSEN_LAG + 2, min(aic_vals)),
             fontsize=11, color='green', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green'))

# 图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax1.set_title('VAR Lag Order Selection: AIC vs BIC\n(Lower = Better)',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(lags_range)

plt.tight_layout()
plt.savefig('/home/qqq/ADS_pandas_new/graphs/modelling/step4_lag_selection.png', dpi=150)
plt.close()
print("✅ 图已保存: figures/step4_lag_selection.png")