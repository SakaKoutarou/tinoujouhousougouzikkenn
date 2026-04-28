import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 演習1.1の関数
def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10

# シードを固定
np.random.seed(0)

# --- 演習1.2の内容 ---
n_samples = 20
x_samples = np.random.uniform(-1, 1, n_samples)
y_true = true_function(x_samples)

df = pd.DataFrame({
    '観測点': x_samples,
    '真値': y_true
})

# --- 演習1.3：ノイズの付与 ---
# 平均0.0, 分散2.0（＝標準偏差は√2.0）の正規分布ノイズを生成
# それをさらに「半分」にする
noise = np.random.normal(0, np.sqrt(2.0), n_samples) / 2

# 「観測値」列として追加（真値 + ノイズ）
df['観測値'] = df['真値'] + noise

# 中身を確認
print(df.head())

# --- グラフの描画 ---
x_line = np.linspace(-1, 1, 100)
y_line = true_function(x_line)

plt.figure(figsize=(8, 5))

# 1. 真の関数の線
plt.plot(x_line, y_line, label='True Function', color='gray', linestyle='--', alpha=0.7)

# 2. 演習1.2の「真値」（点）
plt.scatter(df['観測点'], df['真値'], label='True Samples', color='blue', marker='o', s=30)

# 3. 演習1.3の「観測値」（ノイズあり・点）
plt.scatter(df['観測点'], df['観測値'], label='Observed Samples (with noise)', color='red', marker='x', s=50)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Exercise 1.3: Adding Noise to Samples')
plt.legend()
plt.grid(True)

# 画像を保存
plt.savefig('ex1.3.png')
plt.show()