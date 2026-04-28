import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # pandasをインポート

# 演習1.1の関数
def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10

# シード値を指定（指示通りデフォルト0）
np.random.seed(0)

# 1. 観測点をランダムに20個用意
n_samples = 20
x_samples = np.random.uniform(-1, 1, n_samples)

# 2. 対応する真の値を求める
y_true = true_function(x_samples)

# 3. pandas.DataFrame型（20行2列）に設定
# 列名を「観測点」「真値」にする
df = pd.DataFrame({
    '観測点': x_samples,
    '真値': y_true
})

# 確認用：中身を表示
print(df)

# 4. グラフの描画（演習1.1の線グラフ ＋ サンプル集合をプロット）
x_line = np.linspace(-1, 1, 100)
y_line = true_function(x_line)

plt.figure(figsize=(8, 5))
plt.plot(x_line, y_line, label='True Function', color='gray', linestyle='--') # 1.1の線
plt.scatter(df['観測点'], df['真値'], label='Sample Dataset', color='blue', s=30) # 1.2の点

plt.xlabel('x')
plt.ylabel('y')
plt.title('Exercise 1.2')
plt.legend()
plt.grid(True)

# 5. 画像を保存
plt.savefig('ex1.2.png')
plt.show()