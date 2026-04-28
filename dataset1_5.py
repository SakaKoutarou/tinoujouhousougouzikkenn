import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. TSVファイルからデータを読み込む（★ここが指示のポイント）
df_read = pd.read_csv('dataset.tsv', sep='\t')

# 2. モデルの学習準備
# sklearn用に形状を整える (20,) -> (20, 1)
X = df_read[['観測点']].values
y = df_read['観測値'].values

# 3. 線形回帰モデルの学習
model = LinearRegression()
model.fit(X, y)

# 4. 予測
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
y_pred = model.predict(x_test)

# 5. グラフ描画
plt.figure(figsize=(8, 5))

# 指示にはありませんが、比較用に真の関数も描いておくと丁寧です
def true_function(x): return np.sin(np.pi * x * 0.8) * 10
plt.plot(x_test, true_function(x_test), label='True Function', color='gray', linestyle='--', alpha=0.5)

# 読み込んだデータのプロット
plt.scatter(df_read['観測点'], df_read['観測値'], label='Observed Data (from TSV)', color='red', marker='x')

# 予測直線のプロット
plt.plot(x_test, y_pred, label='Linear Regression', color='blue', linewidth=2)

plt.title('Exercise 1.5: Linear Regression Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.savefig('ex1.5.png')
plt.show()