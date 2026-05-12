# 1. 以前作成した自作モジュールをインポート
import dataset1
import pandas as pd
from sklearn.linear_model import LinearRegression

# 2. dataset1.py の中の関数を使ってデータを読み込む
# 以前作った prepare_dataset 関数を呼び出します
df = dataset1.prepare_dataset()

# 3. 読み込んだデータの内容を確認（正しくインポートできたかチェック）
print("--- データセットの冒頭5行 ---")
print(df.head())
print(f"データセットの形: {df.shape}\n")

# 3. 演習1.8：線形回帰モデルを用意する
model = LinearRegression()

print("--- 演習1.8：モデルの準備が完了しました ---")
print(model)

# --- 演習1.9：フィッティング ---

# 1. データの準備
# 観測点をX（特徴量）、観測値をy（ターゲット）として取り出す
X = df[['観測点']].values
y = df['観測値'].values

# 2. データを学習用(8割)とテスト用(2割)に分割する
X_train = X[:16]
X_test  = X[16:]
y_train = y[:16]
y_test  = y[16:]

# 3. モデルを学習（フィッティング）させる
model.fit(X_train, y_train)

print("\n--- 演習1.9：フィッティングが完了しました ---")
# 学習後のパラメータ（傾きと切片）をチラ見してみる
print(f"学習後の傾き(a): {model.coef_[0]}")
print(f"学習後の切片(b): {model.intercept_}")

# --- 演習1.10：テスト用データセットに対する予測 ---

# 1. テストデータ（残り2割）の入力を使って予測を実行
y_pred = model.predict(X_test)

# 2. 予測結果を出力
print("\n--- 演習1.10：テストデータに対する予測結果 ---")
print(y_pred)

# 3. 比較のために実際の観測値（答え）も表示してみる
print("\n--- 参考：実際の観測値（答え） ---")
print(y_test)

import matplotlib.pyplot as plt
import numpy as np

# --- 演習1.11：モデルの可視化 ---

# 1. グラフ用のデータ作成
# 定義域 -1 <= x <= 1 を細かく分割したデータを作成（滑らかな線を描くため）
x_line = np.linspace(-1, 1, 100).reshape(-1, 1)

# 真の関数（dataset1で定義した sin(x * pi) * 10 と同じ式）
y_true = np.sin(x_line * np.pi) * 10

# AIモデルによる予測値（直線）
y_model = model.predict(x_line)

# 2. プロットの作成
plt.figure(figsize=(10, 6))

# 演習1.3：観測値（実際のデータ点）を散布図で描画
plt.scatter(df['観測点'], df['観測値'], color='blue', alpha=0.5, label='Observations')

# 演習1.1：真の関数を点線で描画
plt.plot(x_line, y_true, color='green', linestyle='--', label='True Function (sin)')

# 演習1.9：構築したモデル（直線）を実線で描画
plt.plot(x_line, y_model, color='red', linewidth=2, label='Linear Regression Model')

# 3. グラフの設定と保存
plt.title('Model Visualization (Linear Regression)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend() # 凡例を表示
plt.grid(True)

# 画像として保存
plt.savefig('ex1.10.png')
print("\n--- 演習1.11：グラフを ex1.10.png として保存しました ---")

# 画面に表示
plt.show()