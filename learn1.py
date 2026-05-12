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