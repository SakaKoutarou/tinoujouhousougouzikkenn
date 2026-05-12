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