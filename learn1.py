# 1. 以前作成した自作モジュールをインポート
import dataset1
import pandas as pd

# 2. dataset1.py の中の関数を使ってデータを読み込む
# 以前作った prepare_dataset 関数を呼び出します
df = dataset1.prepare_dataset()

# 3. 読み込んだデータの内容を確認（正しくインポートできたかチェック）
print(df.head())

print(f"\nデータセットの形: {df.shape}")