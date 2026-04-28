import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 演習1.1: 真の関数
def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10

# 演習1.2 & 1.3: データセットの作成と保存
def prepare_dataset(n_samples=20, seed=0, filename='dataset.tsv'):
    np.random.seed(seed)
    x_samples = np.random.uniform(-1, 1, n_samples)
    y_true = true_function(x_samples)
    
    # 指示通りのノイズ（分散2.0の正規分布を半分に）
    noise = np.random.normal(0, np.sqrt(2.0), n_samples) / 2
    y_observed = y_true + noise
    
    df = pd.DataFrame({
        '観測点': x_samples,
        '真値': y_true,
        '観測値': y_observed
    })
    
    # TSVとして保存
    df.to_csv(filename, sep='\t', index=False)
    return df

# 演習1.5: 学習と予測
def train_and_predict(df):
    X = df[['観測点']].values
    y = df['観測値'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 予測用のデータ作成
    x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
    y_pred = model.predict(x_test)
    
    return x_test, y_pred

# グラフ描画（整理用）
def plot_results(df, x_test, y_pred, filename='ex1.6.png'):
    plt.figure(figsize=(8, 5))
    
    # 真の関数
    x_line = np.linspace(-1, 1, 100)
    plt.plot(x_line, true_function(x_line), label='True Function', color='gray', linestyle='--', alpha=0.5)
    
    # 観測データ
    plt.scatter(df['観測点'], df['観測値'], label='Observed Data', color='red', marker='x')
    
    # 予測
    plt.plot(x_test, y_pred, label='Linear Regression', color='blue', linewidth=2)
    
    plt.title('Exercise 1.6: Refactored Module')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# --- 直接このファイルを実行した場合の動作 ---
if __name__ == "__main__":
    # 1. データ作成
    df = prepare_dataset()
    # 2. 学習と予測
    x_test, y_pred = train_and_predict(df)
    # 3. 可視化
    plot_results(df, x_test, y_pred)