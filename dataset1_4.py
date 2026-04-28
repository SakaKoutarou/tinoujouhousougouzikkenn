import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyDataset:
    def __init__(self, seed=0):
        np.random.seed(seed)
        self.df = None

    def true_function(self, x):
        return np.sin(np.pi * x * 0.8) * 10

    def create(self, n_samples=20):
        x_samples = np.random.uniform(-1, 1, n_samples)
        y_true = self.true_function(x_samples)
        noise = np.random.normal(0, np.sqrt(2.0), n_samples) / 2
        y_observed = y_true + noise

        self.df = pd.DataFrame({
            '観測点': x_samples,
            '真値': y_true,
            '観測値': y_observed
        })
        return self.df

    # ★ここを追加：TSVファイルとして保存する機能
    def save_tsv(self, filename='dataset.tsv'):
        if self.df is not None:
            # sep='\t' でタブ区切りを指定します
            self.df.to_csv(filename, sep='\t', index=False)
            print(f"{filename} を保存しました。")

    def plot(self):
        if self.df is None: return
        x_line = np.linspace(-1, 1, 100)
        y_line = self.true_function(x_line)
        plt.figure(figsize=(8, 5))
        plt.plot(x_line, y_line, label='True Function', color='gray', linestyle='--', alpha=0.7)
        plt.scatter(self.df['観測点'], self.df['真値'], label='True Samples', color='blue', s=30)
        plt.scatter(self.df['観測点'], self.df['観測値'], label='Observed Samples', color='red', marker='x', s=50)
        plt.legend()
        plt.grid(True)
        plt.savefig('ex1.4.png')
        plt.show()

if __name__ == "__main__":
    dataset = MyDataset()
    dataset.create(20)
    dataset.save_tsv('dataset.tsv') # TSV出力
    dataset.plot()