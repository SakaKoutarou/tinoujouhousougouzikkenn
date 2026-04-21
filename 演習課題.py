# データを用意
import numpy as np
X = np.array([[0.0], [2.0], [3.9], [4.0]])
Y = np.array([4.0, 0.0, 3.0, 2.0])

# 1次元データを拡張
# ・1次元目（頭の「1」）は、0乗した値（バイアス項）。
# ・2次元目は、元のデータ。
# ・3次元目は、2乗（degree=2）した値。
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X2 = poly.fit_transform(X)
print(X2)

# 学習機の用意、学習
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X2, Y)

# 描画
#%matplotlib inline
import matplotlib.pyplot as plt

samples_x = np.arange(0, 4.1, 0.1)
samples_x2 = poly.fit_transform(samples_x.reshape(len(samples_x), 1))
samples_y = clf.predict(samples_x2)
plt.plot(samples_x, samples_y, label='alpha = 1.0')
plt.scatter(X, Y, label='data set')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# 1. データの用意（資料 In [1] より）
X = np.array([[0.0], [2.0], [3.9], [4.0]])
Y = np.array([4.0, 0.0, 3.0, 2.0])

# 2. 次数の設定（資料に合わせて degree=2 にします）
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 描画用の細かいX軸データ（資料 In [2] より）
samples_x = np.arange(0, 4.1, 0.1)
samples_x_poly = poly.fit_transform(samples_x.reshape(-1, 1))

# 3. グラフの準備
plt.figure(figsize=(8, 5))
alphas = [0, 0.1, 0.5, 1.0, 10.0]

# 4. 5パターンのalphaでループ処理
for a in alphas:
    # 各alphaで学習
    clf = Ridge(alpha=a)
    clf.fit(X_poly, Y)
    
    # 予測
    samples_y = clf.predict(samples_x_poly)
    
    # 描画（ラベルに alpha の値を設定）
    plt.plot(samples_x, samples_y, label=f'alpha = {a}')

# 5. 仕上げ（資料の Out [3] と同じ見た目にする）
plt.scatter(X, Y, label='data set', zorder=5) # 元のデータ点
plt.legend() # 凡例を表示
plt.show()   # 画面に表示