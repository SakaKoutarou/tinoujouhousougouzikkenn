import numpy as np

# 行列の作成
a = np.array([[1,2,3], [4,5,6]])
print(a)
print(type(a))
print(a.shape)

#行の参照
print(a[0])

#列の参照
print(a[:,0])

#スライス指定も可能
print(a[0:2])

print(a[:,0:2])

# 「行列 + 1」は全要素に対する和を実行
print(a + 1)

# *演算子も同様。
print(a * 2)

#行列演算ではない！
print(a * a)

#転置行列
print(a.T)

#内積を求めるにはdot関数を使う
print(np.dot(a, a.T))

#逆行列
print(np.linalg.inv(np.dot(a, a.T)))

#ゼロ行列、1行列、対角行列
print(np.zeros((2,3)))

print(np.ones((2,3)))

print(np.eye(3))

#特定範囲内で幅を指定してサンプル点を用意。
#例えば、
# 「y=x**2」のグラフを描画したいとき、
#　定義域「-10〜10の範囲で0.1刻みでサンプル点を用意」みたいなときに便利。
print(np.arange(0, 1, 0.3))

#np.arangeで始点、刻み幅を省略すると0から指定個数の整数を用意。
print(np.arange(8))

#行列の形を変形できる。
print(np.reshape(np.arange(6),(2,3)))

#刻み幅はどうでも良いからサンプル数を指定したい場合に便利。
print(np.linspace(0,2,3))

print(np.linspace(0,2,4))

#行列を結合できる。
#縦方向に結合
a = np.array([[1,2,3], [4,5,6]])
b = np.array([[7,8,9], [10,11,12]])
print(np.r_[a, b])

#横方向に結合
print(np.c_[a, b])

import numpy as np

# (1) 5個の要素を持つ列ベクトルを作成せよ。値は全て1とする。
v_col = np.ones((5, 1))
print("(1) 列ベクトル:\n", v_col)

# (2) 2番目の要素を3.14に更新せよ（インデックスは0から）。
v_col[2, 0] = 3.14
print("\n(2) 更新後の列ベクトル:\n", v_col)

# (3) 複製し、転置により行ベクトルに変換せよ。
v_row = v_col.copy().T
print("\n(3) 転置した行ベクトル:\n", v_row)

# (4) 用意した列ベクトルと行ベクトルの内積を求めよ。
dot_product = np.dot(v_row, v_col)
print("\n(4) 内積の結果:", dot_product)

# (5) np.random.randを用いて、10個の要素を持つ列ベクトルを作成せよ。
v_rand = np.random.rand(10, 1)
print("\n(5) ランダムな列ベクトル:\n", v_rand)

# (6) 平均値10、標準偏差2の正規分布に基づく、2行5列の行列を作成せよ。
mat_a = np.random.normal(10, 2, (2, 5))
print("(6) 正規分布に基づく行列:\n", mat_a)

# (7) 6で作成した行列から、2列目の要素を抜き出せ。
col_2nd = mat_a[:, 1]
print("\n(7) 2列目の要素:", col_2nd)

# (8) 6で作成した行列から、3列目と4列目の要素を抜き出せ。
cols_3rd_4th = mat_a[:, 2:4]
print("\n(8) 3列目と4列目の要素:\n", cols_3rd_4th)

# (9) np.random.randで5行2列の行列を用意し、6で用意した行列との積を求めよ。
mat_b = np.random.rand(5, 2)
matrix_product = np.dot(mat_a, mat_b)
print("\n(9) 行列積の結果:\n", matrix_product)