# フォント設定
import matplotlib
import matplotlib.font_manager as font_manager
font_path = '/Library/Fonts/Arial Unicode.ttf'
font_prop = font_manager.FontProperties(fname = font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

import numpy as np
import matplotlib.pyplot as plt

# 1. 真の関数を定義 
def true_function(x):
    """
    >>> true_function(0)
    0.0
    """
    return np.sin(np.pi * x * 0.8) * 10

# 2. ユニットテスト
if __name__ == "__main__":
    import doctest
    doctest.testmod() # x=0のときy=0であることをテストします

    # 3. グラフの描画
    # 定義域 -1 <= x <= 1 を 100分割
    x_values = np.linspace(-1, 1, 100)
    y_values = true_function(x_values)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label='sin(pi * x * 0.8) * 10')
    
    # グラフの装飾
    plt.title('Exercise 1.1: True Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend() 

    # 4. 画像として保存
    plt.savefig('ex1.1.png')
    print("グラフを ex1.1.png として保存しました。")
    
    plt.show()