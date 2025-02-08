# 矢量化 Vectorization

</br>

<p>矢量化就是一项让你的代码变得更高效的艺术。在深度学习的实际应用中，可能会遇到大量的训练数据，因为深度学习算法在这个情况下表现更好，所以代码的运行速度非常重要</p>

![QianJianTec1738987067310](https://github.com/user-attachments/assets/5c0fe87a-6078-44b8-95d9-46eff20ecc41)

![QianJianTec1738987060280](https://github.com/user-attachments/assets/aceae586-9cf2-4bc4-a014-1f03d9fffea6)

```
Non-vectorized
z = 0
for i in range(n_x):
    z += w[i] * x[i]
z += b
```

```
Vectorized: much faster
z = np.dot(w, x) + b
//w^T x
```

<p>可能听过这样的话，大规模的深度学习使用了 GPU 和图像处理单元实现，但是在 Jupyter notebook 上面实现的话，这里只有 CPU。CPU 和 GPU 都有并行化的指令，有时叫做 SIMD 指令，这代表了一个单独指令多维数组。基础意义在于如果使用了 built-in 函数，像 np.dot 或者并不要求你实现循环的函数，将使得 python 中的 numpy 充分利用并行化去更快计算，在 GPU 和 CPU 上面计算，GPU 被标记更快擅长 SIMD 计算但是 CPU 事实上也不是太差</p>


























