</br>
    
# 深度L层神经网络

</br>

![e68b5be7-3ab4-472d-8223-20f310894e5a](https://github.com/user-attachments/assets/f7961837-8f56-4025-8027-67175c1f6c93)

<p>logistic regression</p>

![d1401aa8-6a63-4722-93fe-593d8c9aa09a](https://github.com/user-attachments/assets/b4759ab7-be65-4399-9d34-e61d64e2c946)

<p>1 hidden layer</p>

<p>来看看更深的</p>

![e2f9d210-ce19-4359-bb16-f8f0aa3e42ab](https://github.com/user-attachments/assets/3708979c-8723-4fad-a8f8-949a313d3b91)

![47fc540a-5d55-4734-b5dd-949d8015fa12](https://github.com/user-attachments/assets/678ca35e-dcff-4996-9a9a-855baefae2e3)

<p>对于任何特定的问题来说，可能很难事先得知你需要多深的网络，所以一般会先尝试逻辑回归，然后再尝试一个隐藏层，两个隐藏层。<b>可以把隐藏层的数量作为另一个超参数</b>，可以尝试很多不同的值，然后通过交叉验证或者开发集进行评估</p>

<p>来到喜闻乐见的专业术语：</p>

- L: 神经网络的层数

- n^[l] = 第 l 层上的单元数（n^[0] = n_x）

- 对于每一层 l，使用 a[l] 代表第 l 层中的激活函数（a[l] = g[l](z[l])）

- W[l] 表示 l 层计算中间值 z[l] 的权重

- a[L] = yhat

</br>

# 深度网络中的前向传播

</br>

![8a1d8ef2-0524-452d-a712-ff15734ea6ad](https://github.com/user-attachments/assets/cb0449a2-7bc3-40a3-8ebd-300010fb7363)

<p>和往常一样，先来看对于单独训练样本 x，如何实现前向传播</p>

<p>先计算</p>

![QianJianTec1740317014171](https://github.com/user-attachments/assets/01fe7b70-26a0-4b0e-a246-372bd1da77ae)

<p>x 这里也是 a[0]，然后计算着一层的激活函数</p>

![QianJianTec1740317261438](https://github.com/user-attachments/assets/44eb304d-053b-4898-8187-bb75b70aa1e1)

<p>得出第一个隐藏层的计算公式</p>

<p>关于第二个隐藏层</p>

![QianJianTec1740317602445](https://github.com/user-attachments/assets/3e80edaf-72bd-4c78-b035-0c2ff2ac9500)

![QianJianTec1740317684806](https://github.com/user-attachments/assets/f0eb63ee-d4ad-4798-87f6-33b114b879f7)

<p>后面几层以此类推直到输出层。完成对单一训练样本的前向传播通用公式</p>

<p>向量化只需把 x 换成 X，z 换成 Z，a 换成 A。（把他们堆叠起来变成矩阵）</p>

<p>这里也有一个循环：for l = 1...4。之前说过在使用神经网络时，要尽可能避免使用 for-loop。这里是唯一一处，我觉得除了for以外没有更好解决的地方</p>

</br>

# 正确处理矩阵尺寸

</br>

<p>W[l] : (n[l], n[l - 1])</p>

<p>b[l] : (n[l], 1)</p>

<p>如果正在实现反向传播，那么 dW 的维度与 W 相同</p>

<p>l 层的 a 与 z 维度相同为 (n[l], m)</p>






































































































