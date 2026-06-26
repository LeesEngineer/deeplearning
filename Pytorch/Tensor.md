</p>转换类型：</p>

```
A = X.numpy()
B = torch.tensor(A)
```

<p>降维：可以指定标量沿哪一个轴来通过求和降低维度，指定哪个轴哪个轴在输出中消失。</p>

```
A_sum_axis0 = A.sum(axis = 0)
A.sum(axis = [0, 1])
```

<p>还有非降维求和：</p>

```
sum_A = A.sum(axis = 1, keepdims = True)
```

<p>点积：</p>

`torch.dot(x, y)`

<p>以及</p>

```
torch.mv(A, x)
torch.mm(A, B)
```

<p>范数：</p>

```
u = torch.tensor([3.0, -4.0])
torch.norm(u)
# tensor(5.)
```














































































































































































































