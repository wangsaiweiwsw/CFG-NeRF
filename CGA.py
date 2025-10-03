import flax.linen as nn
import jax.numpy as jnp

# SpatialAttention模块 (适配2D数据)
class SpatialAttention(nn.Module):
    def setup(self):
        self.sa = nn.Dense(features=1, use_bias=True)  # Dense层代替2D卷积

    def __call__(self, x):
        # 通道维度归约
        x_avg = jnp.mean(x, axis=-1, keepdims=True)  # 均值
        x_max = jnp.max(x, axis=-1, keepdims=True)  # 最大值
        # 拼接两个统计量
        x2 = jnp.concatenate([x_avg, x_max], axis=-1)
        # 空间注意力
        sattn = self.sa(x2)
        return sattn


# ChannelAttention模块 (适配2D数据)
class ChannelAttention(nn.Module):
    dim: int
    reduction: int = 8

    def setup(self):
        reduced_dim = max(1, self.dim // self.reduction)  # 确保降维后的维度至少为1
        self.fc1 = nn.Dense(features=reduced_dim)  # 降维
        self.relu = nn.relu
        self.fc2 = nn.Dense(features=self.dim)  # 恢复维度

    def __call__(self, x):
        # 全局池化: 对N维度求平均值，保留最后的通道维度
        x_gap = jnp.mean(x, axis=1, keepdims=True)
        x_gap = self.fc1(x_gap)  # 降维
        x_gap = self.relu(x_gap)
        cattn = self.fc2(x_gap)  # 恢复维度
        return cattn


# PixelAttention模块 (适配2D数据)
class PixelAttention(nn.Module):
    dim: int

    def setup(self):
        self.pa = nn.Dense(features=self.dim)  # Dense层代替卷积
        self.sigmoid = nn.sigmoid

    def __call__(self, x, pattn1):
        # 拼接输入和pattn1 (通道维度)
        x2 = jnp.concatenate([x, pattn1], axis=-1)
        # 像素注意力
        pattn2 = self.pa(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


# CGAFusion模块 (适配2D数据)
class CGAFusion(nn.Module):
    dim: int
    reduction: int = 8

    def setup(self):
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(self.dim, self.reduction)
        self.pa = PixelAttention(self.dim)
        self.fc = nn.Dense(features=self.dim, use_bias=True)  # Dense层代替1x1卷积

    def __call__(self, x, y):
        # 输入x和y的形状为 (B, N, C)，表示批次、点数、通道数
        initial = x + y
        cattn = self.ca(initial)  # 通道注意力 (输出形状: (B, 1, C))
        sattn = self.sa(initial)  # 空间注意力 (输出形状: (B, N, 1))
        
        # 调整形状以匹配 x 和 y
        sattn = jnp.broadcast_to(sattn, x.shape)  # 广播空间注意力到 (B, N, C)
        cattn = jnp.broadcast_to(cattn, x.shape)  # 广播通道注意力到 (B, N, C)
        
        pattn1 = sattn + cattn
        pattn2 = self.pa(initial, pattn1)  # 像素注意力 (输出形状: (B, N, C))

        # 加权融合
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.fc(result)  # 使用全连接层调整最终通道
        return result
