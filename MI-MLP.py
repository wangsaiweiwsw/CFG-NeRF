# ====================== 修改 MipBlock 类 ======================
@gin.configurable
class MipBlock(nn.Module):
    """Stage 3: 多尺度特征融合"""
    features: int
    w0: float = 30.0
    num_scales: int = 4
    scale_factors: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)  # 合适的尺度范围
    
    def setup(self):
        # 添加配置验证
        print("生效尺度数量:", self.num_scales)
        print("生效尺度因子:", self.scale_factors)

        
        # 更小的初始权重范围
        kernel_init = nn.initializers.normal(stddev=0.01)
        bias_init = nn.initializers.zeros
        
        # 创建临时列表用于构建块
        scale_blocks_list = []

        for factor in self.scale_factors[:self.num_scales]:
            layers = [
                # 第一层：特征转换
                nn.Dense(self.features, kernel_init=kernel_init, bias_init=bias_init),
                nn.LayerNorm(),  # 归一化层
                
                # 带尺度系数的正弦激活
                lambda x, f=factor: jnp.sin(self.w0 * f * x),  # 闭包捕获
                
                # 第二层：特征精炼
                nn.Dense(self.features, kernel_init=kernel_init, bias_init=bias_init),
                nn.LayerNorm(),  # 归一化层
                nn.gelu  # GELU激活
            ]
            scale_blocks_list.append(nn.Sequential(layers))
        
        # 将列表转换为元组
        self.scale_blocks = tuple(scale_blocks_list)
        
        # 融合权重初始化为均匀分布
        self.fusion_weights = self.param(
            'fusion_weights', 
            nn.initializers.constant(1.0/self.num_scales),
            (self.num_scales,)
        )
    
    def __call__(self, x):
        scale_outputs = []
        for block in self.scale_blocks:
            scaled_x = block(x)
            scale_outputs.append(scaled_x)
        
        # 计算权重保证稳定性
        weights = jax.nn.softmax(self.fusion_weights)
        
        # 安全的加权和计算
        fused_output = jnp.zeros_like(scale_outputs[0])
        for i in range(len(scale_outputs)):
            fused_output += weights[i] * scale_outputs[i]
        
        return fused_output
   

# 调用
# ======== 修改阶段1的占位代码 ========
    if self.use_mip_mlp:
        # 准备 MipBlock 输入：拼接位置和特征
        mip_input = jnp.concatenate([means, x], axis=-1)
        
        # 调用 MipBlock（阶段2实现核心功能）
        mip_output = self.mip_block(mip_input)
        
        # 残差连接（实际生效）
        x = x + mip_output  # 移除之前的0*占位
        
        # 添加监控点（验证特征范围）
        # wsw添加 - 用于验证阶段2激活函数效果
        self.sow('intermediates', 'mip_pre_act', mip_output)