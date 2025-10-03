# ▼▼▼ 添加特征金字塔融合 ▼▼▼
          if self.use_feature_pyramid and intermediate_features:
              # 获取可用的层索引
              available_layers = sorted(intermediate_features.keys())
              
              # 根据可用层数动态选择特征
              if len(available_layers) >= 3:
                  # 如果有3层或更多，使用前中后三层
                  shallow_idx = available_layers[0]
                  mid_idx = available_layers[len(available_layers)//2]
                  deep_idx = available_layers[-1]
              elif len(available_layers) == 2:
                  # 如果有2层，使用第一层和最后一层
                  shallow_idx = available_layers[0]
                  mid_idx = available_layers[0]  # 重复使用第一层
                  deep_idx = available_layers[1]
              else:
                  # 如果只有1层，重复使用该层
                  shallow_idx = available_layers[0]
                  mid_idx = available_layers[0]
                  deep_idx = available_layers[0]
              
              # 从不同深度获取特征
              shallow_feat = intermediate_features[shallow_idx]
              mid_feat = intermediate_features[mid_idx]
              deep_feat = bottleneck  # 瓶颈层作为深层特征
              
              # 2. 上采样并融合特征
              # 上采样中间层特征到深层特征的分辨率
              mid_feat_up = jax.image.resize(
                  mid_feat, 
                  deep_feat.shape, 
                  method="bilinear"
              )
              
              # 上采样浅层特征到深层特征的分辨率
              shallow_feat_up = jax.image.resize(
                  shallow_feat, 
                  deep_feat.shape, 
                  method="bilinear"
              )
              
              # 3. 融合特征（加权和）
              fused_feat = (
                  self.pyramid_fusion_weights[0] * deep_feat +
                  self.pyramid_fusion_weights[1] * mid_feat_up +
                  self.pyramid_fusion_weights[2] * shallow_feat_up
              )
              
              # 4. 替换原始瓶颈特征
              bottleneck = fused_feat
          # ▲▲▲ 融合结束 ▲▲▲