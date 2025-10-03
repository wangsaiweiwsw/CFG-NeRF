# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NeRF and its MLPs, with helper functions for construction and rendering."""

import functools
import time
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Tuple, Union

"""MLP内部添加增强代码"""
# wsw添加 
# from camp_zipnerf.internal.cga_fusion import CGAFusion  # 假设 cga_fusion.py 和 models.py 在同一目录下
# from camp_zipnerf.internal.augmentations import augment_pipeline_jax  # 假设 cga_fusion.py 和 models.py 在同一目录下

from absl import logging
from flax import linen as nn
import gin
from . import configs
from . import coord
from . import geopoly
from . import grid_utils
from . import image_utils
from . import math_utils
from . import ref_utils
from . import render
from . import stepfun
from . import utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections



gin.config.external_configurable(math_utils.safe_exp, module='math')
gin.config.external_configurable(math_utils.laplace_cdf, module='math')
gin.config.external_configurable(math_utils.scaled_softplus, module='math')
gin.config.external_configurable(math_utils.power_ladder, module='math')
gin.config.external_configurable(math_utils.inv_power_ladder, module='math')
gin.config.external_configurable(coord.contract, module='coord')


def random_split(rng):
  if rng is None:
    key = None
  else:
    key, rng = random.split(rng)
  return key, rng


@gin.configurable
class Model(nn.Module):
  """A mip-Nerf360 model containing all MLPs."""

  config: Any = None

  # 一个包含每个采样轮次的元组（mlp_idx，grid_idx，num_samples）的列表。
  # 这段代码默认使用 mip - NeRF 360 代码库中的设置，即三轮采样，
  # 使用一个“proposal”MLP 和一个“NeRF”MLP且无网格
  sampling_strategy: Tuple[Tuple[int, int, int], Ellipsis] = (
      (0, None, 64),
      (0, None, 64),
      (1, None, 32),
  )


  # 此模型使用的 MLP + 网格的特定参数。这些元组的长度也决定了将构建多少个 MLP/网格。
  # 用户必须确保 MLP/网格的数量与`sampling_strategy`中的配置匹配，否则代码无法运行
  mlp_params_per_level: Tuple[ml_collections.FrozenConfigDict, Ellipsis] = (
      {'disable_rgb': True},
      {'disable_rgb': False},
  )


  # 默认禁用网格。类型为包含ml_collections.FrozenConfigDict类型元素的元组s
  grid_params_per_level: Tuple[ml_collections.FrozenConfigDict, Ellipsis] = ()
  # 背景强度范围，包含两个浮点数，表示背景 RGB 的范围
  bg_intensity_range: Tuple[float, float] = (1.0, 1.0)
  # 退火斜率，值越高退火越快
  anneal_slope: float = 10
  # 如果为 True，则不跨层级反向传播
  stop_level_grad: bool = True
  # 如果为 True，则使用视图方向作为输入
  use_viewdirs: bool = True
  # 投射光线的形状（'cone'或'cylinder'）
  ray_shape: str = 'cone'
  # 如果为 True，则使用位置编码（PE）而不是积分位置编码（IPE），即禁用积分
  disable_integration: bool = False
  # 如果为 True，则对整个光线而不是样本进行抖动
  single_jitter: bool = True
  # GLO 向量长度，如果为 0 则禁用
  num_glo_features: int = 0
  # 训练图像的最大数量上限，用于 GLO 嵌入数量
  num_glo_embeddings: int = 10000
  # 是否学习曝光缩放（RawNeRF）
  learned_exposure_scaling: bool = False
  # 近边界退火速率，如果为 None 则有特殊处理，否则表示近边界的退火速度
  near_anneal_rate: Optional[float] = None
  # 近边界初始化值，范围在[0, 1]
  near_anneal_init: float = 0.95
  # 重采样填充，用于狄利克雷/alpha 直方图的“填充”值
  resample_padding: float = 0.0

  # 以下超参数控制 beta，它是用于从有向距离函数（SDF）到密度转换的尺度参数
  # （见https://arxiv.org/pdf/2106.12052.pdf中的方程2和3）
  scheduled_beta: bool = False
  # 每个采样层级的最终 beta 值
  final_betas: Tuple[float, Ellipsis] = (1.5e-2, 3.0e-3, 1.0e-3)
  # 用于调度 beta 的速率
  rate_beta: float = 0.75
  # 用于光线距离的曲线。可以只是一个函数，如@jnp.log，或者是（fn，fn_inv，**kwargs）的形式，如
  # (@math.power_ladder，@math.inv_power_ladder，{'p': -2，'premult': 10})
  raydist_fn: Union[Tuple[Callable[Ellipsis, Any], Ellipsis], Callable[Ellipsis, Any]] = None
  # 最大曝光值
  max_exposure: float = 1.0


  """
  这是模型类的__call__方法，它定义了模型的前向传播过程。
  该方法接受多个参数，包括随机数生成器、光线数据、训练进度分数、
  计算额外信息的标志等，并返回一个包含颜色（rgb）、距离和透明度（acc）等信息的列表。
  """
  @nn.compact
  def __call__(
      self,
      rng, # 随机数生成器，如果为 None 则输出是确定性的
      rays, # util.Rays类型，光线数据和元数据的树状结构
      train_frac, # 训练完成的比例，取值范围在[0, 1]的浮点数
      compute_extras, # 布尔值，如果为 True，则除了计算颜色之外还计算额外的量
      zero_glo=True,  # 布尔值，如果为 True，当使用 GLO（可能是某种全局光照相关的向量）时传入零向量
      percentiles = (5, 50, 95),  # 深度将为这些百分位数返回相应的值
      train = True, # 在训练时设置为 True
  ):
   
   # 根据mlp_params_per_level中的参数构建多个 MLP（多层感知机）
    mlps = [
        MLP(name=f'MLP_{i}', **params)
        for i, params in enumerate(self.mlp_params_per_level)
    ]
    
    # 根据grid_params_per_level中的参数构建多个网格（可能是用于位置编码之类的哈希编码网格）
    grids = [
        grid_utils.HashEncoding(name=f'grid_{i}', **params)
        for i, params in enumerate(self.grid_params_per_level)
    ]

    if self.num_glo_features > 0:
      if not zero_glo:
        # 如果num_glo_features大于0且zero_glo为False，则构建或获取每个输入光线相机的 GLO 向量。
        # 使用nn.Embed创建一个嵌入层，将cam_idx（光线的相机索引）映射到指定维度的 GLO 向量
        glo_vecs = nn.Embed(self.num_glo_embeddings, self.num_glo_features)
        cam_idx = rays.cam_idx[Ellipsis, 0]
        glo_vec = glo_vecs(cam_idx)
      else:
        # 如果zero_glo为True，则创建一个形状合适的全零向量作为 GLO 向量
        glo_vec = jnp.zeros(rays.origins.shape[:-1] + (self.num_glo_features,))
    else:
      # 如果num_glo_features为0，则GLO向量为 None
      glo_vec = None

    """学习曝光缩放相关部分"""
    # 默认为False
    if self.learned_exposure_scaling:
      # 如果启用了学习曝光缩放
      # 设置输出颜色的学习缩放因子相关操作
      # TODO(bmild): 修复此处对`num_glo_embeddings`的使用问题
      max_num_exposures = self.num_glo_embeddings
      # 使用jax.nn.initializers.zeros初始化函数，将学习缩放偏移初始化为0
      init_fn = jax.nn.initializers.zeros
      # 创建一个嵌入层，用于学习曝光缩放偏移。
      # 它将max_num_exposures个索引映射到维度为3的向量，初始化使用init_fn
      exposure_scaling_offsets = nn.Embed(
          max_num_exposures,
          features=3,
          embedding_init=init_fn,
          name='exposure_scaling_offsets',
      )


    """光线距离映射相关部分"""
    # 定义从归一化到度量光线距离的映射
     # 如果raydist_fn是一个元组
    if isinstance(self.raydist_fn, tuple):
      # 使用coord.construct_ray_warps函数构建光线扭曲相关的映射。
      # 这里使用部分应用（partial）传入fn和fn_inv以及相关参数
      fn, fn_inv, kwargs = self.raydist_fn
      _, s_to_t = coord.construct_ray_warps(
          functools.partial(fn, **kwargs),
          rays.near,
          rays.far,
          fn_inv=functools.partial(fn_inv, **kwargs),
      )
    else:
      # 如果raydist_fn不是元组，直接使用它构建光线扭曲相关的映射
      _, s_to_t = coord.construct_ray_warps(
          self.raydist_fn, rays.near, rays.far
      )

    exposure_values = rays.exposure_values

    # 初始化每条光线的（归一化）距离范围为[0, 1]，
    # 并将该单个区间的权重设置为1。这些距离和权重在后续采样层级过程中会不断更新。
    # `near_anneal_rate`可用于在训练开始时对近边界进行退火，
    # 例如值为0.1表示在训练的前10%对边界进行退火
    if self.near_anneal_rate is None:
      init_s_near = 0.0
    else:
      init_s_near = jnp.clip(
          1 - train_frac / self.near_anneal_rate, 0, self.near_anneal_init
      )
    init_s_far = 1.0
    sdist = jnp.concatenate(
        [
            jnp.full_like(rays.near, init_s_near),
            jnp.full_like(rays.far, init_s_far),
        ],
        axis=-1,
    )
    # 初始化权重为与rays.near形状相同的全1向量
    weights = jnp.ones_like(rays.near)

    



    """
    这段代码主要实现了在每个采样层级中对光线的一系列处理，
    包括 MLP 处理、法线矫正、权重计算、背景颜色处理、曝光逻辑处理、
    光线渲染以及相关结果的存储和可视化信息的收集，这些步骤共同构成了整个模型的光线渲染流程。
    """      
    # 用于存储光线相关历史信息的列表，初始化为空
    ray_history = []
    # 用于存储渲染结果的列表，初始化为空
    renderings = []
    # 长度为mlps列表长度的布尔值列表，初始化为全False，表示每个MLP是否被使用过，初始都为未使用
    mlp_was_used = [False] * len(mlps)
    # 长度为grids列表长度的布尔值列表，初始化为全False，表示每个网格是否被使用过，初始都为未使用
    grid_was_used = [False] * len(grids)
    
    
    for i_level, (i_mlp, i_grid, num_samples) in enumerate(
        self.sampling_strategy
    ):
      # 根据索引获取当前层级的 MLP
      mlp = mlps[i_mlp]
      # 这里假设mlps是一个包含多个MLP（多层感知机）的列表，
      # 通过当前采样层级对应的MLP索引i_mlp从列表中获取相应的MLP。
      # 这样可以在每个采样层级使用特定的MLP进行后续计算。

      # 将对应 MLP 的使用标记设为 True
      mlp_was_used[i_mlp] = True
      # mlp_was_used是一个布尔值列表，用于记录每个MLP是否在当前的采样过程中被使用过。
      # 通过将对应索引i_mlp位置的元素设为True，表示该MLP在当前层级被使用了。
      # 这有助于跟踪模型中各个MLP的使用情况。

      if i_grid is None:
        grid = None
        # 如果当前采样层级对应的网格索引i_grid为None，
        # 说明在这个层级没有网格相关的操作，将grid设为None。
      else:
        grid = grids[i_grid]
        grid_was_used[i_grid] = True
        # 如果i_grid不为None，从grids列表（假设grids是包含网格相关数据结构或对象的列表）
        # 中根据索引i_grid获取当前层级的网格。
        # 同时，将grid_was_used列表中对应索引i_grid位置的元素设为True，
        # 表示该网格在当前层级被使用了，与MLP使用标记的作用类似，用于跟踪网格的使用情况。


      # 根据训练迭代次数选择性地对权重进行退火处理。
      # 退火操作通常用于在训练过程中逐渐调整某些参数，这里是针对权重。
      if self.anneal_slope > 0:
        # 如果退火斜率（anneal_slope）大于0，则使用 Schlick's 偏差函数来计算退火因子。
        # Schlick's 偏差函数常用于计算机图形学等领域的一些参数调整场景，这里用于计算权重退火相关的值。
        bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
        anneal = bias(train_frac, self.anneal_slope)
        # 使用定义的偏差函数bias计算退火因子anneal。
        # 其中train_frac是训练进度分数（取值范围通常在[0, 1]），
        # self.anneal_slope是退火斜率，它控制着退火的速度或程度。
        # 随着训练的进行（train_frac的变化），
        # anneal的值会根据偏差函数和退火斜率而改变，从而影响权重的退火计算。

      else:
        anneal = 1.0
        # 如果退火斜率不大于0（即退火斜率为0或负数），则将退火因子设为1.0。
        # 这意味着不进行退火操作，权重保持不变，因为任何数乘以1都等于其本身。
        

      # 这是一种计算weights**anneal更稳定的方式。如果相邻区间的距离为零，则将其权重固定为0。
      logits_resample = jnp.where(
          sdist[Ellipsis, 1:] > sdist[Ellipsis, :-1],
          # jnp.where是条件判断函数。这里检查sdist中相邻区间的距离是否大于零。
          # 如果大于零，则计算anneal * math.safe_log(weights + self.resample_padding)。
          # anneal是之前计算的退火因子，math.safe_log用于安全地计算对数（避免对数函数定义域问题），
          # weights是权重，self.resample_padding是重采样的填充值。
          anneal * math_utils.safe_log(weights + self.resample_padding),
          -jnp.inf,
          # 如果相邻区间距离不大于零（为零），则将对数几率设为负无穷，
          # 因为这种情况下该区间权重应固定为0，在后续基于对数几率的采样中，负无穷对应的概率为0。
      )

      # 从每条光线当前的权重中抽取采样区间。
      # 使用random_split函数分割随机数生成器rng，
      # 得到新的随机数生成器key和更新后的rng，用于后续的采样操作。
      key, rng = random_split(rng)

      # stepfun.sample_intervals函数用于抽取采样区间。其参数含义如下：
      # key：新的随机数生成器，用于控制抽样的随机性。
      # sdist：之前的距离区间数据，在抽样过程中可能被更新。
      # logits_resample：用于重采样的对数几率，决定抽样的概率分布。
      # num_samples：要抽取的样本数量，决定了采样区间的数量。
      # single_jitter=self.single_jitter：布尔值，表示是否进行单抖动。若为True，
      # 可能对整个光线进行抖动，而不是对每个样本。
      # domain=(init_s_near, init_s_far)：距离范围元组，定义了抽样区间的有效范围，
      # 抽样区间会在这个范围内。
      sdist = stepfun.sample_intervals(
          key,
          sdist,
          logits_resample,
          num_samples,
          single_jitter=self.single_jitter,
          domain=(init_s_near, init_s_far),
      )

      # 如果通过采样传播梯度，优化通常会变为非线性。
      # 在采样过程中，如果梯度反向传播经过采样操作，可能会导致优化问题变得复杂，
      # 因为采样操作本身是具有随机性或离散性的，
      # 这会破坏梯度计算的连续性假设，使得优化过程不再是简单的线性优化问题。

      # 如果设置了self.stop_level_grad为True，
      # 使用jax.lax.stop_gradient函数停止sdist的梯度传播。
      # 这样做是为了避免在优化过程中由于梯度通过采样步骤反向传播而产生的非线性问题，
      # 使得优化过程更易于处理。
      # 默认为True
      if self.stop_level_grad:
        sdist = jax.lax.stop_gradient(sdist)

      # 将归一化距离sdist转换为度量距离tdist，通过调用s_to_t函数实现。
      # 这个转换可能是将之前在某个归一化空间中的距离值转换为实际物理意义或模型
      # 所需的度量空间中的距离值，以便后续计算。
      tdist = s_to_t(sdist)

      # 通过将距离区间（这里是tdist）转换为高斯分布来投射光线。
      # 调用render.cast_rays函数，传入距离tdist、光线原点rays.origins、
      # 光线方向rays.directions、光线半径rays.radii和光线形状self.ray_shape等参数。
      # diag=False可能表示不使用对角形式（具体取决于cast_rays函数的实现细节），
      # 此函数的作用是将光线相关的距离信息转换为高斯分布形式，为后续的计算（如光线渲染）做准备。
      gaussians = render.cast_rays(
          tdist,
          rays.origins,
          rays.directions,
          rays.radii,
          self.ray_shape,
          diag=False,
      )

      # 如果self.disable_integration为True，将高斯样本的协方差设置为0。
      # 这样做是为了禁用集成位置编码中的“集成”部分。这里将gaussians重新赋值，
      # 保持第一个元素不变（可能是均值相关信息），将第二个元素（可能是协方差相关信息）
      # 设置为与原协方差形状相同的全0数组。

      # 如果为 True，则使用位置编码（PE）而不是积分位置编码（IPE），即禁用积分
      if self.disable_integration:
        gaussians = (gaussians[0], jnp.zeros_like(gaussians[1]))

      
      
      """将我们的高斯分布数据传入 MLP（多层感知机）进行处理。"""

      # 再次使用 random_split 函数对随机数生成器 rng 进行分割，
      # 得到新的随机数生成器 key 和更新后的 rng。
      # 这可能是为了在 MLP 计算过程中需要新的随机数，
      # 比如在涉及随机初始化、随机采样或随机变换等操作时使用。
      key, rng = random_split(rng)
      curr_beta = None

      """
      在代码中，self.scheduled_beta为True时，表示采用了计划的beta值策略。
      self.final_betas是每个采样层级最终的beta值序列，self.sampling_strategy是采样策略，
      其中包含了每个采样轮次的信息（如mlp_idx、grid_idx、num_samples）。
      比较它们的长度是为了确保final_betas中的beta值数量与采样层级数量相匹配。
      每个采样层级都应该有一个对应的beta值来参与计算，如果数量不一致，
      就无法正确地为每个层级分配beta值，会导致计算错误。
      """
      if self.scheduled_beta:
        # 检查 final_betas 的长度是否与 sampling_strategy 的长度相等，
        # 并且 final_betas 中的每个 beta 值都大于 0。
        # 如果不满足条件，则抛出 ValueError 异常，提示计划 beta 应该在每个层级都有且为正值。
        if len(self.final_betas) != len(self.sampling_strategy) or (
            any([beta <= 0.0 for beta in self.final_betas])
        ):
          raise ValueError(
              '计划的beta值应该在每个层级都有且为正值。'
          )
        # 通过调用 get_scheduled_beta 函数，
        # 根据当前层级索引 i_level 和训练进度 train_frac 获取当前层级的 beta 值，
        # 并赋给 curr_beta。
        curr_beta = self.get_scheduled_beta(i_level, train_frac)


      # 使用 MLP（mlp）对高斯分布数据（gaussians）进行处理，并传入以下参数：
      # - key：新的随机数生成器，用于 MLP 内部可能的随机操作。
      # - viewdirs：如果使用视线方向（use_viewdirs 为 True），
        # 则传入 rays.viewdirs，否则为 None。视线方向信息可能用于调整 MLP 的输出，
        # 使其与光线的观察方向相关。

      # - imageplane：传入光线的图像平面信息 rays.imageplane，这可能用于与图像相关的计算或调整。
      # - glo_vec：全局光照向量（如果有的话），用于可能的光照相关计算。
      # - exposure：传入经过停止梯度操作的曝光值 jax.lax.stop_gradient(exposure_values)，
        # 这可能是为了避免曝光值对梯度计算产生不必要的影响。
      
      # - curr_beta：当前层级的 beta 值，如果启用了计划 beta。
      # - grid：当前层级的网格信息（如果有），用于可能的位置编码或其他与网格相关的计算。
      ray_results = mlp(
          key,
          gaussians,
          viewdirs=rays.viewdirs if self.use_viewdirs else None,
          imageplane=rays.imageplane,
          glo_vec=glo_vec,
          exposure=jax.lax.stop_gradient(exposure_values),
          curr_beta=curr_beta,
          grid=grid,
          rays=rays,
          tdist=tdist,
          train=train,
      )

      """
      计算矫正法线部分
      """

      #计算所有法线的“矫正”版本，对于背向相机的表面，将其法线的符号翻转，使其朝向相机。
      # （注意，翻转法线符号对 ref - nerf 使用的镜像方向没有影响）
      rectified = {}
      for key, val in ray_results.items():
        # 对于ray_results中的每个键值对，如果键以'normals'开头且值不为None，则进行以下操作：
        if key.startswith('normals') and val is not None:

          # 计算法线向量（val）与视线方向向量（rays.viewdirs）的点积。
          # 这里的Ellipsis是一种灵活的索引表示方式，用于处理多维数组。
          # 计算结果p用于判断法线是朝向相机还是背向相机。
          p = jnp.sum(val * rays.viewdirs[Ellipsis, None, :], axis=-1, keepdims=True)

          # 如果点积p大于0，表示法线背向相机，通过jnp.where函数将法线乘以 -1来翻转其方向。
          # 将矫正后的法线存储在rectified字典中，键为原键加上'_rectified'后缀。
          rectified[key + '_rectified'] = val * jnp.where(p > 0, -1, 1)
      
      # 将矫正后的法线信息更新到ray_results字典中。
      ray_results.update(rectified)



      """
      计算体积渲染权重部分
      """

      # 获取用于体积渲染（以及其他损失计算）的权重。
      # 调用render.compute_alpha_weights函数，传入ray_results中的密度信息
      # （ray_results['density']）、
      # 之前计算得到的度量距离tdist和光线方向rays.directions，计算得到权重weights。
      weights = render.compute_alpha_weights(
          ray_results['density'], tdist, rays.directions
      )
      # 为每条光线定义或采样背景颜色。
      if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
        # 如果背景强度范围的最小值和最大值相等，直接使用该值作为背景颜色。
        bg_rgbs = self.bg_intensity_range[0]
      elif rng is None:
         # 如果渲染是确定性的（即没有随机数生成器，rng为None），则使用背景强度范围的中点作为背景颜色。
        bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
      else:
        # 为每条光线从背景强度范围内采样RGB值作为背景颜色。
        # 使用random_split函数对随机数生成器rng进行分割，得到新的随机数生成器key和更新后的rng。
        key, rng = random_split(rng)

        # 使用random.uniform函数根据新的随机数生成器key，
        # 在指定的背景强度范围内（minval和maxval）采样RGB值。
        # 采样得到的背景颜色形状与weights的除最后一维之外的形状相同，最后一维为3（表示RGB三个通道）。
        bg_rgbs = random.uniform(
            key,
            shape=weights.shape[:-1] + (3,),
            minval=self.bg_intensity_range[0],
            maxval=self.bg_intensity_range[1],
        )


      """这是 RawNeRF 的曝光逻辑部分。"""

       # 如果 ray_results 中存在 RGB 值并且 rays 中有曝光索引。
      if (ray_results['rgb'] is not None) and (rays.exposure_idx is not None):
        # 将 ray_results 中的 RGB 值与 rays.exposure_values 进行逐元素相乘，
        # 实现根据曝光值对颜色进行缩放。
        # Ellipsis 在这里用于处理多维数组的索引，确保维度匹配，
        # 这里是为了在最后一维（颜色通道维度）上进行正确的乘法操作。
        ray_results['rgb'] *= rays.exposure_values[Ellipsis, None, :]

        # 获取 rays.exposure_idx 的第一个元素作为曝光索引。
        # 这里假设 exposure_idx 是一个多维数组，取其第一个维度的值。
        if self.learned_exposure_scaling:
          exposure_idx = rays.exposure_idx[Ellipsis, 0]
          
          # 创建一个掩码 mask，当曝光索引大于 0 时，对应位置为 True，否则为 False。
          # 这样做是为了在曝光索引为 0 时，强制缩放偏移为 0，以此为场景亮度确定一个参考点。
          mask = exposure_idx > 0
          # 计算缩放因子 scaling。当 mask 中对应位置为 True 时，
          # 使用 exposure_scaling_offsets 根据曝光索引计算得到一个偏移量，
          # 然后加上 1；当 mask 中对应位置为 False（即曝光索引为 0）时，scaling 为 1。
          scaling = 1 + mask[Ellipsis, None] * exposure_scaling_offsets(exposure_idx)
          # 将 ray_results 中的 RGB 值再乘以缩放因子 scaling，进一步调整颜色的缩放。
          ray_results['rgb'] *= scaling[Ellipsis, None, :]

      # 渲染每条光线。
      # 定义一个列表 extras_to_render，其中包含要额外渲染的属性，这里只包含 'roughness'，
      # 表示在渲染光线时除了基本的颜色等信息外，还要考虑粗糙度信息（具体取决于渲染函数的实现）。
      extras_to_render = ['roughness']


      """体积渲染部分"""
      rendering = render.volumetric_rendering(
          # 光线结果中的RGB颜色值，作为体积渲染的颜色输入
          ray_results['rgb'],
          # 之前计算得到的权重，用于控制光线在体积渲染中的贡献
          weights,
          # 度量距离，与光线在场景中的传播距离相关，影响渲染效果
          tdist,
          # 背景颜色，用于填充没有光线覆盖的区域或作为光线传播到远处的背景
          bg_rgbs,
          # 一个布尔值，表示是否计算额外信息
          compute_extras,
          # 一个字典，包含额外要渲染的信息。这里通过字典推导式构建，
          # 遍历ray_results中的键值对，只选择键以'normals'开头或者
          # 在extras_to_render列表中的键值对。如果compute_extras为True，
          # 除了基本的颜色信息外，还会渲染法线等额外信息
          extras={
              k: v
              for k, v in ray_results.items()
              if k.startswith('normals') or k in extras_to_render
          },
          # 可能用于在渲染过程中对某些值进行百分位计算，具体取决于volumetric_rendering函数的实现
          percentiles=percentiles,
      )



      """收集可视化光线信息部分"""
      if compute_extras:
        # 收集一些光线用于直接可视化。
        # 通过将这些量命名为`ray_`开头，它们在后续处理中会被区别对待，被视为光线集合，而不是图像块
        n = self.config.vis_num_rays
        # 将sdist（距离信息）重塑为二维数组（第一维是光线数量，第二维是距离相关维度），
        # 然后取前n条光线的距离信息，存储在rendering字典的ray_sdist键下，
        # 用于收集部分光线的距离信息用于可视化或后续分析
        rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
         # 将weights重塑并取前n条光线的权重信息，存储在rendering字典的ray_weights键下
        rendering['ray_weights'] = weights.reshape([-1, weights.shape[-1]])[
            :n, :
        ]
        rgb = ray_results['rgb']
        
        if rgb is not None:
            # 如果光线的颜色值rgb不为None，将其重塑为三维数组（第一维是光线数量，后两维是颜色通道维度），
            # 并取前n条光线的颜色信息，存储在rendering字典的ray_rgbs键下
          rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[
              :n, :, :
          ]
        else:
          # 如果rgb为None，则将rendering['ray_rgbs']设为None
          rendering['ray_rgbs'] = None

      """结果存储部分"""

      # 将当前的渲染结果rendering添加到renderings列表中。
      # 这个列表可能用于存储整个渲染过程中不同阶段或不同层级的渲染结果
      renderings.append(rendering)
      # 将当前的度量距离tdist复制到ray_results字典的tdist键下，更新ray_results中的距离信息
      ray_results['tdist'] = jnp.copy(tdist)
      # 将当前的sdist复制到ray_results字典的sdist键下，
      # 保持ray_results中的sdist信息与当前计算结果一致
      ray_results['sdist'] = jnp.copy(sdist)
      # 将当前的weights复制到ray_results字典的weights键下，
      # 保持ray_results中的weights信息与当前计算结果一致
      ray_results['weights'] = jnp.copy(weights)
      # 将更新后的ray_results添加到ray_history列表中。
      # 这个列表用于记录光线处理的历史信息，可能在调试、分析或者后续的处理中有用
      ray_history.append(ray_results)





    """处理额外信息部分"""
    if compute_extras:
      # 由于提议网络（proposal network）不会产生有意义的颜色，
      # 为了更便于可视化，我们用最终的平均颜色替换它们的颜色。

      # 从渲染结果列表（renderings）中提取每个渲染结果的光线权重（ray_weights），形成一个权重列表。
      weights = [r['ray_weights'] for r in renderings]
      # 从渲染结果列表中提取每个渲染结果的光线 RGB 值（ray_rgbs），形成一个 RGB 值列表。
      rgbs = [r['ray_rgbs'] for r in renderings]
      # 计算最后一个渲染结果中光线 RGB 值与光线权重的加权和，得到最终的平均 RGB 值。
      # 这里的axis=-2可能是对光线维度进行求和操作（假设是多维数组）。
      final_rgb = jnp.sum(rgbs[-1] * weights[-1][Ellipsis, None], axis=-2)
      # 对于除最后一个权重之外的每个权重，将最终的平均 RGB 值广播（broadcast）到与权重形状匹配的维度，
      # 并添加 RGB 通道维度（3），形成平均 RGB 值列表。
      avg_rgbs = [
          jnp.broadcast_to(final_rgb[:, None, :], w.shape + (3,))
          for w in weights[:-1]
      ]

      # 将每个渲染结果（除最后一个）中的光线 RGB 值替换为平均 RGB 值，以便更好地可视化。
      for i, avg_rgb in enumerate(avg_rgbs):
        renderings[i]['ray_rgbs'] = avg_rgb


    """检查 MLP 和网格使用情况部分"""
    if not all(mlp_was_used):
      # 如果不是所有的 MLP 都被使用（mlp_was_used列表中有False值），
      # 则找出未使用的 MLP 索引，将它们连接成字符串，
      # 然后抛出一个 ValueError 异常，提示哪些 MLP 未被采样策略使用。
      s = ', '.join([f'{i}' for i, v in enumerate(mlp_was_used) if not v])
      raise ValueError(f'此MLPs {s} 在采样策略中没有使用。')
    if not all(grid_was_used):
      # 类似地，如果不是所有的网格都被使用（grid_was_used列表中有False值），
      # 则找出未使用的网格索引，将它们连接成字符串，然后抛出一个 ValueError 异常，
      # 提示哪些网格未被采样策略使用。
      s = ', '.join([f'{i}' for i, v in enumerate(grid_was_used) if not v])
      raise ValueError(f'此网格（Grids） {s} 在采样策略中没有使用。')

    # 返回渲染结果列表（renderings）和光线历史信息列表（ray_history）。
    return renderings, ray_history




  def get_scheduled_beta(self, i_level, train_frac=1.0):
    """Scheduling the scale beta for the VolSDF density.
    为VolSDF密度调度尺度beta值。

    Args:
      i_level: int, the index of the sampling level.
      采样层级的索引。

      train_frac: float in [0, 1], what fraction of training is complete.
      训练完成的比例，取值范围在[0, 1]。

    Returns:
      curr_beta: float, the current scale beta.
      当前的尺度beta值。
    """

    # 从self.final_betas中获取当前采样层级（i_level）对应的最小beta值。
    min_beta = self.final_betas[i_level]
    # 设定一个最大beta值为0.5。
    max_beta = 0.5

    # 根据以下公式计算当前beta值：
    # 首先计算分母部分：1.0 + ((max_beta - min_beta) / min_beta) * train_frac**self.rate_beta
    # 其中((max_beta - min_beta) / min_beta)表示最大beta值与最小beta值的
    # 差值相对于最小beta值的比例，
    # train_frac**self.rate_beta表示训练完成比例的rate_beta次方，
    # rate_beta可能是控制训练进度对beta值影响程度的参数。
    # 然后用最大beta值除以这个分母得到当前beta值，
    # 这样随着训练进度（train_frac）的变化，beta值会在最大beta值和最小beta值之间动态调整。
    curr_beta = max_beta * (
        1.0
        / (
            1.0
            + ((max_beta - min_beta) / min_beta) * train_frac**self.rate_beta
        )
    )
    return curr_beta



"""构建一个mip - NeRF 360模型。"""
# rng 随机数生成器，用于模型初始化和其他可能需要随机数的操作。
# rays 输入光线的示例，包含了光线相关的信息，如起点、方向等。
# config 配置类，包含了模型的各种配置参数，如网络结构、训练参数等。
# dataset 数据集对象，用于设置最大曝光值。
# returns:
# model 初始化后的神经网络模块，即带有参数的NeRF模型。
# flax模块的状态，即初始化后的NeRF模型参数。
def construct_model(rng, rays, config, dataset=None):

   # 仅获取10条光线，以在模型构建过程中最小化内存开销。
  ray = jax.tree_util.tree_map(
      # 使用jax.tree_util.tree_map对光线数据结构中的每个元素（假设光线数据是一个树状结构）进行操作。
      # 将每个元素重塑为形状为[-1, x.shape[-1]]，然后取前10个元素，这样可以减少数据量。
      lambda x: jnp.reshape(x, [-1, x.shape[-1]])[:10], rays
  )

  model_kwargs = {}

  # 如果数据集存在且数据集有最大曝光值，则将最大曝光值添加到model_kwargs字典中，用于初始化模型。
  if dataset is not None and dataset.max_exposure is not None:
    model_kwargs['max_exposure'] = dataset.max_exposure

  # 使用给定的配置config和model_kwargs中的其他参数初始化Model对象。
  model = Model(config=config, **model_kwargs)
  init_variables = model.init(
      rng,  # 使用flax初始化随机权重的随机数生成器。
      rng=None,  # 模型内采样使用的随机数生成器，这里设为None。
      rays=ray,
      train_frac=1.0,
      compute_extras=False,
      zero_glo=model.num_glo_features == 0,
  )
  # 使用给定的参数初始化模型，并返回初始化后的变量（参数）。
  # 其中train_frac=1.0表示训练完成比例为100%（可能在某些与训练进度相关的初始化中有作用）。
  # compute_extras=False表示不计算额外信息。
  # zero_glo=model.num_glo_features == 0根据模型全局光照特征数量是否为0来设置相关标志。
  return model, init_variables







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
    




"""这段代码定义了一个名为MLP的类，它是一个可配置的神经网络模块（nn.Module），
用于构建具有多种特性和参数的多层感知机（MLP）结构。这些参数涵盖了网络深度、宽度、
激活函数、各种编码相关的设置（如位置编码、方向编码）、与不同计算相关的设置（如密度计算、
颜色计算、粗糙度计算）、初始化方式、是否包含特定功能（如预测法线、使用反射等）
以及其他各种模型特性相关的参数。"""








# 这个装饰器表明这个类的实例可以通过gin配置进行参数化。
@gin.configurable
class MLP(nn.Module):
  """A PosEnc MLP."""


  # 定义MLP第一部分的深度为8，即网络层数。
  net_depth: int = 8
   # 定义MLP第一部分的宽度为256，即每层的神经元数量
  net_width: int = 256

  # 瓶颈向量的宽度，即特定中间层表示的维度。
  bottleneck_width: int = 256

  # MLP第二部分的深度为1，用于处理与视线方向相关的部分。
  net_depth_viewdirs: int = 1
  # MLP第二部分的宽度为128，即这部分每层的神经元数量。
  net_width_viewdirs: int = 128

  # 定义激活函数为ReLU，用于在网络中引入非线性。
  net_activation: Callable[Ellipsis, Any] = nn.relu

  # 3D点位置编码的最小度数为0。
  min_deg_point: int = 0
   # 3D点位置编码的最大度数为12。
  max_deg_point: int = 12

  # MLP权重的初始化方式为he_uniform。
  weight_init: str = 'he_uniform'

  # 每4层添加一个到输出的跳跃连接。
  skip_layer: int = 4
  # 在第二个MLP中每4层添加一个跳跃连接。
  skip_layer_dir: int = 4

  # RGB通道的数量为3。
  num_rgb_channels: int = 3

  # 视线方向或参考方向的编码度数为4。
  deg_view: int = 4

  # 如果为True，则使用参考方向代替视线方向。
  use_reflections: bool = False

  # 如果为True，则使用特定方向编码（IDE）来编码方向。
  use_directional_enc: bool = False
  # 如果为False且use_directional_enc为True，
  # 在IDE中使用零粗糙度。同时，这个参数用于控制是否启用粗糙度预测。
  enable_pred_roughness: bool = False

  # 粗糙度的激活函数为softplus。
  roughness_activation: Callable[Ellipsis, Any] = nn.softplus
  # 在粗糙度激活前添加到原始粗糙度的偏移量为 -1.0。
  roughness_bias: float = -1.0

  # 如果为True，则预测漫反射和镜面反射颜色。
  use_diffuse_color: bool = False

  # 如果为True，则预测色调。
  use_specular_tint: bool = False

  # 如果为True，将点积(n * 视线方向) 传递给第二个MLP。
  use_n_dot_v: bool = False

  # 添加到瓶颈向量的噪声标准差为0.0。
  bottleneck_noise: float = 0.0

  # 密度的激活函数为softplus。
  density_activation: Callable[Ellipsis, Any] = nn.softplus
  # 在密度激活前添加到原始密度的偏移量为 -1.0。
  density_bias: float = -1.0

  # 添加到原始密度的噪声标准差为0.0。
  density_noise: float = (
      0.0  
  )

  # 如果为True，则使用volsdf表示密度。
  density_as_sdf: bool = False

  # 如果为True，将MLP初始化为球体的有向距离函数（SDF）。
  sphere_init: bool = False
  # 球体初始化的半径为1.0。
  sphere_radius: float = 1.0

  # 在RGB激活前的预乘数为1.0。
  rgb_premultiplier: float = 1.0
  # RGB的激活函数为sigmoid。
  rgb_activation: Callable[Ellipsis, Any] = nn.sigmoid
   # 在颜色激活前添加到原始颜色的偏移量为0.0。
  rgb_bias: float = 0.0
  # 添加到RGB输出的填充值为0.001。
  rgb_padding: float = 0.001

  # 如果为True，则计算预测法线。
  enable_pred_normals: bool = False
  # 如果为True，则不计算与密度相关的法线。
  disable_density_normals: bool = False

  # 如果为True，则不输出RGB值。
  disable_rgb: bool = False

  # 如果为True，使高斯分布是各向同性的。
  isotropize_gaussians: bool = False

  # 可用于扭曲操作的函数，初始化为None。
  warp_fn: Callable[Ellipsis, Any] = None

  # 基础形状为“icosahedron”（二十面体），也可以是“octahedron”（八面体）。
  basis_shape: str = 'icosahedron'

  # 细分数量为2，对于“octahedron”，细分后可能与单位矩阵相关
  basis_subdivisions: int = 2

  # 如果为True，则使用学习到的渐晕映射。
  use_learned_vignette_map: bool = False

  # 如果为True，则在瓶颈处使用曝光相关操作。
  use_exposure_at_bottleneck: bool = False

  # 使用的无迹变换（unscented transform）基础类型为“mean”。
  unscented_mip_basis: str = 'mean'
  # 无迹变换的缩放因子，0表示禁用。
  unscented_scale_mult: float = 0.0

  # GLO向量可以“连接”到瓶颈向量上，或者用于在瓶颈向量上构建“仿射”变换，这里默认是“连接”方式。
  glo_mode: str = 'concatenate'
  
  # 用于在使用GLO代码之前对其进行转换的MLP架构，空元组表示不使用MLP。
  glo_mlp_arch: Tuple[int, Ellipsis] = tuple()
  # GLO MLP的激活函数为silu。
  glo_mlp_act: Callable[Ellipsis, Any] = nn.silu
  # 在处理GLO向量之前的预乘数为1.0。
  glo_premultiplier: float = 1.0

  # 如果density_as_sdf为True，用于volsdf表示的参数，初始值为0.1。
  beta_init: float = 0.1
  # 如果density_as_sdf为True，用于volsdf表示的参数，最小值为0.0001。
  beta_min: float = 0.0001

  # 在计算密度梯度之前应用挤压操作。
  squash_before: bool = False

  # 如果为True，即使启用了网格，也连接位置编码特征。
  use_posenc_with_grid: bool = False
  # 位置编码特征的标量缩放因子，默认为1.0，可用于调整位置编码特征相对于网格特征的权重。
  posenc_feature_scale: float = 1.0
  # 使用瓶颈向量作为方向编码的仿射变换，而不是连接，默认为False。
  use_affine_dir_enc_transform: bool = False
  # 使用一个网格特征作为对数密度，默认为False。
  skip_final_density_layer: bool = False
  
  # 新增参数（阶段1默认关闭）
  use_mip_mlp: bool = False  # 是否启用 Mip-MLP 模块
  mip_block_features: int = 256  # MipBlock 输出维度

  use_feature_pyramid: bool = True  # 是否使用特征金字塔融合
  pyramid_fusion_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)  # 融合权重

  """MLP内部添加增强代码"""
  # wsw添加
  # 添加CGAFusion相关参数
  # use_cga_fusion: bool = True  # 是否使用CGAFusion
  # cga_fusion_dim: int = 256    # CGAFusion的内部维度



  extra_grid_kwargs: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict()
  )




  """这段代码是MLP类中的setup函数，主要执行以下几个功能：
  一是检查在使用反射方向时是否正确设置了法线计算相关的参数，
  如果设置不合理则抛出异常；二是预计算并存储特定基（basis）的转置；
  三是根据是否使用特定方向编码来确定视线方向或参考方向的编码函数。"""
  def setup(self):

    # 新增：初始化 MipBlock（仅在启用时创建）
    if self.use_mip_mlp:
        self.mip_block = MipBlock(features=self.net_width)




    # 检查反射方向相关的法线计算设置，如果不合理则抛出异常
    if self.use_reflections and not (
        self.enable_pred_normals or not self.disable_density_normals
    ):
      raise ValueError('反射方向必须计算法线。')

    # 预计算并存储基（basis）的转置，这里通过调用geopoly.generate_basis生成基
    self.pos_basis_t = jnp.array(
        geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)
    ).T

    # 根据是否使用特定方向编码（use_directional_enc）来确定视线方向或参考方向的编码函数
    if self.use_directional_enc:
      # 如果使用特定方向编码，使用ref_utils.generate_ide_fn生成编码函数
      self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
    else:
       # 如果不使用特定方向编码，定义一个新的编码函数
      def dir_enc_fn(direction, _):
        return coord.pos_enc(
            direction, min_deg=0, max_deg=self.deg_view, append_identity=True
        )
      # 将新定义的编码函数赋值给self.dir_enc_fn
      self.dir_enc_fn = dir_enc_fn






    """MLP内部添加增强代码"""
    # wsw添加
    # 添加CGAFusion模块初始化
    # if hasattr(self, 'use_cga_fusion') and self.use_cga_fusion:
    #     self.cga_fusion = CGAFusion(dim=self.num_rgb_channels)






  """
  这段代码定义了MLP类的__call__方法，实现了多层感知机（MLP）的前向传播过程。
  根据输入的各种参数（如随机数生成器、高斯分布信息、视线方向、图像平面坐标、GLO 向量、
  曝光值、beta 值、网格函数、光线信息、度量距离等），计算并返回光线的相关结果，
  包括颜色（rgb）、密度（density）、法线（normals和normals_pred）、粗糙度（roughness）等。
  在计算过程中，涉及到网络层的初始化、位置编码、密度预测、法线计算、颜色预测以及各种条件判断和
  特殊处理，如根据不同的初始化策略、是否使用特定功能（如反射、GLO 向量、渐晕映射等）来调整计算过程。
  """
  @nn.compact
  def __call__(
      self,
      rng,
      gaussians,
      viewdirs=None,
      imageplane=None,
      glo_vec=None,
      exposure=None,
      curr_beta=None,
      grid=None,
      rays=None,
      tdist=None,
      train = True,
  ):
    """Evaluate the MLP.
    评估多层感知机（MLP）。

    Args:
        rng: jnp.ndarray. Random number generator.
        随机数生成器，用于在需要随机操作的地方生成随机数，比如可能的随机初始化等。

        gaussians: a tuple containing:                                           /
        - mean: [..., n, 3], coordinate means, and                             /
        - cov: [..., n, 3{, 3}], coordinate covariance matrices.
        包含均值和协方差的高斯分布信息。均值是[...，n，3]形状，表示坐标均值；协方差是[...，n，3（可选3×3）]形状，表示坐标协方差矩阵。

        viewdirs: jnp.ndarray(float32), [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
        视线方向向量。如果不为None，它将作为MLP第二部分的输入的一部分，与MLP第一部分的输出向量连接。如果为None，则仅使用MLP的第一部分处理输入x。在原始论文中，此变量是视线方向。

        imageplane: jnp.ndarray(float32), [batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
        图像平面坐标，形状为[batch，2]，是批次中每条光线的xy坐标。对于图像平面操作（如学习的渐晕映射）很有用。

        glo_vec: [..., num_glo_features], The GLO vector for each ray.
        每条光线的GLO向量，形状为[...，num_glo_features]。

        exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.
        每条光线的曝光值，形状为[...，1]，由快门速度和ISO的乘积确定。

        curr_beta: float, beta to be used in the sdf to density transformation, if
        None then using the learned beta.
        用于从有向距离函数（SDF）到密度转换的beta值。如果为None，则使用学习得到的beta值。

        grid: Callable, a function that computes a grid - like feature embeddding
        for a spatial position.
        一个可调用对象（函数），用于计算空间位置的类似网格的特征嵌入。

        rays: util.Rays, a pytree of ray origins, directions, and viewdirs.
        光线信息，包含光线的起点、方向和视线方向的树状结构（pytree）。

        tdist: jnp.ndarray(float32), with a shape of [..., n + 1] containing the
        metric distances of the endpoints of each mip - NeRF interval.
        度量距离数组，形状为[...，n + 1]，包含每个mip - NeRF区间端点的度量距离。

        train: Boolean flag. Set to True when training.
        训练标志，训练时设置为True。

    Returns:
        rgb: jnp.ndarray(float32), with a shape of [..., num_rgb_channels].
        RGB颜色值数组，形状为[...，num_rgb_channels]。

        density: jnp.ndarray(float32), with a shape of [...].
        密度值数组，形状为[...]。

        normals: jnp.ndarray(float32), with a shape of [..., 3], or None.
        法线向量数组，形状为[...，3]或None（如果不计算法线）。

        normals_pred: jnp.ndarray(float32), with a shape of [..., 3], or None.
        预测的法线向量数组，形状为[...，3]或None（如果不预测法线）。

        roughness: jnp.ndarray(float32), with a shape of [..., 1], or None.
        粗糙度值数组，形状为[...，1]或None（如果不计算粗糙度）。
    """

    

    
    
    # 创建一个偏函数 dense_layer，用于创建全连接层。
    # 这里通过 getattr 获取指定的权重初始化函数，并使用该函数来初始化全连接层的权重。
    dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )

    # 创建一个类似的偏函数 view_dependent_dense_layer，用于创建与视图相关的全连接层，
    # 其权重初始化方式与 dense_layer 相同。
    view_dependent_dense_layer = functools.partial(
        nn.Dense,
        kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )

    # 如果 self.sphere_init 为 True，初始化密度相关的全连接层，
    # 使其近似于球体的有向距离函数（参考论文 https://arxiv.org/pdf/1911.10414.pdf 的定理 1）。
    if self.sphere_init:
      # 创建一个偏函数 density_dense_layer，用于创建密度相关的全连接层。
      # 权重使用正态分布初始化，其标准差根据网络宽度计算，偏置初始化为零。
      density_dense_layer = functools.partial(
          nn.Dense,
          kernel_init=jax.nn.initializers.normal(
              jnp.sqrt(2.0) / jnp.sqrt(self.net_width)
          ),
          bias_init=jax.nn.initializers.zeros,
      )

      # 创建一个偏函数 posenc_dense_layer，用于创建与位置编码相关的全连接层。
      # 在初始化时，权重和偏置都初始化为零，这样做是为了仅对位置信息进行建模。
      posenc_dense_layer = functools.partial(
          nn.Dense,
          self.net_width,
          kernel_init=jax.nn.initializers.zeros,
          bias_init=jax.nn.initializers.zeros,
      )

      # 基于球体半径初始化最后一个密度相关的全连接层。
      # 这里可以选择将最后一层初始化为常数值 'init_mean'，但为了增加随机性，添加了一个小的随机变化。
      init_mean = jnp.sqrt(jnp.pi) / jnp.sqrt(self.net_width)
      init_std = 0.0001
      kernel_init = lambda *args: init_mean + random.normal(*args) * init_std
      final_density_dense_layer = functools.partial(
          nn.Dense,
          kernel_init=kernel_init,
          bias_init=jax.nn.initializers.constant(-self.sphere_radius),
      )
    else:
      # 如果 self.sphere_init 为 False，密度相关的全连接层使用之前定义的 dense_layer。
      density_dense_layer = dense_layer
      # 最后一个密度相关的全连接层也使用之前定义的 dense_layer。
      final_density_dense_layer = dense_layer

    # 使用 random_split 函数对随机数生成器 rng 进行分割，
    # 得到用于密度计算的 density_key 和更新后的 rng。
    density_key, rng = random_split(rng)
    # 再次使用 random_split 函数对 rng 进行分割，得到用于网格计算的 grid_key 和更新后的 rng。
    grid_key, rng = random_split(rng)





    """predict_density函数的主要作用是预测密度值，具体如下：
    1. 特征计算与准备
    根据grid是否存在以及其他相关条件计算用于密度预测的特征。如果grid不为None，
    会基于means、control_offsets等计算control，并结合warp_fn、perp_mag等因素确定scale，
    然后将通过grid函数计算得到的特征添加到x列表中。同时，需要注意在 TPU 设备上使用哈希编码相关操作
    （这里涉及到grid相关）会很慢，如果设备是 TPU 则会抛出异常。
    当grid为None或者self.use_posenc_with_grid为True时，使用mip - NeRF 360中的策略对输入进行编码，
    包括对means和covs的处理（如track_linearize、lift_and_diagonalize等操作），
    并将编码结果添加到x中。最后将所有特征x在最后一维上拼接起来。

    2. 密度值计算
    通过一个循环来评估网络以生成密度值。在循环中，根据self.sphere_init和当前层索引i的关系进行不同
    的计算。对于球体初始化相关的特定层（第一层和跳跃连接层），会有特殊的处理来保证初始化与位置的
    关系正确，包括将位置编码部分置零、对输入范数进行平均等操作。其他层则正常进行全连接层计算、
    激活函数应用和跳跃连接操作。
    根据self.skip_final_density_layer的值确定如何获取最终的原始密度raw_density，
    如果self.skip_final_density_layer为True且x的最后一维不是 1，则会抛出异常，否则取x的最后一维
    作为raw_density；如果self.skip_final_density_layer为False，则通过final_density_dense_layer
    计算得到raw_density。
    如果density_key不为None且self.density_noise大于 0，会向raw_density添加噪声进行正则化。

    3. 返回值
    函数最终返回原始密度raw_density和中间计算结果x，x可能在后续其他计算中有用，raw_density则是
    密度预测的核心结果。"""



    def predict_density(means, covs, **kwargs):
      """Helper function to output density."""
      # 创建一个空列表，用于存储后续计算得到的用于计算密度的特征
      x = []

      # 对输入位置进行编码
      if grid is not None:
        control_offsets = kwargs['control_offsets']
        control = means[Ellipsis, None, :] + control_offsets
        perp_mag = kwargs['perp_mag']

        scale = None
        if not self.squash_before and self.warp_fn is not None:
          if perp_mag is not None and self.unscented_scale_mult > 0:
            if self.warp_fn.__wrapped__ == coord.contract:
              # 如果warp_fn是收缩操作（contract），通过特殊处理来加速计算。
              # 直接计算雅可比行列式的立方根，得到缩放因子s，
              # 并根据perp_mag和s计算scale，然后对control应用warp_fn。
              s = coord.contract3_isoscale(control)
              scale = self.unscented_scale_mult * (perp_mag * s)[Ellipsis, None]
              control = self.warp_fn(control)  # pylint: disable=not-callable
            else:
              # 对于其他warp_fn，跟踪等向性变换，更新control和perp_mag，并计算scale。
              control, perp_mag = coord.track_isotropic(
                  self.warp_fn, control, perp_mag
              )
              scale = self.unscented_scale_mult * perp_mag[Ellipsis, None]
          else:
            control = self.warp_fn(control)  # pylint: disable=not-callable

         # 在TPU上从网格中收集/散射数据非常慢，若设备是TPU则抛出异常。
        if utils.device_is_tpu():
          raise ValueError('哈希编码不能用在TPU上边。')

        # 将通过网格函数grid计算得到的特征添加到x列表中，
        # 传入控制参数、缩放因子、训练标志、随机数生成器等相关参数。
        x.append(
            grid(
                control,
                x_scale=scale,
                per_level_fn=math_utils.average_across_multisamples,
                train=train,
                rng=grid_key,
                **self.extra_grid_kwargs,
            )
        )

      if grid is None or self.use_posenc_with_grid:
        # 使用mip - NeRF 360中的策略对输入进行编码
        if not self.squash_before and self.warp_fn is not None:
          means, covs = coord.track_linearize(self.warp_fn, means, covs)

        lifted_means, lifted_vars = coord.lift_and_diagonalize(
            means, covs, self.pos_basis_t
        )
        x.append(
            self.posenc_feature_scale
            * coord.integrated_pos_enc(
                lifted_means,
                lifted_vars,
                self.min_deg_point,
                self.max_deg_point,
            )
        )
      # 将所有计算得到的特征在最后一维上进行拼接
      x = jnp.concatenate(x, axis=-1)

      inputs = x
      # 评估网络以生成输出密度
      

      # 在 predict_density 函数内
      intermediate_features = {}  # 存储中间特征


      # 计算要存储的层索引（基于网络深度）
      if self.net_depth >= 3:
          # 对于深度>=3的网络，存储1/4、1/2和3/4处的层
          layers_to_store = [
              max(1, self.net_depth // 4),
              self.net_depth // 2,
              min(self.net_depth - 1, 3 * self.net_depth // 4)
          ]
      elif self.net_depth == 2:
          # 对于深度为2的网络，只存储第一层
          layers_to_store = [1]
      else:
          # 对于深度为1的网络，存储唯一的一层
          layers_to_store = [0]

      for i in range(self.net_depth):
        # 存储特定层的特征
        if i in layers_to_store:
            intermediate_features[i] = x
            
        if self.sphere_init and (
            i == 0 or ((i - 1) % self.skip_layer == 0 and i > 1)
        ):
          
          # 对于球体初始化，在特定层（第一层和跳跃连接层）在初始化时将位置编码部分置零，
          # 以保证球体初始化与位置（means）的函数关系正确。
          if i == 0:
            x = means
          elif (i - 1) % self.skip_layer == 0 and i > 1:
            x = x[Ellipsis, : -inputs.shape[-1]]
            # 在跳跃连接中，为保持输入范数，对前一层和连接的输入的范数进行平均。
            x = jnp.concatenate([x, means], axis=-1) / jnp.sqrt(2.0)
          x = density_dense_layer(self.net_width)(x) + (
              posenc_dense_layer(self.net_width)(inputs)
         
         
          
          )
        else:
          x = density_dense_layer(self.net_width)(x)
        x = self.net_activation(x)
        if i % self.skip_layer == 0 and i > 0:
          x = jnp.concatenate([x, inputs], axis=-1)
        

      # 密度被硬编码为单通道
      if self.skip_final_density_layer:
        if x.shape[-1] != 1:
          raise ValueError(f'x has {x.shape[-1]} channels, but must have 1.')
        raw_density = x[Ellipsis, 0]
      else:
        raw_density = final_density_dense_layer(1)(x)[Ellipsis, 0]

      # 如果需要，向密度预测添加噪声以进行正则化
      if (density_key is not None) and (self.density_noise > 0):
        raw_density += self.density_noise * random.normal(
            density_key, raw_density.shape
        )
      return raw_density, x, intermediate_features




    # 从输入的高斯分布信息gaussians中提取均值means和协方差covs
    means, covs = gaussians
    # 对输入位置进行编码，如果设置了squash_before且warp_fn不为None，
    # 则对均值means和协方差covs进行线性化跟踪
    if self.squash_before and self.warp_fn is not None:
      means, covs = coord.track_linearize(self.warp_fn, means, covs)

    # 初始化用于预测密度的参数字典
    predict_density_kwargs = {}
    # 如果grid不为None，说明使用网格结构相关的处理
    if grid is not None:
      
      # 网格/哈希结构难以直接与高斯分布进行封闭形式的积分，
      # 所以使用无迹变换（或类似方法）对每个高斯进行采样，并平均采样编码。
      # 首先分割随机数生成器rng，得到用于计算控制点的control_points_key和更新后的rng。
      control_points_key, rng = random_split(rng)
      
      # 计算控制点control和垂直幅度perp_mag，传入均值means、协方差covs、光线rays、
      # 度量距离tdist、control_points_key、无迹变换基类型self.unscented_mip_basis
      # 和无迹变换缩放因子self.unscented_scale_mult等参数。
      control, perp_mag = coord.compute_control_points(
          means,
          covs,
          rays,
          tdist,
          control_points_key,
          self.unscented_mip_basis,
          self.unscented_scale_mult,
      )


      # 计算控制点偏移量control_offsets
      control_offsets = control - means[Ellipsis, None, :]

      # 将控制点偏移量和垂直幅度添加到预测密度的参数字典中
      predict_density_kwargs['control_offsets'] = control_offsets
      predict_density_kwargs['perp_mag'] = perp_mag

    
    # 如果禁用了密度法线计算（self.disable_density_normals为True）
    if self.disable_density_normals:
      # 直接调用predict_density函数计算原始密度raw_density和中间结果x，不计算梯度和法线
      raw_density, x, intermediate_features= predict_density(means, covs, **predict_density_kwargs)
      raw_grad_density = None
      normals = None
    else:
      # 获取输入均值means的维度数减1，用于后续的展平操作
      n_flatten = len(means.shape) - 1
      # 对高斯分布信息（均值means和协方差covs）以及预测密度的参数字典进行展平操作，
      # 使其可以进行向量化操作
      gaussians_flat, pd_kwargs_flat = jax.tree_util.tree_map(
          lambda x: x.reshape((-1,) + x.shape[n_flatten:]),
          ((means, covs), predict_density_kwargs),
      )

      # 创建一个向量化的函数predict_density_and_grad_fn，用于同时计算预测密度函数及其梯度
      predict_density_and_grad_fn = jax.vmap(
          jax.value_and_grad(predict_density, has_aux=True),
      )

      # 调用predict_density_and_grad_fn计算展平后的原始密度、中间结果和原始密度梯度
      (raw_density_flat, x_flat), raw_grad_density_flat = (
          predict_density_and_grad_fn(*gaussians_flat, **pd_kwargs_flat)
      )

      # 对于这个分支，我们暂时不提供 intermediate_features
      intermediate_features = {}

      # 将展平后的原始密度、中间结果和原始密度梯度还原为原始形状
      raw_density = raw_density_flat.reshape(means.shape[:-1])
      x = x_flat.reshape(means.shape[:-1] + (x_flat.shape[-1],))
      raw_grad_density = raw_grad_density_flat.reshape(means.shape)

      # 根据密度是否以有向距离函数（SDF）表示来计算法线向量。
      # 如果是SDF表示，法线向量是原始密度梯度的归一化；否则，是负的原始密度梯度的归一化。
      if self.density_as_sdf:
        normals = ref_utils.l2_normalize(raw_grad_density)
      else:
        normals = -ref_utils.l2_normalize(raw_grad_density)

    # 如果启用了预测法线（self.enable_pred_normals为True）
    if self.enable_pred_normals:
      # 使用全连接层dense_layer计算预测梯度grad_pred
      grad_pred = dense_layer(3)(x)

      # 对预测梯度取负并归一化，得到预测的法线向量normals_pred，后续将使用预测的法线向量
      normals_pred = -ref_utils.l2_normalize(grad_pred)
      normals_to_use = normals_pred
    else:
      grad_pred = None
      normals_pred = None
      normals_to_use = normals




    # 对原始密度应用偏差和激活函数
    # 如果密度表示为有向距离函数（SDF）形式
    if self.density_as_sdf:
      # 如果当前beta值（curr_beta）为None，则从模型参数中获取beta值。
      # 使用名为'beta'的参数，通过nn.initializers.constant初始化为self.beta_init，
      # 形状为空元组（表示标量）。
      # 计算当前beta值curr_beta，取beta的绝对值并加上self.beta_min。
      if curr_beta is None:
        beta = self.param('beta', nn.initializers.constant(self.beta_init), ())
        curr_beta = jnp.abs(beta) + self.beta_min
      
      # 使用计算得到的curr_beta，对原始密度（raw_density）
      # 加上密度偏差（self.density_bias）后进行密度激活函数操作，得到密度值density。
      density = self.density_activation(
          raw_density + self.density_bias, curr_beta
      )
    else:
      # 如果不是SDF形式，直接对原始密度（raw_density）加上密度偏差（self.density_bias）
      # 后进行密度激活函数操作，得到密度值density。
      density = self.density_activation(raw_density + self.density_bias)

    # 初始化粗糙度为None，可能在后续根据条件重新赋值。
    roughness = None






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








    # 如果禁用了RGB输出（self.disable_rgb为True），则将rgb设为None
    if self.disable_rgb:
      rgb = None
    else:
      # 如果视线方向（viewdirs）不为None或者GLO向量（glo_vec）不为None，则进行以下操作
      if viewdirs is not None or glo_vec is not None:
        
        
        # 如果使用漫反射颜色（self.use_diffuse_color为True），
        # 通过全连接层dense_layer计算原始的漫反射RGB值raw_rgb_diffuse
        if self.use_diffuse_color:
          raw_rgb_diffuse = dense_layer(self.num_rgb_channels)(x)


        # 如果使用镜面反射色调（self.use_specular_tint为True），
        # 通过全连接层和sigmoid函数计算色调tint
        if self.use_specular_tint:
          tint = nn.sigmoid(dense_layer(3)(x))


        # 如果启用了粗糙度预测（self.enable_pred_roughness为True），
        # 通过全连接层计算原始粗糙度raw_roughness，并应用粗糙度激活函数得到粗糙度值roughness
        if self.enable_pred_roughness:
          raw_roughness = dense_layer(1)(x)
          roughness = self.roughness_activation(
              raw_roughness + self.roughness_bias
          )
        


        




        """MLP第一部分的输出处理"""
       
       
       
       
        # MLP第一部分的输出处理
        if self.bottleneck_width > 0:
          # 通过全连接层dense_layer计算瓶颈向量bottleneck
          bottleneck = dense_layer(self.bottleneck_width)(x)



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


          # 如果随机数生成器rng不为None且瓶颈噪声（self.bottleneck_noise）大于0，则添加噪声到瓶颈向量
          if (rng is not None) and (self.bottleneck_noise > 0):
            key, rng = random_split(rng)
            bottleneck += self.bottleneck_noise * random.normal(
                key, bottleneck.shape
            )

          # 如果使用在瓶颈处结合曝光（self.use_exposure_at_bottleneck为True）
          # 且曝光值（exposure）不为None，则将曝光值的对数加到瓶颈向量上
          if self.use_exposure_at_bottleneck and exposure is not None:
            bottleneck += jnp.log(exposure)[Ellipsis, None, :]

          x = [bottleneck]
        else:
          x = []


        # 对视线（或反射）方向进行编码
        if viewdirs is not None:
          # 如果使用反射，在反射前先翻转视线方向，因为视线方向是从相机指向点，
          # 而ref_utils.reflect()函数假设方向是朝向相机的。
          # 计算反射方向refdirs，其将从点指向环境。
          if self.use_reflections:
            refdirs = ref_utils.reflect(-viewdirs[Ellipsis, None, :], normals_to_use)
            # 使用dir_enc_fn函数对反射方向进行编码，传入反射方向refdirs和粗糙度roughness
            dir_enc = self.dir_enc_fn(refdirs, roughness)
          else:
            # 如果不使用反射，直接对视线方向viewdirs进行编码，
            # 传入视线方向viewdirs和粗糙度roughness
            dir_enc = self.dir_enc_fn(viewdirs, roughness)

            # 将编码后的方向向量广播到与瓶颈向量bottleneck相同的形状（除了最后一维）
            dir_enc = jnp.broadcast_to(
                dir_enc[Ellipsis, None, :],
                bottleneck.shape[:-1] + (dir_enc.shape[-1],),
            )

          # 将视线（或反射）方向编码添加到x列表中（x列表目前可能只包含瓶颈向量）
          x.append(dir_enc)

        # 如果使用法线与视线方向的点积（self.use_n_dot_v为True），
        # 计算法线向量和视线方向的点积dotprod，并添加到x列表中
        if self.use_n_dot_v:
          dotprod = jnp.sum(
              normals_to_use * viewdirs[Ellipsis, None, :], axis=-1, keepdims=True
          )
          x.append(dotprod)

        # 如果密度表示为有向距离函数（SDF）且不使用反射，则将法线向量normals和均值means添加到x列表中
        if self.density_as_sdf and not self.use_reflections:
          x.append(normals)
          x.append(means)


        # 如果GLO向量可用（glo_vec不为None）
        if glo_vec is not None:
          # 将GLO向量乘以预乘数self.glo_premultiplier
          y = glo_vec * self.glo_premultiplier
          # 可选地将GLO向量通过一个小的MLP进行处理
          for wi, w in enumerate(self.glo_mlp_arch):
            y = self.glo_mlp_act(nn.Dense(w, name=f'GLO_MLP_{wi}')(y))

          if self.glo_mode == 'concatenate':
            # 如果GLO模式为'concatenate'，将转换后的GLO向量连接到瓶颈向量上。
            # 计算连接后的形状，并将GLO向量广播到该形状后添加到x列表中。
            shape = bottleneck.shape[:-1] + y.shape[-1:]
            x.append(jnp.broadcast_to(y[Ellipsis, None, :], shape))

          elif self.glo_mode == 'affine':
            if self.bottleneck_width <= 0:
              # 如果瓶颈宽度小于等于0，使用'affine'模式会有问题，抛出异常
              raise ValueError('瓶颈宽度必须不为零。')
            
            # 如果GLO模式为'affine'，
            # 将转换后的GLO向量通过一个全连接层转换为对瓶颈向量的仿射变换，然后更新瓶颈向量
            y = nn.Dense(
                2 * bottleneck.shape[-1],
                name=f'GLO_MLP_{len(self.glo_mlp_arch)}',
            )(y)
            log_a, b = tuple(
                jnp.moveaxis(y.reshape(y.shape[:-1] + (-1, 2)), -1, 0)
            )
            a = math_utils.safe_exp(log_a)
            bottleneck = a[Ellipsis, None, :] * bottleneck + b[Ellipsis, None, :]
            # 将更新后的瓶颈向量重新赋值给x列表的第一个元素
            x[0] = bottleneck

        # 将瓶颈向量、方向编码和GLO向量（如果有）在最后一维上进行拼接
        x = jnp.concatenate(x, axis=-1)



        """MLP第二部分的输出计算"""


        # Output of the second part of MLP.
        inputs = x
        for i in range(self.net_depth_viewdirs):
          x = view_dependent_dense_layer(self.net_width_viewdirs)(x)
          x = self.net_activation(x)
          if i % self.skip_layer_dir == 0 and i > 0:
            x = jnp.concatenate([x, inputs], axis=-1)

      # 如果使用漫反射/镜面反射颜色，则rgb被视为线性镜面反射颜色，否则视为颜色本身。
      # 通过全连接层view_dependent_dense_layer计算rgb值，并应用rgb激活函数、加上rgb偏差
      rgb = self.rgb_activation(
          self.rgb_premultiplier
          * view_dependent_dense_layer(self.num_rgb_channels)(x)
          + self.rgb_bias
      )

      # 如果使用学习到的渐晕映射，从模型参数中获取渐晕权重vignette_weights（初始化为零，形状为[3,3]）
      if self.use_learned_vignette_map:
        vignette_weights = self.param(
            'VignetteWeights',
            lambda x: jax.nn.initializers.zeros(x, shape=[3, 3]),
        )

        # 使用image_utils.compute_vignette函数计算渐晕效果vignette
        vignette = image_utils.compute_vignette(imageplane, vignette_weights)
        # 将rgb值与渐晕效果相乘，考虑光线采样的额外维度
        rgb *= vignette[Ellipsis, None, :]

      # 如果使用漫反射颜色，初始化线性漫反射颜色，使其组合后的线性颜色初始值约为0.5
      if self.use_diffuse_color:
        diffuse_linear = nn.sigmoid(raw_rgb_diffuse - jnp.log(3.0))
        if self.use_specular_tint:
          specular_linear = tint * rgb
        else:
          specular_linear = 0.5 * rgb

        # 组合镜面反射和漫反射分量，并将线性颜色映射到sRGB颜色空间，限制在[0.0, 1.0]范围内
        rgb = jnp.clip(
            image_utils.linear_to_srgb(specular_linear + diffuse_linear),
            0.0,
            1.0,
        )

      # 应用填充，将颜色映射到[-rgb_padding, 1 + rgb_padding]范围内
      rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding








      """MLP内部添加增强代码"""
      """wsw添加，MLP中融合CGA"""


      # # 第一次尝试，颜色通道混乱    
      # # wsw添加 - CGAFusion融合应该在所有RGB处理完成后进行
      # if self.use_cga_fusion and train and rng is not None:
      #     # 生成增强版本的特征
      #     rng, cga_rng = random.split(rng)
      #     # 使用更合理的增强策略
      #     brightness_factor = 1.0 + 0.2 * random.normal(cga_rng, ())
      #     augmented_rgb = rgb * brightness_factor + 0.05 * random.normal(cga_rng, rgb.shape)
      #     # 应用CGAFusion
      #     rgb = self.cga_fusion(rgb, augmented_rgb)
      # elif self.use_cga_fusion and train:
      #     # 当rng为None时，使用默认增强
      #     augmented_rgb = rgb * 1.2  # 简单的亮度增强
      #     rgb = self.cga_fusion(rgb, augmented_rgb)



      # # 第二次尝试，报初始化错误
      # # 在MLP的__call__方法末尾（所有RGB处理完成后）添加：
      # if self.use_cga_fusion and train and rng is not None:
      #     # 1. 移除随机增强，改用确定性变换
      #     augmented_rgb = jnp.stack([
      #         rgb[...,0] * 1.1,  # R通道轻微增强
      #         rgb[...,1] * 0.9,   # G通道抑制
      #         rgb[...,2] * 0.95   # B通道微调
      #     ], axis=-1)
          
      #     # 2. 带残差的受限融合
      #     fused = self.cga_fusion(rgb, augmented_rgb)
      #     rgb = rgb + 0.1 * (fused - rgb)  # 限制融合强度
          
      #     # 3. 强制输出约束
      #     rgb = jnp.clip(rgb, 
      #                   self.rgb_padding, 
      #                   1 + self.rgb_padding)  # 保持原有padding范围




      # 第三次尝试 失败，放弃MLP内部增强

      # # ▼▼▼ 在此处插入CGAFusion调用 ▼▼▼
      # if self.use_cga_fusion and train:
      #     # 安全增强策略
      #     augmented_rgb = jnp.stack([
      #         rgb[..., 0] * 1.1,  # R轻微增强
      #         rgb[..., 1],        # G保持
      #         rgb[..., 2] * 0.9   # B抑制
      #     ], axis=-1)
          
      #     # 带约束的融合
      #     rgb = self.cga_fusion(
      #         jnp.clip(rgb, 0, 1), 
      #         jnp.clip(augmented_rgb, 0, 1)
      #     )
      #     rgb = jnp.clip(rgb, self.rgb_padding, 1 + self.rgb_padding)
      # # ▲▲▲ 融合结束 ▲▲▲
      
      # # 确保后续没有其他会覆盖rgb的操作
      # if self.use_learned_vignette_map:
      #     vignette = ...  # 原有渐晕计算
      #     rgb *= vignette[..., None, :]


      


    """这段代码的主要目的是根据grid和warp_fn对means进行处理，
    确定哪些点在有效区域内，然后将有效区域外的点的密度设置为零。具体步骤如下：
    如果grid不为None，首先将warped_means初始化为means。如果self.warp_fn函数存在，
    则通过self.warp_fn对means进行变换得到warped_means。
    通过比较warped_means与grid的边界框（grid.bbox）来确定每个点是否在有效区域内。
    jnp.all函数用于检查在最后一个维度上，warped_means的每个元素是否都在边界框范围内。
    如果所有维度的值都满足条件，则density_is_valid中相应的元素为True，否则为False。
    使用jnp.where函数根据density_is_valid的值来更新density。
    如果density_is_valid中的元素为True，则保持density中对应元素的值不变；如果为False，
    则将density中对应元素的值设置为零。这样就实现了将有效区域外的点的密度设置为零的功能。"""

    if grid is not None:
      # 初始化warped_means为means
      warped_means = means
      # 如果warp_fn不为None，对means应用warp_fn得到warped_means
      if self.warp_fn is not None:
        warped_means = self.warp_fn(means)

      # 设置有效区域外的点的密度为零。
      # 如果存在收缩（contraction），这个掩码（mask）对于所有点都为True。
      # 否则，落在边界框（bounding box）外的无效点的密度将被设置为零。
      density_is_valid = jnp.all(
          (warped_means > grid.bbox[0]) & (warped_means < grid.bbox[1]), axis=-1
      )
      density = jnp.where(density_is_valid, density, 0.0)

    


    """
    这段代码的作用是将计算得到的与光线相关的各种属性值整理到一个字典中，并返回这个字典。
    这样，调用__call__函数的代码可以方便地获取和使用这些光线属性结果。
    例如，在基于光线追踪的渲染系统中，这些结果可以用于进一步的渲染计算，如颜色合成、光照计算等。
    """
    
    
    # 创建一个名为ray_results的字典，用于存储光线相关的结果。
    # 将之前计算得到的
    # 密度（density）、
    # 颜色（rgb）、
    # 原始密度梯度（raw_grad_density）、
    # 预测梯度（grad_pred）、
    # 法线（normals）、
    # 预测法线（normals_pred）
    # 和粗糙度（roughness）等信息存入字典。
    ray_results = dict(
        density=density,
        rgb=rgb,
        raw_grad_density=raw_grad_density,
        grad_pred=grad_pred,
        normals=normals,
        normals_pred=normals_pred,
        roughness=roughness,
    )

    # 返回包含光线相关结果的字典ray_results，这些结果将在调用该函数的地方被使用。
    return ray_results


def render_image(
    render_fn,
    rays,
    rng,
    config,
    return_all_levels = False,
    verbose = True,
):
  """Render all the pixels of an image (in test mode).

    Args:
        render_fn: function, jit - ed render function mapping (rng, rays) -> pytree.
        经过JIT编译的渲染函数，将随机数生成器和光线数据映射为一个树状数据结构（pytree）。

        rays: a `Rays` pytree, the rays to be rendered.
        包含光线信息的树状数据结构，这些光线是要被渲染的对象。

        rng: jnp.ndarray, random number generator (used in training mode only).
        随机数生成器，仅在训练模式下使用。

        config: A Config class.
        配置类，包含渲染相关的各种参数设置。

        return_all_levels: return image buffers from ALL levels of nerf resampling.
        是否返回神经辐射场（NeRF）重采样所有层级的图像缓冲区。

        verbose: print progress indicators.
        是否打印进度指示信息。

    Returns:
        rgb: jnp.ndarray, rendered color image_utils.
        渲染后的彩色图像数据。

        disp: jnp.ndarray, rendered disparity image_utils.
        渲染后的视差图像数据。

        acc: jnp.ndarray, rendered accumulated weights per pixel.
        渲染后每个像素的累积权重。
    """
  
  # 获取光线数据中像素的高度和宽度，并计算光线总数
  height, width = rays.pixels.shape[:2]
  num_rays = height * width

  # 将光线数据中的每个张量重塑为(num_rays, -1)的形状
  rays = jax.tree_util.tree_map(lambda r: r.reshape((num_rays, -1)), rays)

  # 获取当前主机的索引
  host_id = jax.process_index()
  chunks = []

  # 生成从0到num_rays，步长为config.render_chunk_size的索引范围
  idx0s = range(0, num_rays, config.render_chunk_size)
  last_chunk_idx = None
  for i_chunk, idx0 in enumerate(idx0s):
    # pylint: disable=cell-var-from-loop
    if verbose and i_chunk % max(1, len(idx0s) // 10) == 0:
      if last_chunk_idx is None:

        # 打印渲染进度信息，如果是第一个块则只打印块编号和总块数
        logging.info('正在渲染块 %d/%d', i_chunk + 1, len(idx0s))
      else:
        # 计算每秒处理的光线数量并打印渲染进度信息，包括块编号、总块数和每秒光线数
        rays_per_sec = (
            (i_chunk - last_chunk_idx)
            * config.render_chunk_size
            / (time.time() - start_chunk_time)
        )
        logging.info(
            '正在渲染块 %d/%d, %0.0f 条光线/每秒',
            i_chunk + 1,
            len(idx0s),
            rays_per_sec,
        )
      start_chunk_time = time.time()
      last_chunk_idx = i_chunk
    # 从光线数据中选取当前块的光线，每个张量的切片操作相同
    chunk_rays = jax.tree_util.tree_map(
        lambda r: r[idx0 : idx0 + config.render_chunk_size], rays
    )

    # 获取当前块实际的光线数量
    actual_chunk_size = chunk_rays.pixels.shape[0]
    # 计算剩余光线数量，用于判断是否需要填充
    rays_remaining = actual_chunk_size % jax.device_count()
    if rays_remaining != 0:
      # 计算填充数量，使光线数量能被设备数量整除
      padding = jax.device_count() - rays_remaining

      def pad_fn(r):
        # 定义填充函数，在第一个维度上填充，其他维度不填充，填充模式为边缘值填充
        return jnp.pad(r, [(0, padding)] + [(0, 0)] * (r.ndim - 1), mode='edge')

      # 对当前块的光线数据进行填充
      chunk_rays = jax.tree_util.tree_map(pad_fn, chunk_rays)
    else:
      padding = 0

    # 填充后，当前块光线数量总是可被主机数量整除
    rays_per_host = chunk_rays.pixels.shape[0] // jax.process_count()
    start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
    
    # 从当前块光线数据中选取当前主机负责处理的光线，并进行分块操作
    chunk_rays = jax.tree_util.tree_map(
        lambda r: utils.shard(r[start:stop]), chunk_rays
    )

    # 调用渲染函数对当前块光线进行渲染，目前未对第二个输出参数进行优化
    chunk_renderings, _ = render_fn(rng, chunk_rays)

    # 对渲染结果进行逆分块操作，考虑填充情况
    chunk_renderings = jax.tree_util.tree_map(
        lambda v: utils.unshard(v[0], padding), chunk_renderings
    )



    

    # # wsw添加
    # 在 models.py 文件的顶部正确引入 CGAFusion
    from camp_zipnerf.internal.cga_fusion import CGAFusion  # 假设 cga_fusion.py 和 models.py 在同一目录下
    from camp_zipnerf.internal.augmentations import augment_pipeline_jax  # 假设 cga_fusion.py 和 models.py 在同一目录下

    # 假设 chunk_renderings 是一个包含多个字典的列表，每个字典可能包含 'ray_rgbs' 键
    for chunk in chunk_renderings:
        if 'ray_rgbs' in chunk:
            # 获取 ray_rgbs 数据并转为 JAX 数组
            ray_rgbs_np = jax.device_get(chunk['ray_rgbs'])  # 获取 NumPy 数组
            input1 = jnp.array(ray_rgbs_np)  # 转换为 JAX 数组

            # 生成随机输入 (形状与 input1 相同)
            # key = jax.random.PRNGKey(0)
            # input2 = jax.random.uniform(key, shape=input1.shape)  # 随机生成对比输入
            
            # # 对 input1 增强并生成 input2
            # brightness_factor = 3  # 定义亮度增强系数
            # input2 = input1 * brightness_factor  # 提升亮度




            rng2 = jax.random.PRNGKey(42)
            rng2, augment_rng = jax.random.split(rng2)
            # 通过增强流水线生成 input2
            input2 = augment_pipeline_jax(input1, augment_rng)
            

            # 实例化 CGAFusion 模块
            cga_fusion = CGAFusion(dim=3)  # dim 设置为 3（RGB 通道数）

            # 初始化 CGAFusion 参数
            params = cga_fusion.init(jax.random.PRNGKey(0), input1, input2)

            # 应用 CGAFusion 模块，进行特征融合
            fused_features = cga_fusion.apply(params, input1, input2)

            # 将融合后的结果更新回 chunk
            chunk['ray_rgbs'] = fused_features  # 融合结果替换原始数据

  




    # 将渲染结果从字典列表转换为列表字典形式
    chunk_renderings = {
        k: [z[k] for z in chunk_renderings if k in z]
        for k in chunk_renderings[-1].keys()
    }

    if not return_all_levels:
      # 如果不返回所有层级，只保留每个图像缓冲区的最后一个层级
      for k in chunk_renderings:
        if not k.startswith('ray_'):
          chunk_renderings[k] = chunk_renderings[k][-1]

    # 将渲染结果移动到CPU上，注意这里原来是jax.block_until_read()，但会导致卡顿，需等待解决方案
    chunk_renderings = jax.device_get(chunk_renderings)

    chunks.append(chunk_renderings)

  # 将所有块的渲染结果在每个树状数据结构的叶子节点上进行拼接
  rendering = jax.tree_util.tree_map(
      lambda *args: jnp.concatenate(args), *chunks
  )

  keys = [k for k in rendering if k.startswith('ray_')]
  if keys:
    num_rays = rendering[keys[0]][0].shape[0]
    ray_idx = random.permutation(random.PRNGKey(0), num_rays)
    ray_idx = ray_idx[: config.vis_num_rays]

  def reshape_fn(key):
    if key.startswith('ray_'):
      # 对于以'ray_'开头的键，从光线可视化缓冲区中随机采样
      return lambda x: x[ray_idx]
    else:
      # 对于其他键，将图像缓冲区重塑为原始分辨率
      return lambda x: x.reshape((height, width) + x.shape[1:])

  # 根据键对渲染结果进行重塑操作
  rendering = {
      k: jax.tree_util.tree_map(reshape_fn(k), z) for k, z in rendering.items()
  }
  if return_all_levels:
    # Throw away useless RGB buffers from proposal network.
    rendering['rgb'] = rendering['rgb'][-1]

  return rendering
