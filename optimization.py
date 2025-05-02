# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# optimization.py

import torch
from torch.optim.lr_scheduler import StepLR

def get_optimizer(model, learning_rate):
    """返回 Adam 优化器和梯度裁剪函数"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 定义梯度裁剪函数
    def clip_gradients():
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)  # 使用正确的梯度裁剪

    return optimizer, clip_gradients  # 返回正确的函数名


def get_step_scheduler(optimizer, config):
        """
        获取学习率调度器

        Args:
            optimizer: 优化器
            config: 配置字典

        Returns:
            torch.optim.lr_scheduler: 学习率调度器
        """
        # 添加默认值，以防配置文件中没有相关设置
        scheduler_config = config.get('scheduler', {})
        step_size = scheduler_config.get('step_size', 1)
        gamma = scheduler_config.get('gamma', 0.95)

        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )

