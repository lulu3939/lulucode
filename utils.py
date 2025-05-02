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

import os
import re
import torch
import glob
import shutil
import logging

logging.basicConfig(level=logging.INFO)

def _create_if_not_exist(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def save_model(output_path, model, epoch, best_score, current_score):
    """
    保存模型参数。

    Args:
        output_path (str): 模型保存的路径。
        model (torch.nn.Module): 要保存的模型。
        epoch (int): 当前的训练轮数。
        best_score (float): 当前最佳分数。
        current_score (float): 当前分数。
    """
    # 确保输出路径存在
    os.makedirs(output_path, exist_ok=True)

    # 如果当前分数优于最佳分数，则保存模型
    if current_score < best_score:
        model_save_path = os.path.join(output_path, f"best_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved to {model_save_path}")
        return model_save_path  # 返回保存路径
    return None  # 未保存模型

def load_model(output_path, model, opt=None, lr_scheduler=None):
    def version(x):
        x = int(x.split("_")[-1])
        return x

    ckpt_paths = glob.glob(os.path.join(output_path, "model_*"))
    steps = 0
    if len(ckpt_paths) > 0:
        output_dir = sorted(ckpt_paths, key=version, reverse=True)[0]

        model_state_dict = torch.load(os.path.join(output_dir, "ckpt.pt"))
        model.load_state_dict(model_state_dict)
        logging.info("load model from  %s" % output_dir)

        if opt is not None and os.path.exists(os.path.join(output_dir, "opt.pt")):
            opt_state_dict = torch.load(os.path.join(output_dir, "opt.pt"))
            opt.load_state_dict(opt_state_dict)
            logging.info("restore optimizer")

        if lr_scheduler is not None and os.path.exists(os.path.join(output_dir, "lr_scheduler.pt")):
            lr_scheduler_state_dict = torch.load(os.path.join(output_dir, "lr_scheduler.pt"))
            lr_scheduler.load_state_dict(lr_scheduler_state_dict)
            logging.info("restore lr_scheduler")

        if os.path.exists(os.path.join(output_dir, "step.pt")):
            steps = torch.load(os.path.join(output_dir, "step.pt"))["global_step"]
            logging.info("restore steps")
        return steps