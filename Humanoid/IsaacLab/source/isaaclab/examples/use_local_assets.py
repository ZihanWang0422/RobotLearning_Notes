#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""示例:如何在Python代码中设置本地资产路径。"""

import os
import sys
from pathlib import Path


def main():
    """展示如何使用本地资产运行IsaacLab场景。"""
    # 确定本地资产路径
    local_assets_path = os.path.join(os.getcwd(), "local_assets")
    os.makedirs(local_assets_path, exist_ok=True)
    local_assets_path = os.path.abspath(local_assets_path)
    
    print(f"使用本地资产路径: {local_assets_path}")
    
    # 将本地路径转换为URI格式
    file_uri = f"file:///{local_assets_path.replace(os.sep, '/')}"
    
    # 方式1: 在启动IsaacSim之前设置
    # ============================
    try:
        from omni.isaac.kit import SimulationApp
    except ImportError:
        print("错误: 无法导入omni.isaac.kit模块。请确保IsaacSim已正确安装。")
        sys.exit(1)
    
    # 创建启动配置
    launch_config = {
        "headless": False,  # 可以设置为True使用无头模式
        "renderer": "RayTracedLighting",
        "override_config": {
            "persistent": {
                "isaac": {
                    "asset_root": {
                        "default": file_uri,
                        "cloud": file_uri,
                        "nvidia": file_uri
                    }
                }
            }
        }
    }
    
    # 启动应用
    print("启动IsaacLab，使用本地资产路径...")
    app = SimulationApp(launch_config)
    
    # 方式2: 在应用运行中设置
    # ============================
    # 注意:此方式仅在SimulationApp已经启动后使用
    try:
        # 直接使用提供的本地资产工具
        from isaaclab.utils.local_assets import (
            set_local_asset_path,
            create_asset_directory_structure,
            setup_local_ant_asset
        )
        
        # 设置本地资产路径
        set_local_asset_path(local_assets_path)
        
        # 创建标准目录结构
        create_asset_directory_structure(local_assets_path)
        
        # 设置蚂蚁模型
        ant_path = setup_local_ant_asset(local_assets_path)
        print(f"蚂蚁模型文件路径: {ant_path}")
        
        # 创建场景
        print("创建测试场景...")
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        
        # 创建地面
        create_prim(
            "/World/Ground",
            "Cube",
            translation=(0, 0, -1),
            scale=(10, 10, 0.1),
            attributes={"physics:collisionEnabled": True}
        )
        
        # 尝试添加蚂蚁
        try:
            add_reference_to_stage(usd_path=ant_path, prim_path="/World/Robot")
        except Exception as e:
            print(f"添加蚂蚁模型时出错: {e}")
        
        # 启动事件循环
        app.update()
        app.run()
    except Exception as e:
        print(f"执行示例时出错: {e}")
    finally:
        app.close()


if __name__ == "__main__":
    main() 