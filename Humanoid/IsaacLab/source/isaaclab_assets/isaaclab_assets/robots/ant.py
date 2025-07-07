# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mujoco Ant robot."""

from __future__ import annotations
import os
import omni.log

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 尝试使用本地资源路径
LOCAL_ANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ant_instanceable.usd")

# 如果路径不存在，创建本地资源目录
if not os.path.exists(os.path.dirname(LOCAL_ANT_PATH)):
    try:
        os.makedirs(os.path.dirname(LOCAL_ANT_PATH))
    except Exception as e:
        omni.log.warn(f"Failed to create local directory for ant assets: {e}")

# 定义一个简单的蚂蚁USD文件以防远程资源不可用
SIMPLE_ANT_USD = """#usda 1.0
(
    defaultPrim = "ant"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "ant" (
    kind = "model"
)
{
    def Capsule "body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        uniform token axis = "Z"
        double height = 0.5
        double radius = 0.25
        
        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0.25, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def Cylinder "front_left_leg" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        uniform token axis = "X"
        double height = 0.5
        double radius = 0.08
        
        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (-0.2, 0.2, -0.25, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def Cylinder "front_right_leg" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        uniform token axis = "X"
        double height = 0.5
        double radius = 0.08
        
        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0.2, 0.2, -0.25, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def Cylinder "back_left_leg" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        uniform token axis = "Y"
        double height = 0.5
        double radius = 0.08
        
        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (-0.2, -0.2, -0.25, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def Cylinder "back_right_leg" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        uniform token axis = "Y"
        double height = 0.5
        double radius = 0.08
        
        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0.2, -0.2, -0.25, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def Sphere "front_left_foot" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double radius = 0.1
        
        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (-0.24, 0.24, -0.5, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def Sphere "front_right_foot" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double radius = 0.1
        
        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0.24, 0.24, -0.5, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def Sphere "left_back_foot" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double radius = 0.1
        
        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (-0.24, -0.24, -0.5, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def Sphere "right_back_foot" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double radius = 0.1
        
        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0.24, -0.24, -0.5, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
"""

# 保存简单蚂蚁模型作为后备方案
if not os.path.exists(LOCAL_ANT_PATH):
    try:
        with open(LOCAL_ANT_PATH, "w") as f:
            f.write(SIMPLE_ANT_USD)
        omni.log.info(f"Created simple ant USD model at {LOCAL_ANT_PATH}")
    except Exception as e:
        omni.log.warn(f"Failed to create simple ant USD file: {e}")

##
# Configuration
##

try:
    # 检查远程路径是否可访问
    from isaaclab.utils.assets import check_file_path
    remote_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Ant/ant_instanceable.usd"
    if check_file_path(remote_path) > 0:
        # 远程路径可访问
        ant_path = remote_path
        omni.log.info(f"Using remote ant model: {remote_path}")
    else:
        # 远程路径不可访问，使用本地路径
        ant_path = LOCAL_ANT_PATH
        omni.log.info(f"Remote ant model not accessible, using local: {LOCAL_ANT_PATH}")
except Exception as e:
    # 出现异常，默认使用本地路径
    ant_path = LOCAL_ANT_PATH
    omni.log.warn(f"Error checking remote ant model, using local path: {e}")

ANT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # 使用确定的路径
        usd_path=ant_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*_leg": 0.0,
            "front_left_foot": 0.785398,  # 45 degrees
            "front_right_foot": -0.785398,
            "left_back_foot": -0.785398,
            "right_back_foot": 0.785398,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Mujoco Ant robot."""
