import torch
import numpy as np
import os

'''
将工作目录修改到当前脚本目录
'''
def change_into_current_py_path():
    # 获取当前文件的完整路径
    current_file_path = __file__
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)

    #获取上一级目录
    current_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    #将工作目录位置设置为当前脚本位置
    os.chdir(current_dir)

# """
# 开启wandb
# """
# def init_wandb(project_name,user_name='pengruichengdu'):
#     import wandb
#     wandb.init(project=project_name, entity=user_name)

"""
创建脚本的logger,用于输出日志信息, 返回的logger对象可以直接使用logger.debug, logger.info,
logger.warning, logger.error, logger.critical等方法
"""
def set_logger(path):
    import logging

    logger = logging.getLogger(__name__)        # 获取当前模块的logger
    logger.setLevel(logging.DEBUG)              # 设置日志级别为DEBUG
    console_handler = logging.StreamHandler()   # 创建一个控制台处理器
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - \33[32m%(message)s\033[0m', "%Y-%m-%d %H:%M:%S")     # 设置日志格式
    console_handler.setFormatter(formatter)      # 将格式化器应用到控制台处理器

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")     # 设置日志格式
    fh.setFormatter(formatter)

    logger.addHandler(console_handler)  # 将处理器添加到logger
    logger.addHandler(fh)

    return logger

if __name__ == '__main__':
    pass