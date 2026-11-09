import os
import logging
import coloredlogs
import logging.config
from config.cfg import Syscfg


COLORS = {
    'DEBUG': '\033[1;32m',  # 绿色
    'INFO': '\033[1;34m',  # 蓝色
    'WARNING': '\033[38;5;208m',  # 橙色
    'ERROR': '\033[1;31m',  # 红色
    'CRITICAL': '\033[1;35m',  # 紫色
}

def getLog(logger_level=""):

    # for level, color in COLORS.items():
    #     logging.addLevelName(getattr(logging, level), f'{color}{level}{color}')

    cfg = Syscfg()
    logger_env_level = cfg.getCurrentEvnLoggerLevel()
    logger_global_level = cfg.read_global("loggermodel")

    if logger_level == '' or len(logger_level) == 0 or logger_level == None:
        if logger_global_level == '' or logger_global_level == None:
            if logger_env_level != None and len(logger_env_level)>0 :
                logger_level = logger_env_level
        else:
            logger_level = logger_global_level

    # logging.conf 里的 FileHandler 用相对路径 logs/*.log；目录不存在会在 fileConfig 阶段直接 FileNotFoundError。
    # 这里先建目录（仓库根 + 当前工作目录各一份），读者 clone 下来不用手动 mkdir。
    os.makedirs(os.path.join(cfg._app_root_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
    logging_configfile = f"{cfg._app_root_dir}/config/logging.conf"
    logging.config.fileConfig(logging_configfile)
    logger = logging.getLogger(logger_level)

    logger_show_level = cfg.getCurrentEvnLoggerShowLevel()
    if logger_show_level == '' or logger_show_level == None:
        coloredlogs.install(level='INFO',logger=logger)
    else:
        coloredlogs.install(level=logger_show_level,logger=logger)

    return logger
