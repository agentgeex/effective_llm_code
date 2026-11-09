########################################################################################################################
#
# 系统配置文件
#
# 配置文件解析策略（阶段 1-A 改造，只改「值从哪来」，不改业务逻辑）：
#   1. 环境变量 AI_STUDY_CONFIG 指定具体文件（若指定但文件不存在 → 直接报错，不静默降级）
#   2. <repo>/config/config.json          （真值文件，不入库，见 .gitignore）
#   3. <repo>/config/config.example.json  （全占位符，入库，供读者照着填）
#   读取优先级：环境变量 > config.json > config.example.json
#   读不到时报 ConfigError，消息里只说「缺哪个 key、该设哪个环境变量」，不含任何值。
#
########################################################################################################################

import json
import os

from common.util.singleton import Singleton

GLOBAL_PARAMETER = {}

# 默认数据库档案名：环境变量 POOLGUARDER_DB 为空时使用；
# 与 config/config.example.json 中 db.mysql / db.redis 下的键一致
DEFAULT_DB_PROFILE = "db"

# 配置文件查找
CONFIG_ENV_KEY = "AI_STUDY_CONFIG"
CONFIG_RELATIVE_PATH = "config/config.json"
CONFIG_EXAMPLE_RELATIVE_PATH = "config/config.example.json"

# example 文件里的占位符标记：命中即视为「未配置」，读密钥时不会把占位符当成真值
PLACEHOLDER_MARKERS = ("REPLACE_ME", "your_")

_UNSET = object()


class ConfigError(RuntimeError):
    """配置缺失或非法。消息中只出现 key 名 / 环境变量名，不出现任何值。"""


def _repo_root():
    """仓库根目录（本文件位于 <repo>/config/cfg.py）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_placeholder(value):
    """值是否是 example 里的占位符（含空串）。"""
    if not isinstance(value, str):
        return False
    v = value.strip()
    return v == "" or any(marker in v for marker in PLACEHOLDER_MARKERS)


@Singleton
class Syscfg:

    _paramter =  {}
    _config_json = None
    _env = {}
    _app_root_dir = ''
    _config_path = ''
    _config_source = ''

    def __init__(self):

        #环境变量的获取方式
        self._paramter['db'] = os.environ.get('POOLGUARDER_DB') or DEFAULT_DB_PROFILE
        self._paramter['modeldir'] = os.environ.get('MODEL_DIR')
        # APP_ROOT_DIR 未设置时回落到本文件所在仓库根目录
        # （旧代码 os.environ.get 返回 None，拼出 "None/config/config.json"，这是 None 的来源）
        self._app_root_dir = os.environ.get("APP_ROOT_DIR") or _repo_root()

        self._config_json = self._load_config()

    # 按 AI_STUDY_CONFIG → config.json → config.example.json 的顺序找配置文件
    def _load_config(self):

        candidates = []
        env_config = os.environ.get(CONFIG_ENV_KEY)
        if env_config:
            candidates.append((os.path.abspath(env_config), f"env {CONFIG_ENV_KEY}"))
        else:
            candidates.append((os.path.join(self._app_root_dir, CONFIG_RELATIVE_PATH), "config.json"))
            candidates.append((os.path.join(self._app_root_dir, CONFIG_EXAMPLE_RELATIVE_PATH), "config.example.json"))

        tried = []
        for path, label in candidates:
            if not os.path.exists(path):
                tried.append(f"{label}: {path}（不存在）")
                continue
            try:
                with open(path, encoding="utf-8") as config_file:
                    self._config_path = path
                    self._config_source = label
                    return json.load(config_file)
            except json.JSONDecodeError as e:
                raise ConfigError(
                    f"配置文件 JSON 解析失败（{label}: {path}，第 {e.lineno} 行第 {e.colno} 列）：{e.msg}"
                ) from None

        raise ConfigError(
            "找不到配置文件。已尝试：" + "；".join(tried) +
            f"。请复制 {os.path.join(self._app_root_dir, CONFIG_EXAMPLE_RELATIVE_PATH)} 为 "
            f"{os.path.join(self._app_root_dir, CONFIG_RELATIVE_PATH)} 并填入真实值，"
            f"或用环境变量 {CONFIG_ENV_KEY} 指定配置文件路径。"
        )

    # 当前生效的配置文件路径 / 来源（排查用，不含值）
    def getConfigPath(self):
        return self._config_path

    def getConfigSource(self):
        return self._config_source

    # 获取命令行参数
    def getCommandLineParam(self,parmkey):
        return self._paramter.get(parmkey)

    #获取系统总配置文件的数据
    def getConfigInfo(self):
        return self._config_json

    # 取嵌套配置值（keys 为逐级 key），取不到时按 need 报错或返回 default
    def _get_nested(self, keys, source_desc, need=True, default=None):

        node = self._config_json
        walked = []
        for key in keys:
            walked.append(str(key))
            if not isinstance(node, dict) or key not in node:
                if not need:
                    return default
                raise ConfigError(
                    f"配置缺失：{source_desc}（{'.'.join(walked)}）。"
                    f"请在 {self._config_path} 中补上该项，或按 .env.example 设置对应环境变量。"
                )
            node = node[key]
        return node

    # 通用取值：环境变量 → config.json → config.example.json → 默认值
    def getValue(self, env_key, *keys, default=_UNSET):
        env_value = os.environ.get(env_key)
        if env_value is not None and env_value.strip() != "":
            return env_value
        if keys:
            value = self._get_nested(keys, f"配置项 {'/'.join(str(k) for k in keys)}", need=False)
            if value is not None and not _is_placeholder(value):
                return value
        if default is _UNSET:
            raise ConfigError(
                f"缺少配置：环境变量 {env_key} 未设置，且配置文件中没有可用值。"
                f"请 export {env_key}=<值>（参见 .env.example）。"
            )
        return default

    # 密钥取值：只认环境变量或 config.json 里的真值；example 的占位符一律当未配置
    def getSecret(self, env_key, *keys):
        env_value = os.environ.get(env_key)
        if env_value is not None and env_value.strip() != "":
            return env_value
        if keys:
            value = self._get_nested(keys, f"密钥项 {'/'.join(str(k) for k in keys)}", need=False)
            if value is not None and not _is_placeholder(value):
                return value
        raise ConfigError(
            f"缺少密钥：环境变量 {env_key} 未设置（config.example.json 中的占位符不算真值）。"
            f"请 export {env_key}=<你的密钥>（变量名见 .env.example）。"
        )

    ####################################################################################################################
    #
    #  业务方法
    #
    ####################################################################################################################

    # 获取当前mysql数据库的配置
    def getCurrentMysqlConfig(self):
        profile = self.getCommandLineParam("db")
        return self._get_nested(
            ['db', 'mysql', profile],
            f"数据库配置 db.mysql.{profile}（档案名由环境变量 POOLGUARDER_DB 决定，默认 '{DEFAULT_DB_PROFILE}'）",
        )

    # 获取当前mysql数据库的配置
    def getCurrentRedisConfig(self):
        profile = self.getCommandLineParam("db")
        return self._get_nested(
            ['db', 'redis', profile],
            f"Redis 配置 db.redis.{profile}（档案名由环境变量 POOLGUARDER_DB 决定，默认 '{DEFAULT_DB_PROFILE}'）",
        )

    def getCurrentRootDir(self):
        return os.environ.get('APP_ROOT_DIR')

    #获取当环境变量中的日志显示级别
    def getCurrentEvnLoggerShowLevel(self):
        return os.environ.get('LOGGING_SHOW_LEVEL')
    #获取当前环境变量中的存储的日志模块
    def getCurrentEvnLoggerLevel(self):
        return os.environ.get('LOGGING_MYSQL_LEVEL')

    def getCuurentEvnBcqueue(self):
        return os.environ.get('BACKGROUND_QUEUE')

    def getEnvValue(self,key):
        return os.environ.get(key)

    """获取超级管理员"""
    def getCurrentadmin(self):
        configinfo = self.getConfigInfo()
        adminusers = self._get_nested(['adminusers'], "管理员列表 adminusers（逗号分隔的 pkid 字符串）")
        return adminusers

    """获取系统的运行模式"""
    def getRunModel(self):
        # configinfo = self.getConfigInfo()
        # return configinfo['app_run_model']
        return os.environ.get('APP_RUN_MODEL')

    def read_global(self,key):
        param = GLOBAL_PARAMETER.get(key,'')
        return param

    def modify_global(self,key,value):
        GLOBAL_PARAMETER[key] = value

    def print_global(self):
        print(GLOBAL_PARAMETER)
