import os
import pymysql
from timeit import default_timer
from dbutils.pooled_db import PooledDB

MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
if not MYSQL_PASSWORD:
    raise RuntimeError("缺少环境变量 MYSQL_PASSWORD：请先 export MYSQL_PASSWORD=<mysql 口令>（见 .env.example）")


class UsingMysql():


    _connpool = None


    def __init__(self, commit=True, log_time=True, log_label='总用时'):
        """
        :param commit: 是否在最后提交事务(设置为False的时候方便单元测试)
        :param log_time:  是否打印程序运行总时间
        :param log_label:  自定义log的文字
        """
        self._log_time = log_time
        self._commit = commit
        self._log_label = log_label

        self._connpool = PooledDB(pymysql, 10,  host="127.0.0.1",  # IP地址
                                            port=3306,  # 端口号
                                            user="root",  # 用户名
                                            password=MYSQL_PASSWORD,  # 密码改为环境变量 MYSQL_PASSWORD，见 .env.example
                                            db="poolguarder-model-train",  # 数据库名
                                            charset="utf8")

    def get_connection(self):

        # conn = pymysql.connect(
        #     host="127.0.0.1",  # IP地址
        #     port=3306,  # 端口号
        #     user="root",  # 用户名
        #     password=MYSQL_PASSWORD,  # 密码（环境变量，见 .env.example）
        #     db="poolguarder-model-train",  # 数据库名
        #     charset="utf8"  # 字符集编码
        # )
        return self._connpool.connection()


    def __enter__(self):

        # 如果需要记录时间
        if self._log_time is True:
            self._start = default_timer()

        # 在进入的时候自动获取连接和cursor
        conn = self.get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        conn.autocommit = False

        self._conn = conn
        self._cursor = cursor
        return self

    def __exit__(self, *exc_info):
        # 提交事务
        if self._commit:
            self._conn.commit()
        # 在退出的时候自动关闭连接和cursor
        # self._cursor.close()
        # self._conn.close()

        if self._log_time is True:
            diff = default_timer() - self._start
            print('-- %s: %.6f 秒' % (self._log_label, diff))

    @property
    def cursor(self):
        return self._cursor



instance = UsingMysql()