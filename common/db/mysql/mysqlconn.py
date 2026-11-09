import pymysql
from timeit import default_timer
from config.cfg import Syscfg
from dbutils.pooled_db import PooledDB
from common.util.singleton import Singleton

from common.log import logservice
logger = logservice.getLog("")


@Singleton
class UsingMysql(object):

    _connpool = None
    _host = None
    _port = 0
    _user = None
    _password = None
    _db = None
    _charset = None
    _ifpool = 0
    _pool_size = 0


    def __init__(self, commit=True, log_time=True, log_label='总用时'):
        """
        :param commit: 是否在最后提交事务(设置为False的时候方便单元测试)
        :param log_time:  是否打印程序运行总时间
        :param log_label:  自定义log的文字
        """
        self._log_time = log_time
        self._commit = commit
        self._log_label = log_label

        cfg = Syscfg()
        dbinfo = cfg.getCurrentMysqlConfig()


        self._host = dbinfo['host']
        self._port = int(dbinfo['port'])
        self._user = dbinfo['username']
        self._password = dbinfo['passwd']
        self._db = dbinfo['database']
        self._charset = dbinfo['charset']
        self._ifpool = dbinfo['ifusrpool']
        self._pool_size = dbinfo['pool_size']


        #初始化连接池
        if self._ifpool == 1:

            self._connpool = PooledDB(  pymysql,
                                        maxconnections=self._pool_size,
                                        host=self._host,
                                        port=self._port,
                                        user=self._user,
                                        password=self._password,
                                        db=self._db,
                                        charset=self._charset,
                                        connect_timeout=60*10,  # 设置连接超时时间为600秒
                                        read_timeout=60*10   # 设置读取超时时间为500秒
                                      )




    #获取链接
    def get_connection(self):

        dbconn = None
        if  self._ifpool == 1:
            dbconn =  self._connpool.connection()
        else:

            dbconn = pymysql.connect(

                                        host=self._host,
                                        port=self._port,
                                        user=self._user,
                                        password=self._password,
                                        db=self._db,
                                        charset=self._charset,
                                        connect_timeout=60 * 10,  # 设置连接超时时间为600秒
                                        read_timeout=60 * 10  # 设置读取超时时间为500秒

                                    )

        return dbconn


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
        try:
            # 提交事务
            if self._commit:
                self._conn.commit()
            # 在退出的时候自动关闭连接和cursor
            self._cursor.close()
            self._conn.close()

            if self._log_time is True:
                diff = default_timer() - self._start
                logger.debug('-- %s: %.6f 秒' % (self._log_label, diff))
        except Exception as e:
            logger.error(f" 当前mysql数据库出现异常，异常信息为  {e} ")
            logger.error(e, exc_info=True)
    @property
    def cursor(self):
        return self._cursor



