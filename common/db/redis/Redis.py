import redis
from common.util.singleton import Singleton
from config.cfg import Syscfg, ConfigError
from common.log import logservice
logger = logservice.getLog("")


#@Singleton
class Redis:

    _connpool = None
    _host = None
    _port = 0
    _password = None
    _db = None
    _max_connections = None
    # _charset = None
    # _ifpool = 0
    # _pool_size = 0
    _redis_comm_pool = None
    _if_unix_socker = 0

    def __init__(self,db=0):


        try:
            cfg = Syscfg()
            dbinfo = cfg.getCurrentRedisConfig()

            self._host = dbinfo['host']
            self._port = int(dbinfo['port'])
            self._password = dbinfo['passwd']
            self._db = db
            self._max_connections = dbinfo['max_connections']
            self._if_unix_socker = dbinfo.get('if_unix_socker',0)
            #logger.info(f"dbinfo = > {dbinfo}")


            # 旧写法的注释里曾带明文口令，已清除；口令统一走 config.json / 环境变量（见 .env.example）
            self._redis_comm_pool = redis.ConnectionPool(host=self._host,port=self._port,max_connections=self._max_connections,db=self._db,password=self._password)

        except ConfigError:  # 配置错误必须上抛，不能被下面的通用 except 静默吞掉
            raise
        except Exception as e:
            logger.error(f" find error {e} ")
            logger.error(f"{e.__traceback__}")
            logger.error(e,exc_info=True)


    def getRedisConn(self):

        try:
            if self._if_unix_socker == 1:
                return redis.Redis(unix_socket_path='/tmp/redis.sock')
            elif self._if_unix_socker == 0:
                return redis.Redis(connection_pool=self._redis_comm_pool)
        except Exception as e:
            logger.error(f" find error {e} ")
            logger.error(f"{e.__traceback__}")
            logger.error(e,exc_info=True)
