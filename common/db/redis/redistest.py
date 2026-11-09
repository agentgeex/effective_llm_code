'''
  python redis
  https://blog.csdn.net/qq_38628046/article/details/128123641
'''
from Redis import Redis

rediscon = Redis(1).getRedisConn()
rediscon.set("mykey_2025-01-09",f"hello python redis")

curr_date_string = rediscon.get('mykey_2025-01-09')
ddd = rediscon.hgetall('ddddd')
print(f"{curr_date_string.decode('utf-8')}")
# rediscon.hset('ask_mat_source_lib_repeat_key',12312341,1)
#
# value = rediscon.hget('ask_mat_source_lib_repeat_key',12312341)
# str_result = int(value)
#value == None
#r.hdel('user1', '12312341')
#print(curr_date_string.decode('utf-8'))

pipe = rediscon.pipeline()
pipe.hset('testmak1','a',1)
pipe.hset('testmak1','b',1)
pipe.hset('testmak1','c',1)
pipe.hset('testmak1','d',1)
pipe.hset('testmak1','e',1)
pipe.hset('testmak1','f',1)
pipe.hset('testmak1','g',1)
result = pipe.execute()
print(result)
