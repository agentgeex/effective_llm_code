# 通用工具类
import base64
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from functools import wraps
import shlex
import subprocess
import time
# from service import commonservice
import traceback
import calendar
import random
from common.log import logservice
logger = logservice.getLog("")


context = {}


def timing_decorator(func):
    """统计方法执行时间的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 更高精度的计时
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"函数 {func.__name__} 执行耗时: {elapsed_time:.6f} 秒")
        return result
    return wrapper


# 获取全局变量
def getcontext(key):
    return context.get(key, None)


# 单例的装饰器
def Singleton(cls):
    instance = {}

    def _singleton_wrapper(*args, **kargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kargs)
        return instance[cls]

    return _singleton_wrapper


# 自动记录进程运行情况的包装器
def background_process_report(commonservice, var1, var2, run_rusult, parms={}):

    def decorator(func):

        def wapper(*args, **kargs):
            try:
                process_pkid = commonservice.crate_model_run_report(f'{func.__name__}', var1, var2)
                context[func.__name__] = process_pkid
                logger.debug("方法开始执行")
                result = func(*args, **kargs)
                logger.debug("方法结束后执行")
                commonservice.update_mode_run_report(pro_pkid=process_pkid, run_result=run_rusult)
            except BaseException as e:
                commonservice.update_mode_run_report(pro_pkid=process_pkid, run_result='发生错误')
                logger.error(str(e))  # 记录错误信息
                logger.error(traceback.format_exc())  # 记录详细的错误信息

        return wapper
    return decorator


# 将图片文件转换成base64
def image2Base64(imagefile):
    with open(imagefile, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    encoded_string = encoded_string.decode("utf-8")
    return encoded_string


"""
        封装一个执行shell的方法
        封装了subprocess的Popen方法，支持超时判断，支持读取stdout和stderr
        参数：
            cwd: 运行路径，如果被设定，子进程会切换到cwd
            timeout: 超时时间， 秒， 支持小数，精度0.1秒
            shell: 是否通过shell运行
        返回： [return_code(int),'stdout(file handle)','stderr(file handle)']
        Raises: TimeoutError: 执行超时
        在外部捕捉此错误
        注意：如果命令带有管道，必须用shell=True

    :param cmd:  string, 执行的命令
    :param cwd:  string, 运行路径，如果被设定，子进程会切换到cwd
    :param timeout:  float, 超时时间， 秒， 支持小数，精度0.1秒
    :param shell:  bool, 是否通过shell运行
    :param _stdin: PIPE, 自定义stdin
    :param _stdout: PIPE, 自定义stdout
    :param _stderr: PIPE, 自定义stderr
    :return: return_code(int), stdout(string), stderr(string)
    :raise TimeoutError: 运行超时
"""


def run(cmd,
        cwd=None,
        timeout=600,
        shell=False,
        _stdin=subprocess.PIPE,
        _stdout=subprocess.PIPE,
        _stderr=subprocess.PIPE):

    # 防止卡死，强制设定一个超时，10分钟
    end_time = datetime.now() + timedelta(seconds=timeout)
    try:
        if shell:
            cmd_list = cmd
        else:
            cmd_list = shlex.split(cmd)
        sub = subprocess.Popen(cmd_list,
                               cwd=cwd,
                               stdin=_stdin,
                               shell=shell,
                               bufsize=4096,
                               stdout=_stdout,
                               stderr=_stderr)
        while sub.poll() is None:
            time.sleep(0.1)
            if end_time < datetime.now():
                raise TimeoutError("Shell Run Timeout(%s sec): %s" % (timeout, cmd))
        if sub.stdout:
            stdout = sub.stdout.read()
            if type(stdout) == bytes:
                stdout = stdout.decode()
        else:
            stdout = ''
        if sub.stderr:
            stderr = sub.stderr.read()
            if type(stderr) == bytes:
                stderr = stderr.decode()
        else:
            stderr = ''
        return int(sub.returncode), stdout, stderr
    except Exception as e:
        return 1, "", str(e)


'''
根据开始日期和结束日期获取时间段信息

identi 的值

curr_day  今天
before_day 前一天
after_day 后一天

curr_week  本周
before_week 前一周
after_week 后一周

curr_month  本月
before_month 上月
after_month 下月


'''


def get_date_slot(start_dt='', end_dt='', identi='curr_day', format="%Y-%m-%d"):

    c_start_dt = '0'
    c_end_dt = '0'

    '''最近三天'''
    if identi == 'last30':

        currdate = datetime.now()
        c_start = currdate - timedelta(days=30)
        c_start_dt = c_start.strftime(format)
        c_end_dt = currdate.strftime(format)

    '''最近三天'''
    if identi == 'last7':

        currdate = datetime.now()
        c_start = currdate - timedelta(days=7)
        c_start_dt = c_start.strftime(format)
        c_end_dt = currdate.strftime(format)

    '''最近三天'''
    if identi == 'last3':

        currdate = datetime.now()
        c_start = currdate - timedelta(days=3)
        c_start_dt = c_start.strftime(format)
        c_end_dt = currdate.strftime(format)

    '''当天'''
    if identi == 'curr_day':
        now = datetime.now()
        c_start_dt = now.strftime(format)
        c_end_dt = now.strftime(format)

    '''前一天'''
    if identi == 'before_day':

        if start_dt == '' or start_dt == '0':
            currdate = datetime.now()
        else:
            currdate = datetime.strptime(start_dt, format)

        target_day = currdate - timedelta(days=1)

        c_start_dt = target_day.strftime(format)
        c_end_dt = target_day.strftime(format)

    '''后一天'''
    if identi == 'after_day':

        if end_dt == '' or end_dt == '0':
            currdate = datetime.now()
        else:
            currdate = datetime.strptime(end_dt, format)

        target_day = currdate + timedelta(days=1)

        c_start_dt = target_day.strftime(format)
        c_end_dt = target_day.strftime(format)

    '''当前周'''
    if identi == 'curr_week':

        currdate = datetime.now()

        previous_week_start = currdate - timedelta(days=currdate.weekday())
        previous_week_end = previous_week_start + timedelta(days=6)

        c_start_dt = previous_week_start.strftime(format)
        c_end_dt = previous_week_end.strftime(format)

    '''获取前一周的数据'''
    if identi == 'before_week':

        if start_dt == '' or start_dt == '0':
            current_date = datetime.now()
        else:
            current_date = datetime.strptime(start_dt, format)

        # 获取当前日期所在周的第一天
        first_day_of_current_week = current_date - timedelta(days=current_date.weekday())

        # 获取前一周的第一天
        first_day_of_previous_week = first_day_of_current_week - relativedelta(weeks=1)

        # 获取前一周的最后一天
        last_day_of_previous_week = first_day_of_previous_week + timedelta(days=6)

        c_start_dt = first_day_of_previous_week.strftime(format)
        c_end_dt = last_day_of_previous_week.strftime(format)

    '''获取下一周的数据'''
    if identi == 'after_week':
        if end_dt == '' or end_dt == '0':
            current_date = datetime.now()
        else:
            current_date = datetime.strptime(end_dt, format)

        # 获取当前日期所在周的第一天
        first_day_of_current_week = current_date - timedelta(days=current_date.weekday())

        # 获取后一周的第一天
        first_day_of_next_week = first_day_of_current_week + relativedelta(weeks=1)

        # 获取后一周的最后一天
        last_day_of_next_week = first_day_of_next_week + timedelta(days=6)

        c_start_dt = first_day_of_next_week.strftime(format)
        c_end_dt = last_day_of_next_week.strftime(format)

    '''当前月'''
    if identi == 'curr_month':

        current_date = datetime.now()

        # 获取当前日期所在月的第一天
        first_day_of_month = datetime(current_date.year, current_date.month, 1)

        last_day = calendar.monthrange(current_date.year, current_date.month)
        # print(last_day[0], last_day[1])
        last_day_of_month = datetime(current_date.year, current_date.month, last_day[1])

        c_start_dt = first_day_of_month.strftime(format)
        c_end_dt = last_day_of_month.strftime(format)

    if identi == 'before_month':

        if start_dt == '' or start_dt == '0':
            current_date = datetime.now()
        else:
            current_date = datetime.strptime(start_dt, format)

        # 获取当前日期所在月的第一天
        first_day_of_month = datetime(current_date.year, current_date.month, 1)
        before_month_oneday = first_day_of_month - timedelta(days=3)

        # 获取当前日期所在月的第一天
        first_day_of_month = datetime(before_month_oneday.year, before_month_oneday.month, 1)

        last_day = calendar.monthrange(before_month_oneday.year, before_month_oneday.month)
        last_day_of_month = datetime(before_month_oneday.year, before_month_oneday.month, last_day[1])

        c_start_dt = first_day_of_month.strftime(format)
        c_end_dt = last_day_of_month.strftime(format)

    if identi == 'after_month':

        if end_dt == '' or end_dt == '0':
            current_date = datetime.now()
        else:
            current_date = datetime.strptime(end_dt, format)

        last_day = calendar.monthrange(current_date.year, current_date.month)
        # 当前日期所在月的最后一天
        last_day_of_month = datetime(current_date.year, current_date.month, last_day[1])

        next_month_one_day = last_day_of_month + timedelta(days=3)

        # 获取当前日期所在月的第一天
        first_day_of_month = datetime(next_month_one_day.year, next_month_one_day.month, 1)

        last_day = calendar.monthrange(next_month_one_day.year, next_month_one_day.month)
        # print(last_day[0], last_day[1])
        last_day_of_month = datetime(next_month_one_day.year, next_month_one_day.month, last_day[1])

        c_start_dt = first_day_of_month.strftime(format)
        c_end_dt = last_day_of_month.strftime(format)

    return c_start_dt, c_end_dt


"""
获取客户单的UUID
该方法将客户端的编号加入到交易编号中
"""


def client_order_uuid(random_len=20):

    now = datetime.now()
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    p = random.sample(base_str, random_len)
    p = "".join(p)

    time_str = now.strftime('%Y%m%d%H%M%S%f')[:-2]
    return f"{p}"
