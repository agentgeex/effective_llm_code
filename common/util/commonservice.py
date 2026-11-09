'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
通用服务模块
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from common.db.mysql import mysqlservice
import os
from common.log import logservice
from datetime import datetime,timedelta



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
添加系统进程运行的状态
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def crate_model_run_report(pro_name,process_annotation,pro_cmt):
    currentdatetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    system_process_run_report = {
                                'process_name': pro_name,
                                'process_cmt': pro_cmt,
                                'process_annotation': process_annotation,
                                'start_dt': currentdatetime,
                                'startdt': currentdatetime,
                                # 'end_dt': currentdatetime,
                                # 'enddt': currentdatetime,
                                'process_pid': os.getpid(),
                                'statuss': 0,
                                }

    pkid = mysqlservice.saveRow('system_process_run_report',system_process_run_report)

    return pkid


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
更新系统进程运行的状态
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def update_mode_run_report(pro_pkid,statuss=1,run_result=''):

    currentdatetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    system_process_run_report = {
                                'end_dt': currentdatetime,
                                'enddt': currentdatetime,
                                'statuss': statuss,
                                }

    mysqlservice.updateRowByPk('system_process_run_report',system_process_run_report,'pkid',pro_pkid)

    if run_result != None and run_result != '':
        update_sql = f"update system_process_run_report set run_result = CONCAT(COALESCE(run_result,''),'{run_result}') where pkid = {pro_pkid}"
        mysqlservice.updateBySql(update_sql)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
更新系统进程运行的状态
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def update_mode_run_report4pid(run_result=''):

    currentdatetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pid = os.getpid()
    system_process_run_report = {
                                'end_dt': currentdatetime,
                                'enddt': currentdatetime,
                                'run_result':run_result
                                }

    mysqlservice.updateRow('system_process_run_report', system_process_run_report, {'process_pid': pid})


