import multiprocessing
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config.cfg import Syscfg
from common.log import logservice
cfg = Syscfg()
logger = logservice.getLog("")

"""
发送邮件
@param receiver_email 接收者邮箱地址
@param subject 邮件主题
@param body 邮件正文
"""
def sendmail(receiver_email,subject,body):

    configinfo = cfg.getConfigInfo()
    mailconfig = configinfo['sys']['mail']

    # 登录信息
    sender_email = mailconfig['smtp_user']   # 账号走 config.json / 环境变量，值不入库
    password = mailconfig['smtp_passwd']     # 口令走 config.json / 环境变量，值不入库（原注释里的明文口令已清除）
    # SMTP服务器配置
    smtp_server = mailconfig['smtp_server']  #"smtp.163.com"
    smtp_port = mailconfig['smtp_port']

    #receiver_email = "411321681@qq.com"

    # 邮件主题和正文
    #subject = "你好,来自conigeek的python问候"
    #body = "这是来自coingeek的python的问候"

    # 创建MIME多部分消息对象
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # 添加邮件正文
    message.attach(MIMEText(body, "plain"))
    server = smtplib.SMTP(smtp_server, smtp_port)

    # 连接到SMTP服务器并发送邮件
    try:
        server.starttls()  # 启用TLS加密
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        logger.debug("邮件发送成功!!!!!! YES! YES! YES! YES! YES! YES!")
    except Exception as e:
        logger.error(f" find error {e} ")
        logger.error(f"{e.__traceback__}")
        logger.error(e, exc_info=True)
    finally:
        logger.debug("邮件发送结束!!!!!")
        server.quit()


"""
异步发送邮件，防止进程被阻塞
@param receiver_email 接收者邮箱地址
@param subject 邮件主题
@param body 邮件正文
"""
def sendmail_aynsc(receiver_email,subject,body):
    process_customer = multiprocessing.Process(target=sendmail,args=(receiver_email,subject,body))
    process_customer_name = f"sendmail"
    process_customer.name = f"{process_customer_name}"
    process_customer.start()
    #pass



if __name__ == '__main__':
    sendmail_aynsc("411321681@qq.com","来自coingeek系统API的问候","来自coingeek系统API的问候")
    for i in range(100):
        print(i)
    pass        