from flask import render_template,request,session
import datetime
from config.cfg import Syscfg
cfg = Syscfg()
from common.log import logservice
logger = logservice.getLog("")


class webutil:


    def __init__(self):
        pass

    # 根据列表生成下拉框
    @classmethod
    def htmlSelect(self,datalist,selectname,mode,selectitem,attrib,pairindex):

        attrib = attrib + " class=\"form-control \" ";
        if mode == '':
            mode = 'Normal'
        else:
            mode = 'Reverse'

        htmlselectstr = ""

        if  attrib.find('multiple')!=-1  :
            htmlselectstr = f"\n  <select name='{selectname}[]'  id='{selectname}'"
        else:
            htmlselectstr = f"\n  <select name='{selectname}'  id='{selectname}'"

        htmlselectstr = htmlselectstr + attrib + ">\n"

        if  len(datalist) == 0:
            htmlselectstr = htmlselectstr+"<option selected label=\"null\" value=\"\" style=\"color:#AAAAAA;\">--------</option>"
        else:
            for item in datalist:
                text = item.get('text')
                value = item['value']

                if value == selectitem:
                    htmlselectstr = htmlselectstr + f"<option value='{value}' selected='selected'>{text}</option>\n"
                else:
                    htmlselectstr = htmlselectstr + f"<option value='{value}'>{text}</option>\n"

        htmlselectstr = htmlselectstr + f"\n</select>\n"

        return  htmlselectstr


    # 通用模板渲染方法
    @classmethod
    def render_template(self, viewpath, data={}):

        # 将url的参数转化成字典表
        url_query_arr = request.query_string.decode('utf-8').split('&')
        url_parm_map = {}
        for item in url_query_arr:
            if item == '': continue
            item_key_value = item.split("=")
            url_parm_map[item_key_value[0]] = item_key_value[1]

        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d")
        data['sys_date'] = date_time
        data['http_request'] = request
        data['http_session'] = session

        # searchparm = request.args["searchparm"]
        # url_parm_map['searchparm'] = searchparm
        data['_GET'] = url_parm_map;
        data['login_user_name'] = session.get('login_user_name','')
        data['session'] = session

        if data.get('formdata') is  None:
            data['formdata']  = {}

        login_user_pkid = session.get("login_user_pkid",0)
        adminuser = cfg.getCurrentadmin()
        adminuser_arr = adminuser.split(',')
        logger.debug(f" adminuser  {adminuser_arr}")

        flag_source_pkid = session.get("source_pkid",0)
        data['source_pkid'] = flag_source_pkid

        flag_sample_pkid = session.get("sample_pkid", 0)
        data['sample_pkid'] = flag_sample_pkid


        logger.debug(f" login_user_pkid => {login_user_pkid}   adminuser_arr => {adminuser_arr} ")

        if str(login_user_pkid) in adminuser_arr:
            data['mifadin'] = '1'
        else:
            data['mifadin'] = '0'

        data['runmodel'] = session['runmodel']

        return render_template(viewpath, viewdata=data)