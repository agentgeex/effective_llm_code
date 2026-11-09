from jinja2 import Environment, FileSystemLoader
from config.cfg import Syscfg
cfg = Syscfg()
APP_ROOT = cfg._app_root_dir
# 设置模板环境，指定模板文件夹
env = Environment(loader=FileSystemLoader(f'{APP_ROOT}/tpl'))


def getContant_tpl(tpl_file,datadict):
    # 加载模板
    template = env.get_template(tpl_file)
    # 渲染模板
    output = template.render(datadict)
    return output
