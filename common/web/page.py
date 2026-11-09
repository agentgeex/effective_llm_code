from flask import request
from common.db.mysql import mysqlservice
import math
from common.web.webutil import webutil
import base64
from common.log import logservice
logger = logservice.getLog("")


class page:

    _nopageurl:str
    '''
    记录总数。
     @ var
    int
    '''
    _RecTotal:int

    '''
    每页显示的记录数。
     @ var
    int
    '''
    _RecPerPage:int

    '''
    分页总数。
     @ var
    int
    '''
    _PageTotal:int

    '''
    当前页面编号。
     @ var
    int
    '''
    _PageID:int

    '''
    链接地址，此地址包含分页信息，比如PageID，PageTotal等。
     @ var
    string
    '''
    _URL=""

    '''
    用于前台跳转用的URL地址，此地址里面不包含分页信息。
     @ var
    string
    '''
    _PartURL = ""

    '''
    表名。
     @ var
    string
    '''
    _TableName:str

    '''
    查询条件。
     @ var
    string
    '''
    _Where:str

    _GET={}

    # 初始化方法
    def __init__(self,TableName, RecTotal , RecPerPage , PageID , Where , URL,URL_QUERY ):

        # 将url的参数转化成字典表
        url_query_arr = URL_QUERY.decode('utf-8').split('&')
        for item in url_query_arr:
            if item == '': continue
            item_key_value = item.split("=")
            self._GET[item_key_value[0]] = item_key_value[1]

        self._TableName = TableName
        self._Where = Where

        self._RecTotal = RecTotal if RecTotal > 0 else self.setRecTotal()

        if( int(self._GET.get('RecPerPage',0)) >0 ):
            self._RecPerPage = int(self._GET['RecPerPage'])
        elif( RecPerPage > 0   ):
            self._RecPerPage = RecPerPage
        else:
            self._RecPerPage = 20

        self.setPageTotal();
        self.setPageID(PageID);

        self.setURL(URL);

        self._nopageurl = URL

        if( URL.find("?") == -1 ):
            self._URL  = self._URL+ "?RecTotal=" + str( self._RecTotal) + f"&RecPerPage={self._RecPerPage}&"
            self._PartURL  += self._PartURL+"?RecTotal="+ str(self._RecTotal)+"&"
        else:
            self._URL = self._URL + "&RecTotal=" + str(self._RecTotal) + f"&RecPerPage={self._RecPerPage}&"
            self._PartURL += self._PartURL + "&RecTotal=" + str(self._RecTotal) + "&"







    # 设定当前页码
    def setPageID(self,PageID):

        PageID_1 = int(PageID)
        #PageID_get = request.args.get('PageID')

        logger.debug(f" 当前页面的值为  {PageID}")

        #PageID = PageID > 0 ? PageID: $_GET['PageID'];
        if( PageID_1 <= 0 ):
            PageID_1 = int(request.args.get('PageID',0))

        logger.debug(f" PageID_1 的值为   {request.args.get('PageID')}")


        if (PageID_1 > 0 and PageID_1 <= self._PageTotal):
            self._PageID = PageID_1
        else:
            self._PageID = 1

    #设定当前总页数
    def setPageTotal(self):
        self._PageTotal = math.ceil(int(self._RecTotal) / int(self._RecPerPage));
        return self._PageTotal;



    #显示分页内容
    def show(self,Align = "left"):
        pass

        FirstPage = "首页"
        EndPage = "尾页";
        PrePage = "前页";
        NextPage = "后页";

        '''/ *如果页码总数为零，则直接退出。 * /'''
        if ( self._RecTotal == 0 ):
            return f" <div class=\"pull-right\">结果</div>"

        PageURL = "'"+self._PartURL+ "PageID=' + PageID + '&RecPerPage=' + RecPerPage";

        RecPerPageRange = [

            {'text':'10','value':10},
            {'text': '20', 'value': 20},
            {'text': '30', 'value': 30},
            {'text': '50', 'value': 50},
            {'text': '100', 'value': 100},

        ]

        PageURL = "'"+ self._PartURL+"PageID=' + PageID + '&RecPerPage=' + RecPerPage";
        select_att = "class=\"form-control input-xs input-sm input-inline\"";
        #RecPerPageList = htmlSelect($RecPerPageRange, "_RecPerPage", "Reverse", $this->_RecPerPage, "onchange='setSelect(this);submitPage(\"changeRecPerPage\");' $select_att");

        webutil1 = webutil()
        RecPerPageList = webutil1.htmlSelect(RecPerPageRange,'_RecPerPage','',self._RecPerPage,f"onchange='setSelect(this);submitPage(\"changeRecPerPage\");' {select_att}",'')

        LinkString = """
        
        <script language='Javascript'>
        
		function setSelect(obj) {
		
		    var array = document.getElementsByName('_RecPerPage');
		
		    for (var i = 0; i < array.length; i++) {
		        array[i].value = obj.value;
		    }
		}
		
		function setText(obj)
		
		{
		    var array = document.getElementsByName('_PageID');
		    for (var i = 0; i < array.length; i++) {
		        array[i].value = obj.value;
		    }
		}
        
        function submitPage(mode)
        {
            PageTotal     = parseInt(document.getElementById('_PageTotal').value);
            if(mode == 'changePageID')
            {
                PageID        =  document.getElementById('_PageID').value;
                if(PageID > PageTotal)
                {
                    PageID = PageTotal;
                }
                else if(PageID < 1)
                {
                    PageID = 1;
                }
            }
            else if(mode == 'changeRecPerPage')
            {
                PageID = 0;
            }
            RecPerPage    = document.getElementById('_RecPerPage').value;
            url_str = %s;
            location.href = url_str ;
        }
        </script>

        
        """
        LinkString = LinkString % (PageURL)


        LinkString  = LinkString + f"总数:<strong>{self._RecTotal}</strong>&nbsp;";
        LinkString  = LinkString + f"{RecPerPageList}[条/页]]&nbsp;";
        LinkString  = LinkString + f"页码:<strong>{self._PageID}</strong>/<strong>{self._PageTotal}</strong>&nbsp;";

        '''/* 生成第一页的链接。 */'''
        if (self._PageID == 1):
            LinkString =LinkString+ self.linktxt(FirstPage,"fristButton")
        else:
            LinkString =LinkString+ self.link(FirstPage, self._URL+"PageID=1","fristButton")


        '''/* 生成前页的链接。*/'''
        if (self._PageID == 1):
            LinkString =LinkString+ self.linktxt(PrePage,"prevButton")
        else:
            LinkString =LinkString+ self.link(PrePage, self._URL + "PageID="+str(self._PageID - 1),"prevButton")


        '''/* 生成后页的链接。*/'''
        if (self._PageID == self._PageTotal):
            LinkString =LinkString+ self.linktxt(NextPage , "nextButton")

        else:
            LinkString =LinkString+ self.link(NextPage, self._URL+ "PageID="+str(self._PageID + 1),"nextButton")


        '''/* 生成最后一页的链接。*/'''
        if (self._PageID == self._PageTotal):
            LinkString =LinkString+ self.linktxt(EndPage,"lastButton")
        else:
            LinkString = LinkString+ self.link(EndPage, self._URL+"PageID="+str(self._PageTotal),"lastButton")


        #生成手工跳转文本框
        LinkString = LinkString + f"<input type='hidden' id='_PageTotal' value='{self._PageTotal}' /><input type = 'text' id = '_PageID' value = '{self._PageID}' style = 'text-align:center;width:30px;' class ='pagination-panel-input form-control input-sm input-inline input-mini' /><button type =\"button\" class=\"btn green btn-sm\" onclick='submitPage(\"changePageID\");' >跳转</button>";

        page_str = f"<div id='Pagination' style='float:{Align}; clear:none' class='pull-right pagination-panel'>{LinkString}</div>";

        return page_str




    #生成SQL语句的limit部分
    def Limit(self):
        return " LIMIT "+self.LimitNum();

    # 获取分页
    def LimitNum(self):
        '''/ *保证最后一页查出的记录数为RecPerPage。加上不等于1的限制是为了防止出现limit - 20, 10
        这样的情况。 * /'''
        if ( int(self._RecTotal) < int(self._RecPerPage) ):
            Limit = "0,"+ str(self._RecTotal);
        else:
            Limit = str(int(self._RecPerPage) * (int(self._PageID) - 1)) + "," + str(self._RecPerPage)

        return Limit


    # 获取分页方法2
    def LimitNum1(self):

        Limit_arr = [];

        '''/ *保证最后一页查出的记录数为RecPerPage。加上不等于1的限制是为了防止出现limit - 20, 10
        这样的情况。 * /'''
        if (self._RecTotal < self._RecPerPage):

            Limit = "0,". self._RecTotal
            Limit_arr['from'] = "0"
            Limit_arr['size'] = self._RecTotal
        else:

            Limit =  self._RecPerPage * (self._PageID - 1)+"," + self._RecPerPage
            ",". self._RecPerPage;
            Limit_arr['from'] = self._RecPerPage * (self._PageID - 1) +"," + self._RecPerPage
            Limit_arr['size'] = self._RecPerPage;


        return Limit_arr;


    #获取当前分页表的总页数
    def setRecTotal(self):
        SQL = f"SELECT COUNT(*) AS RecTotal FROM {self._TableName}  {self._Where}"
        logger.debug(SQL)
        Data = mysqlservice.getRowByPkOne(SQL)
        self._RecTotal = Data['RecTotal']
        return self._RecTotal


    #返回链接
    def linktxt(self,text,elementid="id"):
        return f"<a href=\"#\" class=\"btn btn-default disabled btn-sm\" style=\"margin-right: 2px;\" id=\"{elementid}\">{text}</a>"


    def link(self,text,url,elementid="id"):
        return f"<a href=\"{url}\" class=\"btn green btn-sm\" style=\"margin-right: 2px;\" id=\"{elementid}\">{text}</a>"

    def getDetailInfo(self):
        pass

    # 获取用于分页的url字符串
    def getSearchUrl(self):

        searchurl = ""
        if (self._nopageurl.find("?") == -1):
            searchurl = self._nopageurl + "?RecTotal=" + str(self._RecTotal) + f"&RecPerPage={self._RecPerPage}"
        else:
            searchurl = self._nopageurl + "&RecTotal=" + str(self._RecTotal) + f"&RecPerPage={self._RecPerPage}"
        return searchurl


    def getSearchUrl4Base64(self):

        if (self._nopageurl.find("?") == -1):
            search_url = self._nopageurl + "?RecTotal="+str(self._RecTotal)+"&RecPerPage="+str(self._RecPerPage)+"&PageID="+str(self._PageID)
        else:
            search_url = self._nopageurl + "&RecTotal="+str(self._RecTotal)+"&RecPerPage="+str(self._RecPerPage)+"&PageID="+str(self._PageID)

        #logger.debug(f" 查询url  -> {search_url} ")
        return base64.b64encode(search_url.encode('utf-8')).decode("utf-8")


    def getPageUrl(self):
       return "'"+ self._PartURL+ "PageID=' + PageID + '&RecPerPage=' + RecPerPage";

    ### 以下暂时无需实现

    # 设定当前总页数
    def setURL(self,url):
        self._URL = url