from .mysqlconn import UsingMysql
from  common.log import logservice
logger = logservice.getLog("")


# 新增加一条记录
def saveRow(tablename, columnValues):

    addSqlParams = []
    updatesqlcolumn = []
    updatesqlflag = []

    for key, value in columnValues.items():
        addSqlParams.append(key)
        updatesqlcolumn.append(value)
        updatesqlflag.append("%s")

    addSqlParamsstr = ','.join(addSqlParams)
    updatesqlflagstr = ','.join(updatesqlflag)

    addSql = f"INSERT INTO {tablename}( {addSqlParamsstr} ) VALUES( {updatesqlflagstr}  )"

    logger.debug(addSql)

    with UsingMysql(log_time=True) as um:
        um.cursor.execute(addSql, updatesqlcolumn)
        return um.cursor.lastrowid


# 根据主键更新一条记录
def updateRowByPk(tablename, columnAndValue, pkName, pkValue):

    updateSqlParams = []
    updateParms = []

    updateparm = " set 1=1 "

    for key, value in columnAndValue.items():
        updateSqlParams.append(value)
        updateParms.append(f"{key}=%s")

    updatewhereparm = " (1=1)  "
    searchparm = {pkName: pkValue}


    # for key, value in searchparm.items():
    #     updateSqlParams.append(value)
    #     updatewhereparm = updatewhereparm + f" and {key} = %s "

    updateSqlParams.append(pkValue)

    updateParmsStr = ','.join(updateParms)
    addSql = f"UPDATE {tablename} set {updateParmsStr} WHERE {pkName} = %s"

    logger.debug(addSql)

    with UsingMysql(log_time=True) as um:
        um.cursor.execute(addSql, updateSqlParams)


## 根据查询条件
def updateRow(tablename, columnAndValue, condition):

    updateSqlParams = []
    updateParms = []

    updateparm = " set 1=1 "

    for key, value in columnAndValue.items():
        updateSqlParams.append(value)
        updateParms.append(f"{key}=%s")

    updatewhereparm = " (1=1)  "
    searchparm = condition

    for key, value in searchparm.items():
        updateSqlParams.append(value)
        updatewhereparm = updatewhereparm + f" and {key} = %s "

    updateParmsStr = ','.join(updateParms)
    addSql = f"UPDATE {tablename} set {updateParmsStr} WHERE  {updatewhereparm}"


    logger.debug(addSql)

    with UsingMysql(log_time=True) as um:
        um.cursor.execute(addSql, updateSqlParams)


# 根据sql语句更新一条记录
def updateBySql(updateSql):
    logger.debug(updateSql)
    with UsingMysql(log_time=True) as um:
        um.cursor.execute(updateSql)


## 根据主键获取一条记录
def getRowByPk(tablename, column="*", pkColumn="", value=""):

    sql = f"select {column} from {tablename} where {pkColumn} = %s"
    logger.debug(sql)
    params = value

    with UsingMysql(log_time=True) as um:
        um.cursor.execute(sql, params)
        result_data = um.cursor.fetchone()
        return result_data


##根据主键获取一条记录
def getRowByPkOne(sql):

    logger.debug(sql)
    with UsingMysql(log_time=True) as um:
        um.cursor.execute(sql)
        result_data = um.cursor.fetchone()
        return result_data

#根据sql语获取数据数目
def getRowCount4Sql(sql, numscolumn):

    logger.debug(sql)
    with UsingMysql(log_time=True) as um:
        um.cursor.execute(sql)
        result_data = um.cursor.fetchone()
        return result_data[numscolumn]


#根据特定表的查询条件获取记录集
def getRowsByCondition(tablename, column, condition, orderBy, limit):

    if column == None or column == '':
        column = "*"

    search_where_parm = " (1=1)  "
    search_value_parm = []
    addSqlParams = []

    for key, value in condition.items():
        search_value_parm.append(value)
        search_where_parm = search_where_parm + f" and {key}=%s"

    search_sql = f" select {column} from {tablename}  where {search_where_parm} {orderBy}  {limit}  "

    logger.debug(search_sql)
    with UsingMysql(log_time=True) as um:
        um.cursor.execute(search_sql,search_value_parm)
        result_data = um.cursor.fetchall()
        return result_data


#根据sql语句获取数据集
def getRowsBySQl(sqlchareter, condition, orderBy, limit):

    search_where_parm = " (1=1)  "
    search_value_parm = []


    for key, value in condition.items():
        search_value_parm.append(value)
        searchwhereparm = searchwhereparm + f" and {key}=%s"

    search_sql=""

    if sqlchareter.find('where') != -1 or sqlchareter.find('Where') != -1 or sqlchareter.find('WHERE') != -1 :
        search_sql = f" {sqlchareter} and {search_where_parm}  {orderBy}   {limit}"
    else:
        search_sql = f" {sqlchareter} where {search_where_parm}  {orderBy}  {limit}"

    logger.debug(search_sql)

    with UsingMysql(log_time=True) as um:
        um.cursor.execute(search_sql)
        result_data = um.cursor.fetchall()
        return result_data

#根据完整的sql语句获取记录集
def getRowsByFullSQl(sql):
    logger.debug(sql)
    with UsingMysql(log_time=True) as um:
        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()
        return result_data


#根据查询条件和分页条件查询数据
def getRowsBySQlNoCondtion(sqlchareter, limit):
    logger.debug(sqlchareter)
    sel_sql = f" {sqlchareter} limit {limit} "
    logger.debug(sel_sql)
    with UsingMysql(log_time=True) as um:
        um.cursor.execute(sel_sql)
        result_data = um.cursor.fetchall()
        return result_data

#根据sql语句查询一条数据
def getRowsBySQlCase(sql):

    logger.debug(sql)
    with UsingMysql(log_time=True) as um:
        um.cursor.execute(sql)
        result_data = um.cursor.fetchone()
        return result_data


#根据SQL语句查询键值对
def getSQL2KeyValueMap(sql, keyClm, valueClm):

    logger.debug(sql)
    with UsingMysql(log_time=True) as um:

        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()
        result_map = {}

        for item in result_data:
            key = item[keyClm]
            value = item[valueClm]
            result_map[key] = value

        return result_map


#根据sql获取指定字段和主键对应的MAP
def getSQL2Map4Value(sql, key,valuekey):

    logger.debug(sql)
    with UsingMysql(log_time=True) as um:
        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()

        result_map = {}
        for item in result_data:
            key1 = item[key]
            value1 = item[valuekey]
            result_map[key1] = value1

        return result_map

#根据sql获取指定字段和主键对应的MAP
def getSQL2Map(sql, key):

    logger.debug(sql)
    with UsingMysql(log_time=True) as um:
        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()

        result_map = {}
        for item in result_data:
            key1 = item[key]
            result_map[key1] = item

        return result_map


# 根据SQL语句获取某个字段值的数组
def getSQL2Arr(sql, key):
    logger.debug(sql)
    with UsingMysql(log_time=True) as um:
        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()

        result_array = []
        for item in result_data:
            value = item[key]
            result_array.append(value)

        return result_array


# 根据SQL语句的查询结果，根据指定的字段的值进行过滤，以指定字段的值作为key，该字段的对应的结果集列表为value，组成一个map
def getSQL2Map4Arr(sql, key):
    logger.debug(sql)
    with UsingMysql(log_time=True) as um:

        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()
        result_map = {}

        for item in result_data:

            value = item[key]
            temp_array = result_map.get(value)

            if (temp_array == None )  :
                temp_array =[]
                temp_array.append(item)
                result_map[value] = temp_array
            else:
                temp_array.append(item)
                result_map[value] = temp_array

        return result_map


# 根据SQL语句的查询结果，根据指定的字段的值进行过滤，以指定字段的值作为key，map的值为对象
def getSQL2Map4bean(sql, key):
    logger.debug(sql)
    with UsingMysql(log_time=True) as um:

        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()
        result_map = {}
        print(result_data)
        for item in result_data:

            value = item[key]
            result_map[value] = item

        return result_map



# 根据SQL语句的查询结果，根据指定的字段的值进行过滤，以指定字段的值作为key，该字段的对应的结果集列表为value，组成一个map
def getSQL2Map4keyArr(sql, key,valuekey):
    logger.debug(sql)
    with UsingMysql(log_time=True) as um:

        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()
        result_map = {}

        for item in result_data:

            value = item[key]
            temp_array = result_map.get(value)

            if (temp_array == None )  :
                temp_array =[]
                temp_array.append(item[valuekey])
                result_map[value] = temp_array
            else:
                temp_array.append(item[valuekey])
                result_map[value] = temp_array

        return result_map


# 根据SQL语句的查询结果，根据两个key字段和一个取值字段构双级别的MAP
def getSQL2Map2Map(sql, key,childkey,valuekey):
    logger.debug(sql)
    with UsingMysql(log_time=True) as um:

        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()
        result_map = {}

        for item in result_data:

            key_value = item[key]
            child_key_value = item[childkey]
            valuekey_value = item[valuekey]

            temp_map = result_map.get(key_value)

            if (temp_map == None )  :
                temp_map ={}
                temp_map[child_key_value] = valuekey_value
                result_map[key_value] = temp_map
            else:
                temp_map[child_key_value] = valuekey_value
                result_map[key_value] = temp_map

        return result_map


# 根据SQL语句的查询结果，根据两个key字段和一个取值字段构双级别的MAP
def getSQL2Map2Map4ayy(sql, key,childkey):
    logger.debug(sql)
    with UsingMysql(log_time=True) as um:

        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()
        result_map = {}

        for item in result_data:

            key_value = item[key]
            child_key_value = item[childkey]


            temp_map = result_map.get(key_value)

            if (temp_map == None):
                temp_map = {}
                temp_map[child_key_value] = [item]
                result_map[key_value] = temp_map
            else:
                array_temp = temp_map.get(child_key_value,None)
                if array_temp == None:
                    array_temp = []
                    array_temp.append(item)
                    temp_map[child_key_value] = array_temp
                else:
                    array_temp.append(item)
                    temp_map[child_key_value] = array_temp

                result_map[key_value] = temp_map

        return result_map


#获取sql语句返回值中某个字段所对应值的数组
def getSQL2KeyValueArr(sql, key,ifrepeat=False):
    logger.debug(sql)
    with UsingMysql(log_time=True) as um:

        um.cursor.execute(sql)
        result_data = um.cursor.fetchall()
        result_array = []

        for item in result_data:

            value = item[key]

            if( value != None and value != '' ):
                if ifrepeat:
                    if(  not value in result_array  ):
                        result_array.append(value)
                else:
                    result_array.append(value)
        return result_array
