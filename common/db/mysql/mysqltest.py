from common.db.mysql import mysqlservice
from pymysql.err import IntegrityError

#results = mysqlservice.getRowsByCondition('camera','*',{"id":10},'order by id asc','')

# reuslts1 = mysqlservice.getRowsBySQl("select * from camera where id > 10",{},'','')

# 通过唯一建来避免在外部进行唯一性的校验
def testunique():
    bian_symbol ={
                    'symbol':'1232134',
                   
            }



    for index in range(5):
        try:
            mysqlservice.saveRow('bian_symbol',bian_symbol)
        except IntegrityError as e:
            print(f" err >  {index}  ->   {bian_symbol['basemd5name']}  是重复的    {e}   <<<<<")
            continue



def testgetrow():
    testbean = mysqlservice.getRowsByFullSQl(f" select * from mat_model where pkid = 100000")
    if  len(testbean) == 0 :
        print( f" result is null")
    print(testbean)

def testsaverow_result():
    mat_testcase_rel_testsamplelib = {
                                        'testcase_id': 1,
                                        'mark_sample_id': 1,
                                        'usertype': '1'
                                    }
    pkid = mysqlservice.saveRow('mat_testcase_rel_testsamplelib',mat_testcase_rel_testsamplelib)
    print(pkid)



testunique()
#testgetrow()
#testsaverow_result()


print("2023-08-08-14-52-44_5.jpg".endswith(".jpg"))
print(len("2023-08-08-14-52-44_5.jpg"))