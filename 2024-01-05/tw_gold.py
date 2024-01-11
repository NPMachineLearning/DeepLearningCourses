import mysql.connector as mysql

conn = mysql.connect(
    host="mahaljsp.ddns.net",
    user="lcc",
    password="lcc0507",
    database="cloud"
)

cursor = conn.cursor()
cmd = "select * from 台銀黃金 order by 日期"
cursor.execute(cmd)
rs = cursor.fetchall()

for r in rs:
    print(r)