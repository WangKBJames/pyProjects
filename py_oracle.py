import cx_Oracle

db = cx_Oracle.connect('bridge_detection_jy', 'bridge_detection_jy', ' 58.213.45.94:2221/orcl')  # 连接数据库
print(db.version)  # 打印版本看看 显示 11.2.0.1.0
cur = db.cursor()  # 游标操作

t_start_str = '2022-03-12 00:00:00'
t_end_str = '2022-03-12 23:55:00'
flow_str = "select substr(t.evt_time,12,2), count(ID) from T_STG_WEIGH_VALUE t where to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
              f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
              "'YYYY-MM-DD HH24:MI:SS') group by substr(t.evt_time,12,2) order by substr(t.evt_time,12,2)"
cur.execute(flow_str)  # 执行sql语句
flow_rows = cur.fetchall()  # 获取数据

lane_no_str = "select LANE_NO, count(ID) from T_STG_WEIGH_VALUE t where to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
              f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
              "'YYYY-MM-DD HH24:MI:SS') group by LANE_NO order by LANE_NO"
cur.execute(lane_no_str)  # 执行sql语句
lane_rows = cur.fetchall()  # 获取数据

velocity_str = "select substr(t.evt_time,12,2), avg(speed) from T_STG_WEIGH_VALUE t where to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
              f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
              "'YYYY-MM-DD HH24:MI:SS') group by substr(t.evt_time,12,2) order by substr(t.evt_time,12,2)"
cur.execute(velocity_str)  # 执行sql语句
velocity_rows = cur.fetchall()  # 获取数据

# 昼日流量比
day_num_str = "select count(ID) from T_STG_WEIGH_VALUE t where (to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
              f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
              "'YYYY-MM-DD HH24:MI:SS')) and (to_number(substr(t.evt_time,12,2)) between 6 and 18) "
cur.execute(day_num_str)  # 执行sql语句
day_num_rows = cur.fetchall()  # 获取数据
night_num_str = "select count(ID) from T_STG_WEIGH_VALUE t where (to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
              f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
              "'YYYY-MM-DD HH24:MI:SS')) and (to_number(substr(t.evt_time,12,2))<6 or to_number(substr(t.evt_time,12," \
              "2))>18) "
cur.execute(night_num_str)  # 执行sql语句
night_num_rows = cur.fetchall()  # 获取数据
day_flow_ratio = int(day_num_rows[0][0]) / (int(day_num_rows[0][0]) + int(night_num_rows[0][0]))

# 高峰小时系数

db.close()


# 打印数据
# for row in day_num_rows:
    # print(f"{row[0]} ", end='\n')
print(day_flow_ratio, end='\n')
