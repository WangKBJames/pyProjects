# -*- coding: utf-8 -*-
import cx_Oracle
import numpy as np

main_path = r".\vehicle_load\\"

# 数据库参数
user_name = 'bridge_detection_jy'
code = 'bridge_detection_jy'
ipaddress = '58.213.45.94:2221/orcl'


# t_start_str = '2022-03-12 00:00:00'
# t_end_str = '2022-03-12 23:55:00'


def traffic_stat(t_start_str, t_end_str):
    '''

    :param t_start_str:
    :param t_end_str:
    :return:
    '''

    db = cx_Oracle.connect(user_name, code, ipaddress)  # 连接数据库
    # print(db.version)  # 打印版本看看 显示 11.2.0.1.0
    cur = db.cursor()  # 游标操作

    # 小时车流量
    flow_str = "select substr(t.evt_time,12,2), count(ID) from T_STG_WEIGH_VALUE t where to_date(t.evt_time, " \
               "'YYYY-MM-DD HH24:MI:SS') " \
               f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
               "'YYYY-MM-DD HH24:MI:SS') group by substr(t.evt_time,12,2) order by substr(t.evt_time,12,2)"
    cur.execute(flow_str)  # 执行sql语句
    flow_rows = cur.fetchall()  # 获取数据
    h_list_flow = [int(x[0]) for x in flow_rows]
    flow = [x[1] for x in flow_rows]

    # 各车道车流量
    lane_no_str = "select LANE_NO, count(ID) from T_STG_WEIGH_VALUE t where to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
                  f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
                  "'YYYY-MM-DD HH24:MI:SS') group by LANE_NO order by LANE_NO"
    cur.execute(lane_no_str)  # 执行sql语句
    lane_rows = cur.fetchall()  # 获取数据
    lane_list = [int(x[0]) for x in lane_rows]
    lane_count = [x[1] for x in lane_rows]

    # 小时车速
    velocity_str = "select substr(t.evt_time,12,2), avg(speed) from T_STG_WEIGH_VALUE t where to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
                   f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
                   "'YYYY-MM-DD HH24:MI:SS') group by substr(t.evt_time,12,2) order by substr(t.evt_time,12,2)"
    cur.execute(velocity_str)  # 执行sql语句
    velocity_rows = cur.fetchall()  # 获取数据
    h_list_velocity = [int(x[0]) for x in velocity_rows]
    velocity = [float(x[1]) for x in velocity_rows]

    # 车重分布
    weight_t_str = "select floor(t.total_weight/1000), count(*) from T_STG_WEIGH_VALUE t where to_date(t.evt_time, " \
                   "'YYYY-MM-DD HH24:MI:SS') " \
                   f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}'," \
                   " 'YYYY-MM-DD HH24:MI:SS') group by floor(t.total_weight/1000) order by floor(" \
                   "t.total_weight/1000) "
    cur.execute(weight_t_str)  # 执行sql语句
    weight_dist_rows = cur.fetchall()  # 获取数据
    weight_list = [x[0] for x in weight_dist_rows]
    weight_count = [x[1] for x in weight_dist_rows]

    # 上、下行流量比
    up_count_str = "select count(*) from T_STG_WEIGH_VALUE t where (to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
                   f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
                   "'YYYY-MM-DD HH24:MI:SS')) and (t.LANE_NO <= 3)"
    cur.execute(up_count_str)  # 执行sql语句
    up_count_rows = cur.fetchall()  # 获取数据
    down_count_str = "select count(*) from T_STG_WEIGH_VALUE t where (to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
                     f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
                     "'YYYY-MM-DD HH24:MI:SS')) and (t.LANE_NO >= 4)"
    cur.execute(down_count_str)  # 执行sql语句
    down_count_rows = cur.fetchall()  # 获取数据
    if int(down_count_rows[0][0]) != 0:
        up_down_ratio = int(up_count_rows[0][0]) / int(down_count_rows[0][0])
    else:
        up_down_ratio = 0

    # 客车流量、货车流量、客货比
    car_count_str = "select count(*) from T_STG_WEIGH_VALUE t where (to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
                    f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
                    "'YYYY-MM-DD HH24:MI:SS')) and (t.AXLES_COUNT < 3)"
    cur.execute(car_count_str)  # 执行sql语句
    car_count_rows = cur.fetchall()  # 获取数据
    truck_count_str = "select count(*) from T_STG_WEIGH_VALUE t where (to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
                      f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
                      "'YYYY-MM-DD HH24:MI:SS')) and (t.AXLES_COUNT >= 3)"
    cur.execute(truck_count_str)  # 执行sql语句
    truck_count_rows = cur.fetchall()  # 获取数据
    car_count = int(car_count_rows[0][0])
    truck_count = int(truck_count_rows[0][0])
    if truck_count != 0:
        ct_ratio = car_count / truck_count
    else:
        ct_ratio = 0

    # 超限车辆数
    over_weight_count_str = "select count(*) from T_STG_WEIGH_VALUE t where (to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
                            f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
                            "'YYYY-MM-DD HH24:MI:SS')) and (t.total_weight/1000 > 57)"
    cur.execute(over_weight_count_str)  # 执行sql语句
    over_weight_count_rows = cur.fetchall()  # 获取数据
    over_weight_count = int(over_weight_count_rows[0][0])

    # 最重车
    weight_max_str = "select max(t.total_weight/1000) from T_STG_WEIGH_VALUE t where (to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') " \
                     f"between to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', " \
                     "'YYYY-MM-DD HH24:MI:SS'))"
    cur.execute(weight_max_str)  # 执行sql语句
    weight_max_rows = cur.fetchall()  # 获取数据
    if weight_max_rows[0][0]:
        weight_max = float(weight_max_rows[0][0])
    else:
        weight_max = 0


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
    all_count = int(day_num_rows[0][0]) + int(night_num_rows[0][0])
    if int(day_num_rows[0][0]) != 0:
        day_flow_ratio = int(day_num_rows[0][0]) / (int(day_num_rows[0][0]) + int(night_num_rows[0][0]))
    else:
        day_flow_ratio = 0

    # 高峰小时系数
    count_15 = "select " \
               "max(count(*)) from T_STG_WEIGH_VALUE t where to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') between " \
               f"to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', 'YYYY-MM-DD HH24:MI:SS') " \
               "group by substr(t.evt_time,1,14) || case when floor(to_number(substr(t.evt_time,15,2) / 15)) * 15 = 0 " \
               "then '00' else to_char(floor(to_number(substr(t.evt_time,15,2) / 15)) * 15) end || ':00' order by " \
               "count(*) desc "
    cur.execute(count_15)  # 执行sql语句
    count_15_rows = cur.fetchall()  # 获取数据
    count_60 = "select " \
               "max(count(*)) from T_STG_WEIGH_VALUE t where to_date(t.evt_time, 'YYYY-MM-DD HH24:MI:SS') between " \
               f"to_date('{t_start_str}', 'YYYY-MM-DD HH24:MI:SS') and to_date('{t_end_str}', 'YYYY-MM-DD HH24:MI:SS') " \
               "group by substr(t.evt_time,1,13) " \
               "order by " \
               "max(count(*)) desc "
    cur.execute(count_60)  # 执行sql语句
    count_60_rows = cur.fetchall()  # 获取数据
    if count_15_rows[0][0]:
        peak_hour_coff = float(count_60_rows[0][0]) / float(count_15_rows[0][0]) / 4
    else:
        peak_hour_coff = 0

    db.close()

    return [h_list_flow, flow, lane_list, lane_count, h_list_velocity, velocity, weight_list, weight_count,
            all_count, up_down_ratio, car_count, truck_count, ct_ratio, over_weight_count, weight_max,
            day_flow_ratio, peak_hour_coff]


def process():
    par = np.loadtxt(main_path + r"input\par.txt", dtype=str, delimiter=',')
    t_start_str = par[0]
    t_end_str = par[1]
    h_list_flow, flow, lane_list, lane_count, h_list_velocity, velocity, weight_list, weight_count, all_count, \
    up_down_ratio, car_count, truck_count, ct_ratio, over_weight_count, weight_max, day_flow_ratio, peak_hour_coff \
        = traffic_stat(t_start_str, t_end_str)
    np.savetxt(main_path + r"output\fig1_x_1.txt", h_list_flow)
    np.savetxt(main_path + r"output\fig1_y_1.txt", flow)
    np.savetxt(main_path + r"output\fig2_x_1.txt", lane_list)
    np.savetxt(main_path + r"output\fig2_y_1.txt", lane_count)
    np.savetxt(main_path + r"output\fig3_x_1.txt", h_list_velocity)
    np.savetxt(main_path + r"output\fig3_y_1.txt", velocity)
    np.savetxt(main_path + r"output\fig4_x_1.txt", weight_list)
    np.savetxt(main_path + r"output\fig4_y_1.txt", weight_count)
    if weight_max != 0:
        doc_str = "车辆荷载统计：\n" \
                  "总车流量：{0:d}辆\n" \
                  "上、下行车流量比：{1:.2f}\n" \
                  "客车流量：{2:d}辆\n" \
                  "货车流量：{3:d}辆\n" \
                  "客货比：{4:.2f}\n" \
                  "超载车辆数：{5:d}辆\n" \
                  "最重车辆：{6:.2f}t\n" \
                  "昼日流量比：{7:.2f}\n" \
                  "高峰小时系数：{8:.2f}".format(all_count, up_down_ratio, car_count, truck_count, ct_ratio,
                                          over_weight_count, weight_max, day_flow_ratio, peak_hour_coff)
    else:
        doc_str = "该时段动态承重数据暂未入库！"
    with open(main_path + r"output\shuoming.txt", "w", encoding="utf-8") as f:
        f.write(doc_str)
    return


if __name__ == "__main__":
    # h_list_flow, flow, lane_list, lane_count, h_list_velocity, velocity, weight_list, weight_count, all_count, \
    #     up_down_ratio, car_count, truck_count, ct_ratio, over_weight_count, weight_max, day_flow_ratio, peak_hour_coff \
    #     = traffic_stat(t_start_str, t_end_str)
    #
    # print(peak_hour_coff)
    process()
