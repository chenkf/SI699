import pandas as pd
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import pickle
import json
import math
import numpy as np
import random
import warnings
from sklearn.cluster import DBSCAN
warnings.filterwarnings("ignore")

data = pd.read_csv("data/123",sep = "\t" , header = None)

stop_time_exp = {0: None,
     1: 2892.710037174721,
     2: 6124.4740740740745,
     3: 2637.6167400881059,
     4: 5356.7735042735039,
     5: 3978.1254612546127,
     6: 6468.1999999999998,
     7: 2471.073732718894,
     8: 4996.2790697674418,
     9: 3103.2907801418442,
     10: 4485.6366906474823,
     11: 5937.9452054794519}

stop_order = {
    7:1,
    3:2,
    1:3,
    9:4,
    5:5,
    10:6,
    0:7,
    8:8,
    4:9,
    11:10,
    2:11,
    6:12
}


def removing_invalid(df):
    data = data[(data[4]> 112) & (data[4]<113.5)]
    data = data[(data[5]> 22) & (data[5]<24)]
    seconds = [str(s).zfill(2)  for s in range(60)]
    minutes = [str(s).zfill(2)  for s in range(60)]
    hours = [str(s).zfill(2)  for s in range(24)]
    months = [str(s).zfill(2)  for s in range(1,13)]

    df = df[df[6].str[-2:].isin(seconds)]
    df = df[df[6].str[-5:-3].isin(minutes)]
    df = df[df[6].str[-8:-6].isin(hours)]
    df = df[df[6].str[5:7].isin(months)]
    df = df[df[6].str[:4].isin(["2015"])]
    df = df.reset_index(drop=True)
    df = df.reset_index()
    df.columns = ["raw_index","GPS", "licence_plate_number","route","direction","longitude","latitude",
               "calibrated_time","velocity","angle","raw_time"]
    out_df = pd.DataFrame(columns=["raw_index","GPS", "licence_plate_number","route","direction","longitude","latitude",
               "calibrated_time","velocity","angle","raw_time","lon_cp","lat_cp","time_cp","delta_lon", "delta_lat","delta_time"])

    for plate in df["licence_plate_number"].unique():
        bus_df = df[df["licence_plate_number"] == plate]
        bus_df = bus_df.reset_index(drop = True)
        bus_df.loc[:,"lon_cp"] = bus_df.loc[:,"longitude"]
        bus_df.loc[:,"lat_cp"] = bus_df.loc[:,"latitude"]
        bus_df.loc[:,"time_cp"] = bus_df.loc[:,"calibrated_time"]
        bus_df.lon_cp = bus_df.lon_cp.shift(1)
        bus_df.lat_cp = bus_df.lat_cp.shift(1)
        bus_df.time_cp = bus_df.time_cp.shift(1)
        bus_df.loc[:,"delta_lon"] = bus_df.loc[:,"longitude"] -  bus_df.loc[:,"lon_cp"]
        bus_df.loc[:,"delta_lat"] = bus_df.loc[:,"latitude"] -  bus_df.loc[:,"lat_cp"]
        bus_df.loc[:,"delta_time"] = pd.to_datetime(bus_df.loc[:,"calibrated_time"]) -  pd.to_datetime(bus_df.loc[:,"time_cp"])
        out_df = pd.concat([out_df, bus_df])

    out_df = out_df[["raw_index","route","licence_plate_number","calibrated_time","longitude",
                              "latitude", "delta_lon", "delta_lat","delta_time"]]
    out_df = out_df.set_index("raw_index")
    out_df = out_df.drop_duplicates(subset=["calibrated_time","longitude","latitude"], keep="first")
    out_df = out_df[out_df["delta_time"].isin(list(out_df["delta_time"].value_counts().index[:6]))]
    out_df["delta_time"] = out_df["delta_time"].dt.seconds
    return out_df

def busstop(df):
    df['calibrated_time'] = pd.to_datetime(df['calibrated_time'])
    not_working = []
    for plate in df["licence_plate_number"].unique():
        bus1 = df[df["licence_plate_number"] == plate]
    #     bus1 = bus1.reset_index(drop=True)
        first_index = bus1.index[0]
        last_loc = (bus1.loc[first_index]["longitude"], bus1.loc[first_index]["latitude"])
        caching = []

        for index, row in bus1.iterrows():
            if row["longitude"] == last_loc[0] and row["latitude"] == last_loc[1]:
                caching.append(index)
            else:
                if len(caching) > 1:
                    not_working.append(caching)
                caching = [index]
                last_loc = (row["longitude"], row["latitude"])
    stop_index_15 = [point[0] for point in not_working if (df.loc[point[-1],"calibrated_time"] - df.loc[point[0],"calibrated_time"]).total_seconds() > 15]
    stop_loc_15_df = df.loc[stop_index_15][["longitude","latitude"]]
    stop_loc_15_df = stop_loc_15_df[(stop_loc_15_df["latitude"]>22.8)&
                          (stop_loc_15_df["longitude"]>113.00) &(stop_loc_15_df["longitude"]<113.2)]

    stop_loc_15_df["nearby"] = 0
    for i in stop_loc_15_df.index:

        stop_loc_15_df.loc[i,"nearby"] = stop_loc_15_df[(stop_loc_15_df["longitude"] > (stop_loc_15_df.loc[i,"longitude"] - 0.0002))&
                                                  (stop_loc_15_df["longitude"] < (stop_loc_15_df.loc[i,"longitude"] + 0.0002))&
                                                  (stop_loc_15_df["latitude"] > (stop_loc_15_df.loc[i,"latitude"]-0.0002))&
                                                  (stop_loc_15_df["latitude"] < (stop_loc_15_df.loc[i,"latitude"]+0.0002))].shape[0]

    stop_loc_15_df_unique = stop_loc_15_df.drop_duplicates().sort_values("nearby",ascending = False)
    stop_loc_15_df_unique = stop_loc_15_df_unique[(stop_loc_15_df_unique["nearby"]<10000)&(stop_loc_15_df_unique["nearby"]>200)]
    DB = DBSCAN(eps=0.0005, min_samples=200).fit(stop_loc_15_df_unique[["longitude","latitude"]])
    loc = pd.DataFrame(DB.components_.tolist())
    loc.index = DB.core_sample_indices_
    loc.columns = ["lon","lat"]
    lab = pd.DataFrame(DB.labels_)
    lab.columns = ["label"]
    df = pd.concat([loc, lab], axis=1)
    df = df.dropna(axis=0, how='any')
    stop_loc = df.groupby("label").mean()
    stop = stop_loc[(stop_loc.lon<113.12)&(stop_loc.lon>113.06)]

    return stop_loc, stop

def cutting_shifts():
    complete_df['calibrated_time'] = pd.to_datetime(complete_df['calibrated_time'])
    start_range = [(113.145, 113.147), (23.0205, 23.0225)]
    end_range = [(113.0165, 113.0185), (23.046, 23.048)]

    shifts = []
    for bus in complete_df.licence_plate_number.unique().tolist()[:]:
        temp = []
        temp_bus = complete_df[complete_df.licence_plate_number == bus]
        for i in temp_bus.index:
            if temp_bus.loc[i,"longitude"] < start_range[0][1] and temp_bus.loc[i,"longitude"] > start_range[0][0] and temp_bus.loc[i,"latitude"] < start_range[1][1] and temp_bus.loc[i,"latitude"] > start_range[1][0]:

                temp = [i]
            elif temp_bus.loc[i,"longitude"] < end_range[0][1] and temp_bus.loc[i,"longitude"] > end_range[0][0] and temp_bus.loc[i,"latitude"] < end_range[1][1] and temp_bus.loc[i,"latitude"] > end_range[1][0]:
                if len(temp)> 100 :
                    if (temp_bus.loc[temp[-1],"calibrated_time"] - temp_bus.loc[temp[0],"calibrated_time"]).total_seconds()>3600 and (temp_bus.loc[temp[-1],"calibrated_time"] - temp_bus.loc[temp[0],"calibrated_time"]).total_seconds()<9600 :
                        shifts.append(temp)
                temp = []

            elif len(temp)>0:
                temp.append(i)

    valid_shift = []

    for shift in shifts[:]:
        ind = [s for s in shift]
        timelist = list(pd.to_datetime(complete_df.loc[ind]["calibrated_time"]))
        timelist = [timedelta.total_seconds(t-timelist[0]) for t in timelist if timedelta.total_seconds(t-timelist[0])< 8000 and timedelta.total_seconds(t-timelist[0])> 0]
        platelist = list(complete_df.loc[ind]["licence_plate_number"])
        xlist = list(complete_df.loc[ind]["longitude"])
        ylist = list(complete_df.loc[ind]["latitude"])
        tlist = list(complete_df.loc[ind]["calibrated_time"])
        deltax = list(complete_df.loc[ind]["delta_lon"])
        deltay = list(complete_df.loc[ind]["delta_lat"])
        deltat = list(complete_df.loc[ind]["delta_time"])
        temp = zip(platelist, xlist, ylist, tlist, timelist, deltax, deltay, deltat)

        if max(xlist)>113.2:
            continue
        valid_shift.append(temp)



    agg_df = pd.DataFrame(columns = ["Plate", "Lon", "Lat", "Timestamp", "timepast",
                                     "deltalon", "deltalat", "deltat", "speed", "deltad", "distance",
                                 "average_speed", "speed_smooth_3", "speed_smooth_5", "speed_smooth_10", "shift", "train_test"])

    for shift, shift_data in enumerate(valid_shift):
        shift_df = pd.DataFrame(shift_data)
        shift_df.columns = ["Plate", "Lon", "Lat", "Timestamp", "timepast", "deltalon", "deltalat", "deltat"]
        shift_df["speed"] = ((shift_df["deltalat"]*100000)**2 + (shift_df["deltalat"]*100000)**2).apply(math.sqrt) / shift_df["deltat"]
        shift_df["deltad"] = shift_df["speed"] * shift_df["deltat"]
        shift_df["distance"] = shift_df["deltad"].expanding().sum()
        shift_df["average_speed"] = shift_df["distance"] / shift_df["timepast"]
        shift_df["speed_smooth_3"] = shift_df["speed"].rolling(window=3).mean()
        shift_df["speed_smooth_5"] = shift_df["speed"].rolling(window=5).mean()
        shift_df["speed_smooth_10"] = shift_df["speed"].rolling(window=10).mean()
        shift_df["shift"] = shift
        shift_df["train_test"] = 1 if random.random() > 0.8 else 0
        agg_df = agg_df.append(shift_df, ignore_index = True)

    agg_df["Lon_round"] = agg_df["Lon"].round(4)
    agg_df["Lat_round"] = agg_df["Lat"].round(4)
    grid_df = agg_df[agg_df["train_test"] == 0].groupby(["Lon_round", "Lat_round"])[["timepast", "speed_smooth_3", "speed_smooth_5", "speed_smooth_10"]].mean()
    grid_df = grid_df.reset_index()
    grid_df.columns = ["Lon_round", "Lat_round", "time_avg", "speed_avg_3", "speed_avg_5", "speed_avg_10"]
    new_df = pd.merge(agg_df, grid_df,  how='left', left_on=['Lon_round','Lat_round'], right_on = ['Lon_round','Lat_round'])
    new_df["delay"] = new_df["timepast"] - new_df["time_avg"]
    new_df["delay_12"] = new_df["delay"].shift(12)
    new_df["delay_24"] = new_df["delay"].shift(24)
    new_df["delay_36"] = new_df["delay"].shift(36)
    new_df["delay_48"] = new_df["delay"].shift(48)
    new_df["delay_60"] = new_df["delay"].shift(12)

    stop_dic = {}
    for index, row in stop.iterrows():
        stop_dic[index] = []
    for shift in range(len(valid_shift)):
    #     start = time.time()
        res_df = pd.DataFrame(columns = [u'Plate', u'Lon', u'Lat', u'Timestamp', u'timepast', u'deltalon',
                                         u'deltalat', u'deltat', u'speed', u'deltad', u'distance',
                                         u'average_speed', u'speed_smooth_3', u'speed_smooth_5',
                                         u'speed_smooth_10', u'shift', u'train_test', u'Lon_round', u'Lat_round',
                                         u'time_avg', u'speed_avg_3', u'speed_avg_5', u'speed_avg_10', u'delay',
                                         u'delay_12', u'delay_24', u'delay_36', u'delay_48', u'delay_60', u'timeleft',
                                         u'tostop', u'stoplon', u'stoplat',
                                        ])

        shift_df = new_df[new_df["shift"] == shift]

        stop_info = []

        for index, row in stop.iterrows():
            # 34 meters
            lon_range = [row["lon"] - 0.0002, row["lon"] + 0.0002]
            lat_range = [row["lat"] - 0.0002, row["lat"] + 0.0002]

            for point in shift_df.iterrows():
                if point[1]["Lon"] < lon_range[1] and point[1]["Lon"] > lon_range[0] and point[1]["Lat"] < lat_range[1] and point[1]["Lat"] > lat_range[0]:
                    stop_info.append([index, point[1]["timepast"], point[1]["Lon"], point[1]["Lat"]])
                    if point[1]["train_test"] == 0:
                        stop_dic[index].append(point[1]["timepast"])
                    break

        if len(stop_info) < 5:
            continue

        for st in stop_info:
            sub_df = shift_df[shift_df.timepast<st[1]]
            sub_df["timeleft"] = st[1] - sub_df.timepast
            sub_df["tostop"] = st[0]
            sub_df["stoplon"] = st[2]
            sub_df["stoplat"] = st[3]
            res_df = res_df.append(sub_df, ignore_index = True)
        res_df.to_csv("shifts_116/%s.csv" % shift)


def final_filter(n):
    for shift in range(n):
        try:
            shift_df = pd.read_csv("shifts_116/%s.csv" % shift)
            shift_df["tostop_estimate"] = shift_df["tostop"].apply(lambda x:stop_time_exp[x])
            shift_df["ordinal_stop"] = shift_df["tostop"].apply(lambda x:stop_order[x])
            shift_df.to_csv("shifts_116_valid/%s.csv" % shift)
        except:
            pass


if __name__ == "__main__":
    complete_df = removing_invalid(data)
    stop_loc, stop = busstop(complete_df)
    cutting_shifts()
    final_filter(500)
