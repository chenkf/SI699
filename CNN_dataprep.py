import pandas as pd
import numpy as np
import pickle


with open('data/shifts_new_116.pickle', 'rb') as handle:
    df = pickle.load(handle)


missing_columns = df.columns[df.isnull().any()].tolist()
df = df.dropna(subset=missing_columns,how='any')

shifts_list = df["shift"].unique().tolist()
stops_list = df["tostop"].unique().tolist()
# shifts_list
# stops_list
df_out_train = pd.DataFrame()
df_out_test = pd.DataFrame()
count = 0
for shift in shifts_list:
    print (count)
    count +=1
    for stop in stops_list:
        df_sub = (df[(df["shift"]==shift) & (df["tostop"] == stop)])
        try:
            test = df_sub["train_test"].tolist()[0]
        except:
            continue
        df_sub = df_sub[["Lon", "Lat", "hour", "weekday", "speed", "timepast", "delay", "stoplon", "stoplat", "timeleft"]]
        df_sub = df_sub.reset_index().drop("index", axis = 1)
        df_out = pd.DataFrame()
        for index, row in df_sub.iterrows():
            if index < 10:
                continue
            new_row = []
            df_row = df_sub.loc[index - 9:index][["Lon", "Lat", "hour", "weekday", "speed", "timepast", "delay", "stoplon", "stoplat"]].as_matrix()
            df_row =  pd.DataFrame(df_row.reshape((1,90)))

            if test == 0:
                df_out_train = pd.concat([df_out_train, df_row] , ignore_index=True)
            else:
                df_out_test = pd.concat([df_out_test, df_row] , ignore_index=True)

with open('data/116_CNN_training.pickle', 'wb') as handle:
    pickle.dump(df_out_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/116_CNN_test.pickle', 'wb') as handle:
    pickle.dump(df_out_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
