import pandas as pd
import numpy as np
from datetime import datetime
import bisect

# import data set and prepare original data
holidays = ['1/1','1/28','2/14','5/30','7/4','9/5','10/10','11/11','11/24','12/25']
numOfStaions = 663

def isWeekend(strs):
    result = list()
    for i in range(len(strs)):
        day = datetime.strptime( strs[i] , '%m/%d/%y  %H:%M').weekday()
        if( day == 0 or day == 6 ):
            result.append(1)
        else:
            result.append(0)
    return result

def isHoliday(strs):
    res = list()
    for str in strs:
        curr = 0
        for holiday in holidays:
            if holiday in str :
                curr = 1
                break
        res.append(curr)
    return res

def getHour(strs):
    hours = list()
    for str in strs:
        hours.append(datetime.strptime( str , '%m/%d/%y  %H:%M').hour)
    return hours

df = pd.read_csv('201608-citibike-tripdata.csv', header=0, nrows = 1000)
# add time slot
df.insert(1,'hour', getHour(df['starttime']))
# set user type => 0/1
df.loc[df['usertype'] == "Subscriber", 'usertype'] = 1
df.loc[df['usertype'] == "Customer", 'usertype'] = 0
# set age
df.loc[df['birth year'] >= 0, 'birth year'] = (2016 - df['birth year'])
df.columns = df.columns.str.replace('birth year','age')
# set trip duration to minute
df.loc[df['tripduration'] >= 0, 'tripduration'] = df['tripduration']/60
# add var 'weekend'
df.insert(1,'weekend', isWeekend(df['starttime']))
# add var 'holiday'
df.insert(1,'holiday', 0)
df.loc[df['weekend'] == 0, 'holiday'] = isHoliday(df['starttime'])


# open another file for all station ids
df_2 = pd.read_csv('station_status.csv', header = 0)
stations = df_2['station_id']
# create blank data frame to append to df
df_ = pd.DataFrame( columns = stations)
df_ = df_.fillna(0)
result = pd.concat([df, df_], axis=1)


toReturn = list()
# initialize begining status
for station in stations:
        result.set_value(0,station,0)

for i in range(len(result.index)):
    start_station = result.iloc[i]['start station id']
    end_station = result.iloc[i]['end station id']
    
    # keep toReturn list sorted
    bisect.insort_right(toReturn,[result.iloc[i]['stoptime'],end_station])
    # copy last status
    if i > 0 :  
        for station in stations:
            result.set_value(i,station,result.iloc[i-1][station])

    # do return first
    if(len(toReturn) >0 and toReturn[0][0] in stations and toReturn[0][0] <= result.iloc[i]['starttime']):
        returnStation = toReturn[0][1]
        value = result.iloc[i][returnStation]+1
        result.set_value(i, returnStation, value)
        toReturn.pop(0)

    # do rent
    if(start_station in stations):
        result.set_value(i,start_station,result.iloc[i][start_station]-1)
    
writer = pd.ExcelWriter('pandas_simple_all.xlsx', engine='xlsxwriter')
result.to_excel(writer, sheet_name='Sheet1')
writer.save()