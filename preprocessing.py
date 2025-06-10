import pandas as pd
import numpy as np


train = pd.read_csv("C:/Users/LS/Desktop/프로젝트_2/data/train.csv")
train = train[train['id'] != 19327]

train['id'].unique() # 칼럼 제거
train['line'].unique() # 칼럼 제거
train['name'].unique() # 칼럼 제거
train['mold_name'].unique() # 칼럼 제거
train['time'].unique() # 칼럼 제거
train['date'].unique() # 칼럼 제거
train['count'].unique()
train['working'].unique() # 칼럼 제거
train['emergency_stop'].unique() # 결측행 제거 및 칼럼 제거
train['molten_temp'].unique()
train['facility_operation_cycleTime'].unique() # 칼럼 제거
train['production_cycletime'].unique() # 칼럼 제거
train['low_section_speed'].unique()
train['high_section_speed'].unique()
train['molten_volume'].unique() # 칼럼 제거
train['cast_pressure'].unique()
train['biscuit_thickness'].unique()
train['upper_mold_temp1'].unique()
train['upper_mold_temp2'].unique()
train['upper_mold_temp3'].unique() # 칼럼 제거
train['lower_mold_temp1'].unique()
train['lower_mold_temp2'].unique()
train['lower_mold_temp3'].unique() # 칼럼 제거
train['sleeve_temperature'].unique()
train['physical_strength'].unique()
train['Coolant_temperature'].unique()
train['EMS_operation_time'].unique() # 칼럼 제거
train['registration_time'].unique() # 칼럼 제거
train['passorfail'].unique() # 목적 변수
train['tryshot_signal'].unique() # D행 제거 및 칼럼 제거
train['mold_code'].unique() # 명목형 처리
train['heating_furnace'].unique() # 칼럼 제거

# 모델 적용 컬럼
# count, molten_temp, low_section_speed, high_section_speed,
# cast_pressure, biscuit_thickness, upper_mold_temp1, upper_mold_temp2
# lower_mold_temp1, lower_mold_temp2, sleeve_temperature, physical_strength,
# Coolant_temperature, mold_code