import pandas as pd

column_sensor_velocity_30m_1_AVG = 'AVGP5'
column_sensor_velocity_30m_1_MAX = 'MAXP5'
column_sensor_velocity_30m_1_MIN = 'MINP5'
column_sensor_velocity_30m_1_DEV = 'DEVP5'

column_sensor_velocity_20m_1_AVG = 'AVGP13'
column_sensor_velocity_20m_1_MAX = 'MAXP13'
column_sensor_velocity_20m_1_MIN = 'MINP13'
column_sensor_velocity_20m_1_DEV = 'DEVP13'

column_sensor_velocity_30m_2_AVG = 'AVGP9'
column_sensor_velocity_30m_2_MAX = 'MAXP9'
column_sensor_velocity_30m_2_MIN = 'MINP9'
column_sensor_velocity_30m_2_DEV = 'DEVP9'

column_sensor_direction_30m_1_AVG = 'AVGP6'
column_sensor_direction_30m_1_MAX = 'MAXP6'
column_sensor_direction_30m_1_MIN = 'MINP6'
column_sensor_direction_30m_1_DEV = 'DEVP6'

column_sensor_direction_20m_1_AVG = 'AVGP14'
column_sensor_direction_20m_1_MAX = 'MAXP14'
column_sensor_direction_20m_1_MIN = 'MINP14'
column_sensor_direction_20m_1_DEV = 'DEVP14'

column_sensor_pressure_AVG = 'AVGP4'
column_sensor_pressure_MAX = 'MAXP4'
column_sensor_pressure_MIN = 'MINP4'
column_sensor_pressure_DEV = 'DEVP4'

column_sensor_temperature_AVG = 'AVGP1'
column_sensor_temperature_MAX = 'MAXP1'
column_sensor_temperature_MIN = 'MINP1'
column_sensor_temperature_DEV = 'DEVP1'

column_sensor_humidity_AVG = 'AVGP3'
column_sensor_humidity_MAX = 'MAXP3'
column_sensor_humidity_MIN = 'MINP3'
column_sensor_humidity_DEV = 'DEVP3'


column_date = 'DATE'
column_time = 'TIME'
column_datetime = 'Timestamp'
column_voltage_batery = 'AVGCi'
column_current_batery = 'MINVi'


# Define columns to read
use_columns = [
    column_datetime, column_sensor_temperature_AVG, column_sensor_temperature_MIN,
    column_sensor_temperature_MAX, column_sensor_humidity_AVG, column_sensor_humidity_MIN,
    column_sensor_humidity_MAX, column_sensor_velocity_30m_1_AVG, column_sensor_velocity_30m_1_MAX,
    column_sensor_velocity_30m_1_MIN, column_sensor_direction_30m_1_AVG, column_sensor_velocity_30m_2_AVG,
    column_sensor_velocity_30m_2_MIN, column_sensor_velocity_30m_2_MAX, column_sensor_velocity_20m_1_AVG,
    column_sensor_velocity_20m_1_MIN, column_sensor_velocity_20m_1_MAX, column_sensor_direction_20m_1_AVG,
    column_sensor_pressure_AVG, column_sensor_pressure_MIN, column_sensor_pressure_MAX,
    column_sensor_temperature_DEV, column_sensor_humidity_DEV, column_sensor_velocity_30m_1_DEV,
    column_sensor_direction_30m_1_MIN, column_sensor_direction_30m_1_MAX, column_sensor_direction_30m_1_DEV,
    column_sensor_velocity_30m_2_DEV, column_sensor_pressure_DEV, column_sensor_velocity_20m_1_DEV,
    column_sensor_direction_20m_1_MIN, column_sensor_direction_20m_1_MAX, column_sensor_direction_20m_1_DEV
]

columns_sensor_velocity_1_30_m = [column_sensor_velocity_30m_1_AVG, column_sensor_velocity_30m_1_MAX,
                                  column_sensor_velocity_30m_1_MIN, column_sensor_velocity_30m_1_DEV]

columns_sensor_velocity_2_30_m = [column_sensor_velocity_30m_2_AVG, column_sensor_velocity_30m_2_MAX,
                                  column_sensor_velocity_30m_2_MIN, column_sensor_velocity_30m_2_DEV]

columns_sensor_velocity_1_20_m = [column_sensor_velocity_20m_1_AVG, column_sensor_velocity_20m_1_MAX,
                                  column_sensor_velocity_20m_1_MIN, column_sensor_velocity_20m_1_DEV]

columns_sensor_direction_1_30_m = [column_sensor_direction_30m_1_AVG, column_sensor_direction_30m_1_MAX,
                                   column_sensor_direction_30m_1_MIN, column_sensor_direction_30m_1_DEV]

columns_sensor_direction_1_20_m = [column_sensor_direction_20m_1_AVG, column_sensor_direction_20m_1_MAX,
                                   column_sensor_direction_20m_1_MIN, column_sensor_direction_20m_1_DEV]

columns_sensor_pressure = [column_sensor_pressure_AVG, column_sensor_pressure_MAX, column_sensor_pressure_MIN,
                           column_sensor_pressure_DEV]

columns_sensor_temperature = [column_sensor_temperature_AVG, column_sensor_temperature_MAX,
                              column_sensor_temperature_MIN, column_sensor_temperature_DEV]

columns_sensor_humidity = [column_sensor_humidity_AVG, column_sensor_humidity_MAX,
                           column_sensor_humidity_MIN, column_sensor_humidity_DEV]

tower_numer_1 = '1'
tower_numer_2 = '2'
tower_numer_3 = '3'
tower_numer_4 = '4'
tower_number_5 = '5'


def load_data_tower(csv_file_path: str) -> pd.DataFrame:
    global use_columns
    df = pd.read_csv(csv_file_path, usecols=use_columns, delimiter=";")
    df = df.dropna()
    df['datetime'] = pd.to_datetime(df[column_datetime], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('datetime')
    df = df.sort_index()
    return df


