import numpy as np
import pandas as pd
import os

# 地球半径
R_earth = 6371000  # km

# 卫星参数
sat_lat = 34.0      # 卫星星下点纬度
sat_lon = -118.0    # 卫星星下点经度
sat_alt = 1100000      # 卫星轨道高度，单位km

# 小区参数
cell_radius = 50000    # 小区半径，单位km

# 六边形蜂窝布局相对坐标（单位：小区半径）
hex_coords = [
    (0, 0),
    (1, 0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2), (-1, 0), (-0.5, -np.sqrt(3)/2), (0.5, -np.sqrt(3)/2),
    (2, 0), (1.5, np.sqrt(3)), (0, np.sqrt(3)), (-1.5, np.sqrt(3)), (-2, 0), (-1.5, -np.sqrt(3)), (0, -np.sqrt(3)), (1.5, -np.sqrt(3)),
    (1, np.sqrt(3)), (-1, np.sqrt(3)), (-1, -np.sqrt(3)), (1, -np.sqrt(3))
]

def offset_latlon(lat, lon, dx, dy):
    # dx, dy 单位为km
    R = R_earth
    dlat = (dy / R) * (180 / np.pi)
    dlon = (dx / (R * np.cos(np.pi * lat / 180))) * (180 / np.pi)
    return lat + dlat, lon + dlon

def geodetic_to_ecef(lat, lon, alt):
    # 输入纬度、经度（度），高度（km），输出ECEF坐标（km）
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    r = R_earth + alt
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return x, y, z

# 卫星三维坐标
sat_x, sat_y, sat_z = geodetic_to_ecef(sat_lat, sat_lon, sat_alt)
satellite = {'sat_id': 0, 'x': sat_x, 'y': sat_y, 'z': sat_z}

# 小区三维坐标
cell_positions = []
for i, (x, y) in enumerate(hex_coords):
    dx = x * cell_radius * 2 * 0.5
    dy = y * cell_radius * 2 * 0.5
    lat, lon = offset_latlon(sat_lat, sat_lon, dx, dy)
    cx, cy, cz = geodetic_to_ecef(lat, lon, 0)
    cell_positions.append({'cell_id': i, 'x': cx, 'y': cy, 'z': cz})

# 保存为csv
os.makedirs('data', exist_ok=True)
pd.DataFrame([satellite]).to_csv('data/fixed_satellite_position_xyz.csv', index=False)
pd.DataFrame(cell_positions).to_csv('data/fixed_cell_positions_xyz.csv', index=False)

print("三维坐标已保存到 data/fixed_satellite_position_xyz.csv 和 data/fixed_cell_positions_xyz.csv")