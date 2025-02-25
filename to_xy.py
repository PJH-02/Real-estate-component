#1
import json
import pandas as pd
import requests
from tqdm import tqdm
import numpy as np

#시간 재는 코드
import math
import time
start = time.time()

#데이터 읽어오기
raw_data = pd.read_csv('서울시 부동산 실거래가 정보(2024).csv', encoding='cp949')
data = raw_data.loc[:, ['자치구명', '본번', '부번', '지번 주소']].dropna(axis=0)
new = data.loc[data.자치구명 == '마포구'].loc[:, '지번 주소']
test_data = pd.DataFrame({'addr': new}).reset_index(drop=True)

#브이월드 API
url = 'https://api.vworld.kr/req/address?'
params = 'service=address&request=getcoord&version=2.0&crs=epsg:4326&address='
param2 = '&refine=false&simple=false&format=json&type='
road_type = 'PARCEL'
keys = '&key='
primary_key = '3EBE0A5D-090A-3E9E-A3B8-C2FB63B11D2B' #D24685C3-5613-3AC4-B6C6-EF3471CAB1B7

#json으로 url 요청
def request_geo(road):
    page = requests.get(url+params+road+param2+road_type+keys+primary_key)
    json_data = page.json()
    return json_data

#받은 데이터를 데이터프레임에 concat
def extraction_geo(test_data):
    geocode = pd.DataFrame(columns = ['address', 'x', 'y'])
    none = None
    for idx, road in tqdm(zip(test_data.index ,test_data['addr'])):
        if len(str(road)) <= 5:
            geocode = pd.concat([geocode, pd.DataFrame({'address':road, 'x':none, 'y':none}, index=[idx])], axis = 0)
            continue

        json_data = request_geo(road)

        if json_data['response']['status'] == 'NOT_FOUND' or json_data['response']['status'] == 'ERROR':
            geocode = pd.concat([geocode, pd.DataFrame({'address':road, 'x':none, 'y':none}, index=[idx])], axis = 0)
            continue

        x = json_data['response']['result']['point']['x']
        y = json_data['response']['result']['point']['y']

        geocode = pd.concat([geocode, pd.DataFrame({'address':road, 'x':float(x), 'y':float(y)}, index=[idx])], axis = 0)
    return geocode

result = extraction_geo(test_data)

#2
import pandas as pd
import geopandas as gpd
import numpy as np

# 지하철 역사 좌표 읽어오기
metro = pd.read_csv('서울시 역사마스터 정보.csv', encoding='cp949')
print(f"Loaded {len(metro)} subway stations")

# 부동산 좌표 데이터 읽어오기
house=result
house.columns = ['addr', '경도', '위도']

# 데이터 전처리 및 유효성 검사
print(f"Original house entries: {len(house)}")

# Remove rows with NaN values
house = house.dropna()
print(f"House entries after removing NaN: {len(house)}")

# 좌표 범위 확인 (서울시 대략적 범위)
valid_lat_range = (37.4, 37.7)  # 서울시 위도 범위
valid_lon_range = (126.8, 127.2)  # 서울시 경도 범위

house = house[
    (house['위도'].between(*valid_lat_range)) & 
    (house['경도'].between(*valid_lon_range))
]
print(f"House entries after coordinate validation: {len(house)}")

# geometry 사용하기 위해 위도, 경도를 GeoDataFrame으로 변경
metro_gdf = gpd.GeoDataFrame(metro, geometry=gpd.points_from_xy(metro.경도, metro.위도), crs='epsg:4326')
house_gdf = gpd.GeoDataFrame(house, geometry=gpd.points_from_xy(house.경도, house.위도), crs='epsg:4326')

# 결과를 저장할 리스트 생성
nearest_stations = []
distances = []
processed_count = 0
error_count = 0

# 각 집마다 가장 가까운 지하철역과 거리 계산
for i in range(len(house_gdf)):
    try:
        # 거리 계산
        current_house = house_gdf.iloc[i]
        example_distances = metro_gdf.to_crs(epsg=5186).geometry.distance(house_gdf.to_crs(epsg=5186).iloc[i].geometry)
        
        # 가장 가까운 역의 인덱스와 거리 찾기
        min_idx = example_distances.idxmin()
        min_distance = example_distances.min()
        
        # 거리가 너무 큰 경우 체크 (10km 이상인 경우)
        if min_distance > 10000:
            print(f"Warning: Large distance ({min_distance:.0f}m) for address: {current_house['addr']}")
        
        # 결과 저장
        nearest_stations.append(metro.iloc[min_idx]['역사명'])
        distances.append(min_distance)
        processed_count += 1
        
        # 진행상황 출력 (100개마다)
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(house_gdf)} entries")
            
    except Exception as e:
        print(f"Error processing entry {i}, address: {house_gdf.iloc[i]['addr']}")
        print(f"Error message: {str(e)}")
        nearest_stations.append(None)
        distances.append(None)
        error_count += 1

# 결과를 DataFrame에 추가
sub_df = house_gdf[['addr']].copy()
sub_df['역명'] = nearest_stations
sub_df['distance'] = distances

# 결과 검증
print("\nProcessing Summary:")
print(f"Total entries processed: {len(house_gdf)}")
print(f"Successful calculations: {processed_count}")
print(f"Errors encountered: {error_count}")
print(f"Entries with missing distances: {sub_df['distance'].isna().sum()}")

# 거리 통계
valid_distances = sub_df['distance'].dropna()
if len(valid_distances) > 0:
    print("\nDistance Statistics:")
    print(f"Average distance: {valid_distances.mean():.0f}m")
    print(f"Minimum distance: {valid_distances.min():.0f}m")
    print(f"Maximum distance: {valid_distances.max():.0f}m")

# NaN 값이 있는 행 출력
if sub_df['distance'].isna().any():
    print("\nEntries with missing distances:")
    print(sub_df[sub_df['distance'].isna()])
print(f"Total rows in saved file: {len(sub_df)}")
print(f"Rows with missing distances: {sub_df['distance'].isna().sum()}")

# 학교 좌표 읽어오기
metro = pd.read_csv('school_xy.csv', encoding='cp949')
print(f"Loaded {len(metro)} subway stations")

# 부동산 좌표 데이터 읽어오기
house = result
house.columns = ['addr', '경도', '위도']

# 데이터 전처리 및 유효성 검사
print(f"Original house entries: {len(house)}")

# 결측치 제거
house = house.dropna()
print(f"House entries after removing NaN: {len(house)}")

# 좌표 범위 확인 (서울시 대략적 범위)
valid_lat_range = (37.4, 37.7)  # 서울시 위도 범위
valid_lon_range = (126.8, 127.2)  # 서울시 경도 범위

house = house[
    (house['위도'].between(*valid_lat_range)) & 
    (house['경도'].between(*valid_lon_range))
]
print(f"House entries after coordinate validation: {len(house)}")

# geometry 사용하기 위해 위도, 경도를 GeoDataFrame으로 변경
metro_gdf = gpd.GeoDataFrame(metro, geometry=gpd.points_from_xy(metro.경도, metro.위도), crs='epsg:4326')
house_gdf = gpd.GeoDataFrame(house, geometry=gpd.points_from_xy(house.경도, house.위도), crs='epsg:4326')

# 결과를 저장할 리스트 생성
nearest_stations = []
distances = []
processed_count = 0
error_count = 0

# 각 집마다 가장 가까운 지하철역과 거리 계산
for i in range(len(house_gdf)):
    try:
        # 거리 계산
        current_house = house_gdf.iloc[i]
        example_distances = metro_gdf.to_crs(epsg=5186).geometry.distance(house_gdf.to_crs(epsg=5186).iloc[i].geometry)
        
        # 가장 가까운 역의 인덱스와 거리 찾기
        min_idx = example_distances.idxmin()
        min_distance = example_distances.min()
        
        # 거리가 너무 큰 경우 체크 (10km 이상인 경우)
        if min_distance > 10000:
            print(f"Warning: Large distance ({min_distance:.0f}m) for address: {current_house['addr']}")
        
        # 결과 저장
        nearest_stations.append(metro.iloc[min_idx]['학교명'])
        distances.append(min_distance)
        processed_count += 1
        
        # 진행상황 출력 (100개마다)
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(house_gdf)} entries")
            
    except Exception as e:
        print(f"Error processing entry {i}, address: {house_gdf.iloc[i]['addr']}")
        print(f"Error message: {str(e)}")
        nearest_stations.append(None)
        distances.append(None)
        error_count += 1

# 결과를 DataFrame에 추가
school_df = house_gdf[['addr']].copy()
school_df['학교명'] = nearest_stations
school_df['distance'] = distances

# 결과 검증
print("\nProcessing Summary:")
print(f"Total entries processed: {len(house_gdf)}")
print(f"Successful calculations: {processed_count}")
print(f"Errors encountered: {error_count}")
print(f"Entries with missing distances: {school_df['distance'].isna().sum()}")

# 거리 통계
valid_distances = school_df['distance'].dropna()
if len(valid_distances) > 0:
    print("\nDistance Statistics:")
    print(f"Average distance: {valid_distances.mean():.0f}m")
    print(f"Minimum distance: {valid_distances.min():.0f}m")
    print(f"Maximum distance: {valid_distances.max():.0f}m")

# NaN 값이 있는 행 출력
if school_df['distance'].isna().any():
    print("\nEntries with missing distances:")
    print(school_df[school_df['distance'].isna()])
print("\nSaved file verification:")
print(f"Total rows in saved file: {len(school_df)}")
print(f"Rows with missing distances: {school_df['distance'].isna().sum()}")



# 좌표 읽어오기
metro = pd.read_csv('서울시 병의원 위치 정보.csv', encoding='cp949')
print(f"Loaded {len(metro)} subway stations")

# 부동산 좌표 데이터 읽어오기
house = result
house.columns = ['addr', '경도', '위도']

# 데이터 전처리 및 유효성 검사
print(f"Original house entries: {len(house)}")

# 결측치 제거
house = house.dropna()
print(f"House entries after removing NaN: {len(house)}")

# 좌표 범위 확인 (서울시 대략적 범위)
valid_lat_range = (37.4, 37.7)  # 서울시 위도 범위
valid_lon_range = (126.8, 127.2)  # 서울시 경도 범위

house = house[
    (house['위도'].between(*valid_lat_range)) & 
    (house['경도'].between(*valid_lon_range))
]
print(f"House entries after coordinate validation: {len(house)}")

# geometry 사용하기 위해 위도, 경도를 GeoDataFrame으로 변경
metro_gdf = gpd.GeoDataFrame(metro, geometry=gpd.points_from_xy(metro.경도, metro.위도), crs='epsg:4326')
house_gdf = gpd.GeoDataFrame(house, geometry=gpd.points_from_xy(house.경도, house.위도), crs='epsg:4326')

# 결과를 저장할 리스트 생성
nearest_stations = []
distances = []
processed_count = 0
error_count = 0

# 각 집마다 가장 가까운 지하철역과 거리 계산
for i in range(len(house_gdf)):
    try:
        # 거리 계산
        current_house = house_gdf.iloc[i]
        example_distances = metro_gdf.to_crs(epsg=5186).geometry.distance(house_gdf.to_crs(epsg=5186).iloc[i].geometry)
        
        # 가장 가까운 역의 인덱스와 거리 찾기
        min_idx = example_distances.idxmin()
        min_distance = example_distances.min()
        
        # 거리가 너무 큰 경우 체크 (10km 이상인 경우)
        if min_distance > 10000:
            print(f"Warning: Large distance ({min_distance:.0f}m) for address: {current_house['addr']}")
        
        # 결과 저장
        nearest_stations.append(metro.iloc[min_idx]['기관명'])
        distances.append(min_distance)
        processed_count += 1
        
        # 진행상황 출력 (100개마다)
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(house_gdf)} entries")
            
    except Exception as e:
        print(f"Error processing entry {i}, address: {house_gdf.iloc[i]['addr']}")
        print(f"Error message: {str(e)}")
        nearest_stations.append(None)
        distances.append(None)
        error_count += 1

# 결과를 DataFrame에 추가
doc_df = house_gdf[['addr']].copy()
doc_df['기관명'] = nearest_stations
doc_df['distance'] = distances

# 결과 검증
print("\nProcessing Summary:")
print(f"Total entries processed: {len(house_gdf)}")
print(f"Successful calculations: {processed_count}")
print(f"Errors encountered: {error_count}")
print(f"Entries with missing distances: {doc_df['distance'].isna().sum()}")

# 거리 통계
valid_distances = doc_df['distance'].dropna()
if len(valid_distances) > 0:
    print("\nDistance Statistics:")
    print(f"Average distance: {valid_distances.mean():.0f}m")
    print(f"Minimum distance: {valid_distances.min():.0f}m")
    print(f"Maximum distance: {valid_distances.max():.0f}m")

# NaN 값이 있는 행 출력
if doc_df['distance'].isna().any():
    print("\nEntries with missing distances:")
    print(doc_df[doc_df['distance'].isna()])

print("\nSaved file verification:")
print(f"Total rows in saved file: {len(doc_df)}")
print(f"Rows with missing distances: {doc_df['distance'].isna().sum()}")


sub_df.columns = ['addr', '역명', '지하철 거리']
school_df.columns = ['addr', '학교명', '학교 거리']
doc_df.columns = ['addr', '기관명', '병원 거리']

A = pd.concat([sub_df, school_df], axis=1, join='inner')
GNG = pd.concat([A, doc_df], axis=1, join='inner')
GNG.to_csv('마포구 거리 데이터.csv', encoding='cp949')

#3
import pandas as pd

def merge_csv_files(file1_path, file2_path, output_path):
    """
    Merges two CSV files with duplicate handling and diagnostic information.
    """
    # Read both CSV files
    df1 = pd.read_csv(file1_path, encoding='cp949')
    df2 = pd.read_csv(file2_path, encoding='cp949')

    # Print initial diagnostic information
    print("\nInitial diagnostics:")
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")
    print(f"\nFile 1 'addr' duplicates: {df1['addr'].duplicated().sum()}")
    print(f"File 2 '지번 주소' duplicates: {df2['지번 주소'].duplicated().sum()}")

    # Rename column in df2
    df2 = df2.rename(columns={'지번 주소': 'addr'})

    # Check for unique values in the matching column
    print(f"\nUnique addresses in File 1: {df1['addr'].nunique()}")
    print(f"Unique addresses in File 2: {df2['addr'].nunique()}")

    # Option 1: Keep first occurrence of duplicates in df2
    df2_dedup = df2.drop_duplicates(subset=['addr'])

    # Merge with deduplicated df2
    merged_df = pd.merge(
        df1,
        df2_dedup[['addr', '면적당금액']],
        on='addr',
        how='left'
    )

    # Print final diagnostic information
    print("\nFinal diagnostics:")
    print(f"Final shape: {merged_df.shape}")
    print(f"Null values in '면적당금액': {merged_df['면적당금액'].isnull().sum()}")

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_path, encoding='cp949', index=False)

    return merged_df

# Example usage:
if __name__ == "__main__":
    file1_path = "강남구 거리 데이터.csv"
    file2_path = "서울시 부동산 실거래가 정보(2024).csv"
    output_path = "강남구 최종 데이터.csv"

    try:
        merged_data = merge_csv_files(file1_path, file2_path, output_path)
        print("\nFiles merged successfully! Output saved to:", output_path)
    except Exception as e:
        print(f"An error occurred: {str(e)}")