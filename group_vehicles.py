import pandas as pd
import numpy as np
from collections import defaultdict

# Загружаем данные
df = pd.read_csv('Archive/geo_locations_astana_hackathon.csv')

print(f"Общее количество записей: {len(df)}")
print(f"Количество уникальных автомобилей: {df['randomized_id'].nunique()}")
print(f"Среднее количество точек на автомобиль: {len(df) / df['randomized_id'].nunique():.2f}")

# Группируем данные по randomized_id
grouped_data = defaultdict(list)

for _, row in df.iterrows():
    vehicle_id = row['randomized_id']
    grouped_data[vehicle_id].append({
        'lat': row['lat'],
        'lng': row['lng'],
        'alt': row['alt'],
        'spd': row['spd'],
        'azm': row['azm']
    })

# Анализ данных по каждому автомобилю
print("\nАнализ траекторий автомобилей:")
print("-" * 50)

for vehicle_id, points in list(grouped_data.items())[:10]:  # Показываем первые 10
    print(f"Автомобиль ID: {vehicle_id}")
    print(f"  Количество точек: {len(points)}")
    
    # Вычисляем статистику
    speeds = [p['spd'] for p in points if p['spd'] >= 0]  # Исключаем отрицательные скорости
    if speeds:
        print(f"  Средняя скорость: {np.mean(speeds):.2f} км/ч")
        print(f"  Максимальная скорость: {max(speeds):.2f} км/ч")
    
    # Вычисляем общее расстояние (приблизительно)
    total_distance = 0
    for i in range(1, len(points)):
        lat1, lng1 = points[i-1]['lat'], points[i-1]['lng']
        lat2, lng2 = points[i]['lat'], points[i]['lng']
        
        # Простое вычисление расстояния (приблизительное)
        distance = np.sqrt((lat2-lat1)**2 + (lng2-lng1)**2) * 111000  # примерно в метрах
        total_distance += distance
    
    print(f"  Приблизительное расстояние: {total_distance/1000:.2f} км")
    print()

# Сохраняем сгруппированные данные
print("Сохранение сгруппированных данных...")

# Создаем DataFrame с дополнительной информацией
result_data = []
for vehicle_id, points in grouped_data.items():
    for i, point in enumerate(points):
        result_data.append({
            'randomized_id': vehicle_id,
            'point_sequence': i + 1,  # Порядковый номер точки
            'total_points': len(points),
            'lat': point['lat'],
            'lng': point['lng'],
            'alt': point['alt'],
            'spd': point['spd'],
            'azm': point['azm']
        })

result_df = pd.DataFrame(result_data)

# Сортируем по ID автомобиля и порядковому номеру точки
result_df = result_df.sort_values(['randomized_id', 'point_sequence'])

# Сохраняем результат
result_df.to_csv('grouped_vehicles_data.csv', index=False)
print(f"Данные сохранены в файл 'grouped_vehicles_data.csv'")

# Статистика по количеству точек на автомобиль
points_per_vehicle = df.groupby('randomized_id').size()
print(f"\nСтатистика по количеству точек на автомобиль:")
print(f"Минимум точек: {points_per_vehicle.min()}")
print(f"Максимум точек: {points_per_vehicle.max()}")
print(f"Медиана: {points_per_vehicle.median()}")
print(f"Среднее: {points_per_vehicle.mean():.2f}")