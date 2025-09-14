import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import warnings
warnings.filterwarnings('ignore')

def load_gps_data(csv_path):
    """
    Загрузка GPS данных из CSV файла
    """
    print("Загрузка GPS данных...")
    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} GPS точек")
    return df

def create_grid_heatmap(df, grid_size=50):
    """
    Создание хитмапа на основе сетки с улучшенным алгоритмом загруженности
    grid_size: количество ячеек по каждой оси
    """
    print(f"Создание сетки {grid_size}x{grid_size} с новым алгоритмом загруженности...")
    
    # Определяем границы области
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lng_min, lng_max = df['lng'].min(), df['lng'].max()
    
    # Создаем сетку
    lat_bins = np.linspace(lat_min, lat_max, grid_size + 1)
    lng_bins = np.linspace(lng_min, lng_max, grid_size + 1)
    
    # Инициализируем массивы для хитмапа
    point_density = np.zeros((grid_size, grid_size))
    speed_avg = np.zeros((grid_size, grid_size))
    speed_max = np.zeros((grid_size, grid_size))  # Базовая скорость свободного движения
    speed_std = np.zeros((grid_size, grid_size))  # Стандартное отклонение скорости
    traffic_load = np.zeros((grid_size, grid_size))  # Новый коэффициент загруженности
    
    # Минимальное количество точек для расчета загруженности
    min_points_threshold = 5
    
    # Заполняем сетку данными
    for i in range(grid_size):
        for j in range(grid_size):
            # Определяем границы ячейки
            lat_low, lat_high = lat_bins[i], lat_bins[i + 1]
            lng_low, lng_high = lng_bins[j], lng_bins[j + 1]
            
            # Находим точки в этой ячейке
            mask = ((df['lat'] >= lat_low) & (df['lat'] < lat_high) & 
                   (df['lng'] >= lng_low) & (df['lng'] < lng_high))
            
            cell_points = df[mask]
            
            if len(cell_points) >= min_points_threshold:
                # Количество точек (плотность)
                point_density[i, j] = len(cell_points)
                
                # Статистика скорости в ячейке
                speeds = cell_points['spd']
                speed_avg[i, j] = speeds.mean()
                speed_std[i, j] = speeds.std() if len(speeds) > 1 else 0
                
                # Базовая скорость свободного движения (90-й процентиль)
                speed_max[i, j] = np.percentile(speeds, 90)
                
                # Расчет коэффициента загруженности
                if speed_max[i, j] > 5:  # Минимальная базовая скорость 5 км/ч
                    # Основная формула: (Базовая_скорость - Средняя_скорость) / Базовая_скорость
                    congestion_ratio = (speed_max[i, j] - speed_avg[i, j]) / speed_max[i, j]
                    congestion_ratio = max(0, min(1, congestion_ratio))  # Ограничиваем от 0 до 1
                    
                    # Дополнительный фактор: учитываем вариацию скорости
                    # Если скорость сильно варьируется, это признак переменной загруженности
                    variation_factor = min(speed_std[i, j] / speed_max[i, j], 0.3) if speed_max[i, j] > 0 else 0
                    
                    # Фактор плотности: больше точек = больше активности
                    density_factor = min(point_density[i, j] / 100, 0.2)  # Максимум 0.2
                    
                    # Итоговая загруженность
                    traffic_load[i, j] = congestion_ratio + variation_factor + density_factor
                    traffic_load[i, j] = min(traffic_load[i, j], 1.0)  # Максимум 1.0
                else:
                    # Если базовая скорость слишком низкая, считаем это дворовой территорией
                    traffic_load[i, j] = 0.1  # Минимальная загруженность
    
    print(f"Обработано ячеек с достаточным количеством данных: {np.sum(point_density > 0)}")
    print(f"Средняя базовая скорость: {np.mean(speed_max[speed_max > 0]):.1f} км/ч")
    print(f"Средняя загруженность: {np.mean(traffic_load[traffic_load > 0]):.3f}")
    
    return lat_bins, lng_bins, point_density, speed_avg, traffic_load

def smooth_heatmap(heatmap, sigma=1.0):
    """
    Сглаживание хитмапа для плавных переходов
    """
    from scipy import ndimage
    try:
        return ndimage.gaussian_filter(heatmap, sigma=sigma)
    except ImportError:
        print("Scipy не установлен, используем простое сглаживание")
        # Простое сглаживание без scipy
        kernel = np.ones((3, 3)) / 9
        smoothed = np.zeros_like(heatmap)
        for i in range(1, heatmap.shape[0] - 1):
            for j in range(1, heatmap.shape[1] - 1):
                smoothed[i, j] = np.sum(heatmap[i-1:i+2, j-1:j+2] * kernel)
        return smoothed

def create_visualization(df, lat_bins, lng_bins, point_density, speed_avg, traffic_load, output_path='heatmap.png'):
    """
    Создание визуализации хитмапа с улучшенными цветовыми схемами
    """
    print("Создание визуализации...")
    
    # Применяем сглаживание
    point_density_smooth = smooth_heatmap(point_density, sigma=1.0)
    speed_avg_smooth = smooth_heatmap(speed_avg, sigma=1.0)
    traffic_load_smooth = smooth_heatmap(traffic_load, sigma=1.0)
    
    # Создаем улучшенные цветовые карты
    # Для загруженность: зеленый -> желтый -> красный
    colors_traffic = ['#00ff00', '#ffff00', '#ff8000', '#ff0000', '#800000']
    traffic_cmap = LinearSegmentedColormap.from_list('traffic', colors_traffic, N=256)
    
    # Для скорости: синий -> голубой -> зеленый -> желтый -> красный
    colors_speed = ['#000080', '#0080ff', '#00ff80', '#ffff00', '#ff0000']
    speed_cmap = LinearSegmentedColormap.from_list('speed', colors_speed, N=256)
    
    # Создаем фигуру с 4 подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Анализ дорожного трафика с улучшенным алгоритмом загруженности', fontsize=16, fontweight='bold')
    
    # 1. Исходные GPS точки
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['lng'], df['lat'], c=df['spd'], cmap='viridis', alpha=0.6, s=1)
    ax1.set_title('GPS точки (цвет = скорость)', fontweight='bold')
    ax1.set_xlabel('Долгота')
    ax1.set_ylabel('Широта')
    plt.colorbar(scatter, ax=ax1, label='Скорость (км/ч)')
    
    # 2. Плотность точек
    ax2 = axes[0, 1]
    im2 = ax2.imshow(point_density_smooth, cmap='YlOrRd', aspect='auto', 
                     extent=[lng_bins[0], lng_bins[-1], lat_bins[0], lat_bins[-1]], origin='lower')
    ax2.set_title('Плотность GPS точек', fontweight='bold')
    ax2.set_xlabel('Долгота')
    ax2.set_ylabel('Широта')
    plt.colorbar(im2, ax=ax2, label='Количество точек')
    
    # 3. Средняя скорость
    ax3 = axes[1, 0]
    # Маскируем нулевые значения
    speed_masked = np.ma.masked_where(speed_avg_smooth == 0, speed_avg_smooth)
    im3 = ax3.imshow(speed_masked, cmap=speed_cmap, aspect='auto',
                     extent=[lng_bins[0], lng_bins[-1], lat_bins[0], lat_bins[-1]], origin='lower')
    ax3.set_title('Средняя скорость движения', fontweight='bold')
    ax3.set_xlabel('Долгота')
    ax3.set_ylabel('Широта')
    plt.colorbar(im3, ax=ax3, label='Скорость (км/ч)')
    
    # 4. Загруженность дорог (новый алгоритм)
    ax4 = axes[1, 1]
    # Маскируем нулевые значения
    traffic_masked = np.ma.masked_where(traffic_load_smooth == 0, traffic_load_smooth)
    im4 = ax4.imshow(traffic_masked, cmap=traffic_cmap, aspect='auto',
                     extent=[lng_bins[0], lng_bins[-1], lat_bins[0], lat_bins[-1]], origin='lower')
    ax4.set_title('Загруженность дорог (улучшенный алгоритм)', fontweight='bold')
    ax4.set_xlabel('Долгота')
    ax4.set_ylabel('Широта')
    cbar4 = plt.colorbar(im4, ax=ax4, label='Коэффициент загруженности')
    
    # Добавляем подписи к цветовой шкале загруженности
    cbar4.ax.text(1.15, 0.1, 'Свободно', transform=cbar4.ax.transAxes, 
                  verticalalignment='center', color='green', fontweight='bold')
    cbar4.ax.text(1.15, 0.5, 'Умеренно', transform=cbar4.ax.transAxes, 
                  verticalalignment='center', color='orange', fontweight='bold')
    cbar4.ax.text(1.15, 0.9, 'Пробка', transform=cbar4.ax.transAxes, 
                  verticalalignment='center', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Визуализация сохранена: {output_path}")
    plt.show()

def print_statistics(point_density, speed_avg, traffic_load):
    """
    Вывод статистики по хитмапу
    """
    print("\n" + "="*50)
    print("СТАТИСТИКА ЗАГРУЖЕННОСТИ")
    print("="*50)
    
    # Убираем нулевые значения для статистики
    non_zero_density = point_density[point_density > 0]
    non_zero_speed = speed_avg[speed_avg > 0]
    non_zero_traffic = traffic_load[traffic_load > 0]
    
    if len(non_zero_density) > 0:
        print(f"Общее количество активных ячеек: {len(non_zero_density)}")
        print(f"Средняя плотность точек: {non_zero_density.mean():.1f}")
        print(f"Максимальная плотность: {non_zero_density.max():.0f}")
    
    if len(non_zero_speed) > 0:
        print(f"Средняя скорость: {non_zero_speed.mean():.1f} км/ч")
        print(f"Минимальная скорость: {non_zero_speed.min():.1f} км/ч")
        print(f"Максимальная скорость: {non_zero_speed.max():.1f} км/ч")
    
    if len(non_zero_traffic) > 0:
        print(f"Средняя загруженность: {non_zero_traffic.mean():.2f}")
        print(f"Максимальная загруженность: {non_zero_traffic.max():.2f}")
        
        # Находим самые загруженные области
        top_traffic_threshold = np.percentile(non_zero_traffic, 90)
        high_traffic_areas = np.sum(traffic_load >= top_traffic_threshold)
        print(f"Количество сильно загруженных областей (топ 10%): {high_traffic_areas}")

def save_grid_data_to_file(lat_bins, lng_bins, point_density, speed_avg, traffic_load, output_path='grid_data.json'):
    """
    Сохранение данных сетки в файл для отображения на реальной карте
    """
    print(f"Сохранение данных сетки в {output_path}...")
    
    grid_data = []
    
    for i in range(len(lat_bins) - 1):
        for j in range(len(lng_bins) - 1):
            # Пропускаем пустые ячейки
            if point_density[i, j] > 0:
                # Координаты центра ячейки
                lat_center = (lat_bins[i] + lat_bins[i + 1]) / 2
                lng_center = (lng_bins[j] + lng_bins[j + 1]) / 2
                
                # Границы ячейки
                lat_min, lat_max = lat_bins[i], lat_bins[i + 1]
                lng_min, lng_max = lng_bins[j], lng_bins[j + 1]
                
                cell_data = {
                    'cell_id': f"{i}_{j}",
                    'center': {
                        'lat': float(lat_center),
                        'lng': float(lng_center)
                    },
                    'bounds': {
                        'lat_min': float(lat_min),
                        'lat_max': float(lat_max),
                        'lng_min': float(lng_min),
                        'lng_max': float(lng_max)
                    },
                    'metrics': {
                        'point_density': int(point_density[i, j]),
                        'avg_speed': float(speed_avg[i, j]),
                        'traffic_load': float(traffic_load[i, j])
                    }
                }
                grid_data.append(cell_data)
    
    # Сохраняем в JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total_cells': len(grid_data),
                'grid_size': len(lat_bins) - 1,
                'bounds': {
                    'lat_min': float(lat_bins[0]),
                    'lat_max': float(lat_bins[-1]),
                    'lng_min': float(lng_bins[0]),
                    'lng_max': float(lng_bins[-1])
                }
            },
            'cells': grid_data
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Сохранено {len(grid_data)} активных ячеек в {output_path}")
    return output_path

def save_grid_data_to_csv(lat_bins, lng_bins, point_density, speed_avg, traffic_load, output_path='grid_data.csv'):
    """
    Сохранение данных сетки в CSV файл
    """
    print(f"Сохранение данных сетки в CSV формате: {output_path}...")
    
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['cell_id', 'lat_center', 'lng_center', 'lat_min', 'lat_max', 
                     'lng_min', 'lng_max', 'point_density', 'avg_speed', 'traffic_load']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for i in range(len(lat_bins) - 1):
            for j in range(len(lng_bins) - 1):
                if point_density[i, j] > 0:
                    lat_center = (lat_bins[i] + lat_bins[i + 1]) / 2
                    lng_center = (lng_bins[j] + lng_bins[j + 1]) / 2
                    
                    writer.writerow({
                        'cell_id': f"{i}_{j}",
                        'lat_center': lat_center,
                        'lng_center': lng_center,
                        'lat_min': lat_bins[i],
                        'lat_max': lat_bins[i + 1],
                        'lng_min': lng_bins[j],
                        'lng_max': lng_bins[j + 1],
                        'point_density': int(point_density[i, j]),
                        'avg_speed': speed_avg[i, j],
                        'traffic_load': traffic_load[i, j]
                    })
    
    print(f"CSV файл сохранен: {output_path}")
    return output_path

def create_geojson_for_map(lat_bins, lng_bins, point_density, speed_avg, traffic_load, output_path='grid_data.geojson'):
    """
    Создание GeoJSON файла для отображения на картах (Leaflet, Google Maps и т.д.)
    """
    print(f"Создание GeoJSON файла: {output_path}...")
    
    features = []
    
    for i in range(len(lat_bins) - 1):
        for j in range(len(lng_bins) - 1):
            if point_density[i, j] > 0:
                # Создаем прямоугольник (полигон) для ячейки
                lat_min, lat_max = lat_bins[i], lat_bins[i + 1]
                lng_min, lng_max = lng_bins[j], lng_bins[j + 1]
                
                # Координаты углов прямоугольника
                coordinates = [[
                    [lng_min, lat_min],  # нижний левый
                    [lng_max, lat_min],  # нижний правый
                    [lng_max, lat_max],  # верхний правый
                    [lng_min, lat_max],  # верхний левый
                    [lng_min, lat_min]   # замыкаем полигон
                ]]
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "cell_id": f"{i}_{j}",
                        "point_density": int(point_density[i, j]),
                        "avg_speed": float(speed_avg[i, j]),
                        "traffic_load": float(traffic_load[i, j]),
                        "center_lat": float((lat_min + lat_max) / 2),
                        "center_lng": float((lng_min + lng_max) / 2)
                    }
                }
                features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    
    print(f"GeoJSON файл создан: {output_path} ({len(features)} ячеек)")
    return output_path

def main():
    """
    Основная функция для создания хитмапа загруженности
    """
    # Параметры конфигурации
    CSV_PATH = '/Users/sultankarilov/Desktop/projects/decentra_2_4.0/grouped_vehicles_data.csv'
    GRID_SIZE = 300  # Размер сетки (50x50 ячеек)
    
    print("Создание хитмапа загруженности дорожного движения")
    print("=" * 55)
    
    try:
        # Шаг 1: Загрузка данных
        df = load_gps_data(CSV_PATH)
        
        # Шаг 2: Создание сетки и вычисление метрик
        lat_bins, lng_bins, point_density, speed_avg, traffic_load = create_grid_heatmap(df, GRID_SIZE)
        
        # Шаг 3: Создание визуализации
        create_visualization(df, lat_bins, lng_bins, point_density, speed_avg, traffic_load)
        
        # Шаг 4: Сохранение данных для карт
        print("\nСохранение данных для отображения на картах...")
        save_grid_data_to_file(lat_bins, lng_bins, point_density, speed_avg, traffic_load, 'grid_data.json')
        save_grid_data_to_csv(lat_bins, lng_bins, point_density, speed_avg, traffic_load, 'grid_data.csv')
        create_geojson_for_map(lat_bins, lng_bins, point_density, speed_avg, traffic_load, 'grid_data.geojson')
        
        # Шаг 5: Вывод статистики
        print_statistics(point_density, speed_avg, traffic_load)
        
        print("\nХитмап загруженности создан успешно!")
        print("Файлы для карт:")
        print("- grid_data.json (структурированные данные)")
        print("- grid_data.csv (табличный формат)")
        print("- grid_data.geojson (для веб-карт)")
        
    except FileNotFoundError:
        print(f"Ошибка: Не удалось найти CSV файл по пути {CSV_PATH}")
        print("Проверьте путь к файлу и попробуйте снова.")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        print("Проверьте формат данных и попробуйте снова.")

if __name__ == "__main__":
    main()