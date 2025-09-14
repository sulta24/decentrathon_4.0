import pandas as pd
import numpy as np
import folium
from folium import plugins
import json
import warnings
warnings.filterwarnings('ignore')

def load_grid_data(json_path):
    """
    Загрузка данных сетки из JSON файла
    """
    print(f"Загрузка данных сетки из {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_interactive_map(grid_data, output_path='traffic_map.html'):
    """
    Создание интерактивной карты с данными о загруженности
    """
    print("Создание интерактивной карты...")
    
    # Получаем границы области
    bounds = grid_data['metadata']['bounds']
    center_lat = (bounds['lat_min'] + bounds['lat_max']) / 2
    center_lng = (bounds['lng_min'] + bounds['lng_max']) / 2
    
    # Создаем базовую карту
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Добавляем альтернативные слои карт
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
    
    # Получаем максимальные значения для нормализации цветов
    max_density = max(cell['metrics']['point_density'] for cell in grid_data['cells'])
    max_traffic = max(cell['metrics']['traffic_load'] for cell in grid_data['cells'])
    max_speed = max(cell['metrics']['avg_speed'] for cell in grid_data['cells'])
    
    # Создаем группы слоев
    density_group = folium.FeatureGroup(name='Плотность точек')
    traffic_group = folium.FeatureGroup(name='Загруженность движения')
    speed_group = folium.FeatureGroup(name='Средняя скорость')
    
    # Функция для получения цвета по загруженности
    def get_traffic_color(traffic_load, max_val):
        normalized = traffic_load / max_val if max_val > 0 else 0
        if normalized < 0.2:
            return '#00ff00'  # зеленый
        elif normalized < 0.4:
            return '#80ff00'  # желто-зеленый
        elif normalized < 0.6:
            return '#ffff00'  # желтый
        elif normalized < 0.8:
            return '#ff8000'  # оранжевый
        else:
            return '#ff0000'  # красный
    
    # Функция для получения цвета по скорости
    def get_speed_color(speed, max_val):
        normalized = speed / max_val if max_val > 0 else 0
        if normalized > 0.8:
            return '#0000ff'  # синий (высокая скорость)
        elif normalized > 0.6:
            return '#0080ff'  # голубой
        elif normalized > 0.4:
            return '#00ffff'  # циан
        elif normalized > 0.2:
            return '#80ff80'  # светло-зеленый
        else:
            return '#ff0000'  # красный (низкая скорость)
    
    # Функция для получения цвета по плотности
    def get_density_color(density, max_val):
        normalized = density / max_val if max_val > 0 else 0
        if normalized < 0.2:
            return '#ffffcc'  # очень светлый
        elif normalized < 0.4:
            return '#ffeda0'  # светлый
        elif normalized < 0.6:
            return '#fed976'  # средний
        elif normalized < 0.8:
            return '#feb24c'  # темный
        else:
            return '#f03b20'  # очень темный
    
    # Добавляем ячейки на карту
    for cell in grid_data['cells']:
        bounds_data = cell['bounds']
        metrics = cell['metrics']
        
        # Координаты прямоугольника
        rectangle_coords = [
            [bounds_data['lat_min'], bounds_data['lng_min']],
            [bounds_data['lat_max'], bounds_data['lng_max']]
        ]
        
        # Создаем popup с информацией
        popup_text = f"""
        <b>Ячейка {cell['cell_id']}</b><br>
        Плотность точек: {metrics['point_density']}<br>
        Средняя скорость: {metrics['avg_speed']:.1f} км/ч<br>
        Загруженность: {metrics['traffic_load']:.2f}<br>
        Координаты центра: {cell['center']['lat']:.6f}, {cell['center']['lng']:.6f}
        """
        
        # Слой плотности
        folium.Rectangle(
            bounds=rectangle_coords,
            color='black',
            weight=1,
            fill=True,
            fillColor=get_density_color(metrics['point_density'], max_density),
            fillOpacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(density_group)
        
        # Слой загруженности
        folium.Rectangle(
            bounds=rectangle_coords,
            color='black',
            weight=1,
            fill=True,
            fillColor=get_traffic_color(metrics['traffic_load'], max_traffic),
            fillOpacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(traffic_group)
        
        # Слой скорости
        folium.Rectangle(
            bounds=rectangle_coords,
            color='black',
            weight=1,
            fill=True,
            fillColor=get_speed_color(metrics['avg_speed'], max_speed),
            fillOpacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(speed_group)
    
    # Добавляем группы на карту (по умолчанию показываем загруженность)
    traffic_group.add_to(m)
    density_group.add_to(m)
    speed_group.add_to(m)
    
    # Добавляем контроль слоев
    folium.LayerControl().add_to(m)
    
    # Добавляем легенду для загруженности
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Загруженность</b></p>
    <p><i class="fa fa-square" style="color:#00ff00"></i> Низкая</p>
    <p><i class="fa fa-square" style="color:#ffff00"></i> Средняя</p>
    <p><i class="fa fa-square" style="color:#ff8000"></i> Высокая</p>
    <p><i class="fa fa-square" style="color:#ff0000"></i> Очень высокая</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Сохраняем карту
    m.save(output_path)
    print(f"Интерактивная карта сохранена как {output_path}")
    return m

def create_heatmap_overlay(grid_data, output_path='heatmap_overlay.html'):
    """
    Создание карты с тепловой картой (heatmap overlay)
    """
    print("Создание тепловой карты...")
    
    bounds = grid_data['metadata']['bounds']
    center_lat = (bounds['lat_min'] + bounds['lat_max']) / 2
    center_lng = (bounds['lng_min'] + bounds['lng_max']) / 2
    
    # Создаем карту
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Подготавливаем данные для heatmap
    heat_data = []
    for cell in grid_data['cells']:
        lat = cell['center']['lat']
        lng = cell['center']['lng']
        weight = cell['metrics']['traffic_load']
        heat_data.append([lat, lng, weight])
    
    # Добавляем heatmap
    plugins.HeatMap(
        heat_data,
        min_opacity=0.2,
        max_zoom=18,
        radius=25,
        blur=15,
        gradient={
            0.2: 'blue',
            0.4: 'lime', 
            0.6: 'orange',
            0.8: 'red',
            1.0: 'darkred'
        }
    ).add_to(m)
    
    # Сохраняем карту
    m.save(output_path)
    print(f"Тепловая карта сохранена как {output_path}")
    return m

def add_gps_points_to_map(csv_path, map_obj, sample_size=1000):
    """
    Добавление исходных GPS точек на карту (выборка для производительности)
    """
    print(f"Добавление GPS точек на карту (выборка {sample_size} точек)...")
    
    # Загружаем данные
    df = pd.read_csv(csv_path)
    
    # Берем случайную выборку для производительности
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    # Добавляем точки на карту
    for _, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=2,
            popup=f"Скорость: {row['spd']:.1f} км/ч",
            color='blue',
            fill=True,
            fillOpacity=0.6
        ).add_to(map_obj)
    
    return map_obj

def main():
    """
    Основная функция для создания карт
    """
    # Пути к файлам
    GRID_DATA_PATH = 'Archive/grid_data.json'
    CSV_PATH = 'Archive/grouped_vehicles_data.csv'
    
    print("Создание интерактивных карт загруженности")
    print("=" * 45)
    
    try:
        # Проверяем наличие файла с данными сетки
        try:
            grid_data = load_grid_data(GRID_DATA_PATH)
        except FileNotFoundError:
            print(f"Файл {GRID_DATA_PATH} не найден.")
            print("Сначала запустите heatmap_visualization.py для создания данных сетки.")
            return
        
        # Создаем интерактивную карту с прямоугольниками
        print("\n1. Создание интерактивной карты с ячейками...")
        interactive_map = create_interactive_map(grid_data, 'traffic_interactive_map.html')
        
        # Создаем тепловую карту
        print("\n2. Создание тепловой карты...")
        heatmap = create_heatmap_overlay(grid_data, 'traffic_heatmap.html')
        
        # Создаем карту с GPS точками
        print("\n3. Создание карты с GPS точками...")
        gps_map = folium.Map(
            location=[grid_data['metadata']['bounds']['lat_min'] + 
                     (grid_data['metadata']['bounds']['lat_max'] - grid_data['metadata']['bounds']['lat_min'])/2,
                     grid_data['metadata']['bounds']['lng_min'] + 
                     (grid_data['metadata']['bounds']['lng_max'] - grid_data['metadata']['bounds']['lng_min'])/2],
            zoom_start=12
        )
        add_gps_points_to_map(CSV_PATH, gps_map)
        gps_map.save('gps_points_map.html')
        
        print("\n" + "=" * 45)
        print("Карты созданы успешно!")
        print("\nСозданные файлы:")
        print("- traffic_interactive_map.html (интерактивная карта с слоями)")
        print("- traffic_heatmap.html (тепловая карта)")
        print("- gps_points_map.html (карта с GPS точками)")
        print("\nОткройте любой из HTML файлов в браузере для просмотра.")
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        print("Убедитесь, что установлен folium: pip install folium")

if __name__ == "__main__":
    main()