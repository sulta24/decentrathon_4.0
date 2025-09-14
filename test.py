
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загружаем данные
target_id = 7637058049336049989  # укажи нужный ID
df = pd.read_csv("geo_locations_astana_hackathon.csv")  # замени на свой путь

df_target = df[df['randomized_id'] == target_id].copy()
print(df['randomized_id'].nunique())

if df_target.empty:
    print(f"Для ID {target_id} нет данных")
else:
    coords = df_target[['lng', 'lat']].to_numpy()

    # Начнем с первой точки
    visited = [0]
    remaining = set(range(1, len(coords)))

    while remaining:
        last = coords[visited[-1]]
        # ищем ближайшую точку
        next_point = min(remaining, key=lambda i: np.linalg.norm(coords[i] - last))
        visited.append(next_point)
        remaining.remove(next_point)

    coords_sorted = coords[visited]

    # Цветовой градиент от начала к концу
    colors = np.linspace(0, 1, len(coords_sorted))

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(coords_sorted[:, 0], coords_sorted[:, 1],
                     c=colors, cmap='viridis', s=10)
    plt.plot(coords_sorted[:, 0], coords_sorted[:, 1], color='gray', alpha=0.5)
    plt.title(f"Маршрут для randomized_id = {target_id} (приблизительный)")
    plt.xlabel("Долгота")
    plt.ylabel("Широта")
    plt.colorbar(sc, label="Порядок точек (начало → конец)")
    plt.grid(True)
    plt.show()
