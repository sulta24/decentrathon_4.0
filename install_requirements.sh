#!/bin/bash

# Создание виртуального окружения (если не существует)
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Активация окружения
source venv/bin/activate

# Обновление pip
pip install --upgrade pip

# Установка зависимостей
pip install -r requirements.txt

echo "Зависимости установлены успешно!"
echo "Для активации окружения используйте: source venv/bin/activate"