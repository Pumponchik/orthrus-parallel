# Установка и запуск

Команды выполняются из корня репозитория.


Поддерживаемая проверенная среда: **Linux x86_64, NVIDIA GPU, Python 3.11.17, PyTorch 2.6.0+cu118, transformers 5.12.1**. Все зависимости закреплены в `uv.lock`. На Mac можно читать и проверять структуру проекта; CUDA-инференс выполняется на Linux-сервере.

```bash
# Один раз: установить локальный uv, затем окружение. Системный Python не меняется.
python3 bootstrap_uv.py
./lab sync --locked --no-dev --no-build

# Сначала показать размер и место назначения, затем явно скачать checkpoint.
./lab run --locked python scripts/download_model.py
./lab run --locked python scripts/download_model.py --download

# Генерация: 48 выбранных ветвей.
./lab run --locked python -m orthrus_parallel --method selected48 \
  --prompt 'Explain binary search with a Python example.' --max-new-tokens 256

# Обычный Orthrus для сравнения.
./lab run --locked python -m orthrus_parallel --method baseline16 \
  --prompt 'Explain binary search with a Python example.' --max-new-tokens 256
```

Окружение потребует несколько гигабайт загрузок; checkpoint с токенизатором — **1 555 948 837 байт**. Downloader показывает план без загрузки, фиксирует ревизию, проверяет размер и хеш каждого файла. Если модель уже есть, укажите `--checkpoint /path/to/checkpoint`; повторная копия не нужна. Выбирайте свободную GPU через `--gpu 0`.

`--prompt-file` читает запрос из файла, `--output runs/answer.json` сохраняет результат и счётчики, `--trace` добавляет детали циклов. Время одиночного холодного запроса не заменяет прогретый benchmark.

