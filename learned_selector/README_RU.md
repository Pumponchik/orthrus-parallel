# Новый селектор: 48 ветвей из расширенного набора

Блоки draft и ветвей — по 16 токенов. Выбор среди первых 32 возможных якорей на каждой из 16 границ; вычисляются 48 ветвей суммарно. Общая подготовка скорера оценивает 16×64 пары, затем исключает ранги выше 32.

На вход: вероятности draft, ранг/позиция/уверенность, hidden state draft и embedding конкретного кандидата. Модель классифицирует пару «сколько предложений примут + какой токен AR выдаст следующим». Дополнительного прохода большой модели нет. Обученные веса и фиксированные проекции приложены.

## Запуск

Из корня репозитория на Linux x86_64 / NVIDIA CUDA:

```bash
uv sync --project learned_selector --locked
CUDA_VISIBLE_DEVICES=0 uv run --project learned_selector --locked --no-sync python \
  learned_selector/infer.py --checkpoint /path/to/ZaryaOrthrus-0.6B \
  --prompt 'Explain binary search briefly.' --max-new-tokens 256 \
  --output /tmp/orthrus-result.json
```

Checkpoint скачивается отдельно. Нужен `ai-forever/ZaryaOrthrus-0.6B`, revision `6c93071b27ab0d2061007dda0416f0439ed4d7df`; файл весов SHA256 `c4c18aa39cfabe07cf8b6d2cfdca53f8bffcb1f9abbacba1320e6442893babe3`. Используйте существующий checkpoint или [downloader](../docs/setup.md) с проверкой размера и хэшей. Loader исполняет оригинальный код checkpoint с `trust_remote_code=True`.

Среда закреплена в `pyproject.toml` и `uv.lock`: Python 3.11, Torch 2.6.0+cu118, Transformers 5.12.1. Дополнительная группа `selector` содержит CatBoost для исследовательских сравнений; выбранному инференсу она не нужна. Системный Python и драйвер не меняются.

`--prompt-file` принимает UTF-8-файл вместо `--prompt`. JSON содержит ответ, сгенерированные IDs, время, число токенов и счётчики. Одиночный холодный запуск не является прогретым бенчмарком.

Измерено на RTX 2080 Ti: 69,57 токена/с, +1,68% [0,79; 2,61]% к прежнему селектору. Цель 3% не достигнута. Новый и прежний варианты совпали 60/60 раз; с обычным Orthrus различались 5/30 запросов. Универсальная lossless-эквивалентность не установлена.

[Отчёт](../docs/selector_training_20261015.md) · [Аудит поставки](../results/selector_training_20261015/publication_validation.json)
