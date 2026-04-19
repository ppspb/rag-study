# Стартовый пакет: uv + git + исследование датасета

## Что внутри

- `pyproject.toml` — базовое окружение без локального тяжёлого ML-стека;
- `configs/datasets.yaml` — декларативный список стартовых датасетов;
- `scripts/profile_datasets.py` — первичный профиль датасета через `ir_datasets`;
- `scripts/smoke_lmstudio.py` — smoke-test подключения к LM Studio API;
- `docs/dataset_topology_playbook.md` — как смотреть на датасет как на topology;
- `docs/textbook_background.md` — краткая учебная справка.

## Рекомендуемая раскладка среды

- **WSL2**: git, uv, Python, скрипты, данные, анализ;
- **Windows**: LM Studio и локальные embedding models.

## Шаг 1. Создай каталог проекта и распакуй архив

```bash
mkdir -p ~/projects
cd ~/projects
unzip rag_starter_uv_git.zip -d rag_starter_uv_git
cd rag_starter_uv_git
```

## Шаг 2. Подними окружение через uv

```bash
uv python install 3.12
uv sync
```

Это создаст `.venv` и установит только безопасный базовый слой.

## Шаг 3. Подготовь git

```bash
git init
git branch -M main
git add .
git commit -m "init: rag starter with dataset profiling"
```

Если GitHub CLI уже залогинен:

```bash
gh repo create
```

Дальше выбери вариант push existing local repository.

## Шаг 4. Подготовь `.env`

```bash
cp .env.example .env
```

Потом впиши имя embedding-модели из LM Studio.

## Шаг 5. Проверь, что LM Studio API виден

На Windows включи Local Server в LM Studio.

Потом из WSL2:

```bash
uv run python scripts/smoke_lmstudio.py
```

Если всё хорошо, увидишь число векторов и размерность embedding.

## Шаг 6. Исследуй датасет до любых retrieval run

```bash
uv run python scripts/profile_datasets.py --profile starter_small
```

Артефакты появятся в `outputs/`.

## Что анализировать в первую очередь

1. Размер корпуса
2. Размер query set
3. Плотность qrels
4. Длины документов
5. Длины запросов
6. Перекос релевантности
7. Живые примеры документов и запросов

## Как думать об этом топологически

Не «у нас есть датасет», а:

- есть множество документов;
- есть множество запросов;
- есть relation relevance между ними;
- есть signal-bearing fields;
- есть неоднородность структуры.

Это и есть карта retrieval-задачи, с которой потом будут работать BM25,
dense retrieval, hybrid fusion и reranking.

## Декларативный режим работы

Не прописывай решение в коде раньше времени. Сначала фиксируй:

- какие датасеты в scope;
- какие поля документа считаем текстом;
- какие метки считаем положительными;
- какие артефакты профилирования обязательны.

То есть сначала декларации, потом execution.

## Как коммитить результаты правильно

После первого профилирования:

```bash
git add configs/datasets.yaml outputs/dataset_profile_starter_small.json outputs/dataset_profile_starter_small.csv

git commit -m "data: add initial dataset topology profile"
```

## Что делать потом

Правильный следующий шаг после этого архива:

1. прочитать `outputs/dataset_profile_*.json`;
2. выписать 3-5 гипотез о поведении BM25 на этих датасетах;
3. только потом переходить к lexical baseline.
