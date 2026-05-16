# megaminx-thesis

Проект для дипломных экспериментов по сравнению двух эвристик в beam search на графе Кэли мегаминкса:

- `State Model`: baseline и teacher, предсказывает оценку одного состояния;
- `Neighbour Model`: дистиллированная модель, предсказывает оценки сразу для всех соседей текущего состояния;
- главный критерий: solver-level качество на sweep по ширине луча, с последующим сравнением длины решения с фактическим временем и FLOPs.

## Структура

- `megathesis/`: Python-код экспериментов.
- `scripts/`: отдельные bash-скрипты для каждого эксперимента.
- `configs/default.env`: параметры по умолчанию.
- `data/generators/p900.json`: UTM move table.
- `data/targets/p900-t000.pt`: solved state.
- `datasets/superflip.pt`: отдельный case study, не часть основного test set.
- `datasets/`, `checkpoints/`, `logs/`: генерируемые артефакты.

## Установка

```bash
cd /home/ananasclassic/projects/megaminx_solver/megaminx-thesis
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

На сервере с CUDA можно заранее поставить нужный wheel PyTorch под конкретную версию CUDA, затем выполнить `pip install -r requirements.txt`.

## Быстрая проверка

```bash
cd megaminx-thesis
DEVICE=cpu scripts/99_smoke.sh
```

## Основной пайплайн

```bash
scripts/01_make_datasets.sh
scripts/02_train_state.sh
scripts/03_eval_state.sh
scripts/04_make_neighbour_labels.sh
scripts/05_train_neighbour.sh
scripts/06_eval_neighbour.sh
scripts/07_fixed_beam.sh
scripts/08_beam_sweep.sh
scripts/10_ablation_depth.sh
scripts/11_superflip.sh
scripts/12_summarize.sh
```

Параметры меняются через переменные окружения:

```bash
METRIC=UTM DEVICE=cuda:0 TRAIN_PER_DEPTH=1 VAL_PER_DEPTH=10000 scripts/01_make_datasets.sh
DEVICE=cuda:0 STATE_EPOCHS=1024 STATE_STEPS_PER_EPOCH=512 scripts/02_train_state.sh
DEVICE=cuda:0 NEIGH_EPOCHS=4096 NEIGH_STEPS_PER_EPOCH=16 scripts/05_train_neighbour.sh
BEAM_WIDTH=65536 MAX_DEPTH=200 TESTS=100 scripts/07_fixed_beam.sh
BEAM_WIDTHS="4096 8192 16384 32768 65536 131072 262144" TESTS=100 scripts/08_beam_sweep.sh
```

Для `FTM` запускать отдельную серию:

```bash
METRIC=FTM scripts/01_make_datasets.sh
METRIC=FTM scripts/02_train_state.sh
METRIC=FTM scripts/04_make_neighbour_labels.sh
METRIC=FTM scripts/05_train_neighbour.sh
METRIC=FTM scripts/08_beam_sweep.sh
```

`UTM` и `FTM` используют разные validation/test/search датасеты, разные teacher labels для validation и разные `latest_*` чекпоинты. Во время обучения `State Model` и `Neighbour Model` генерируют новые random walks каждую эпоху; сохраненный `state_train.pt` задает только метрику и depth buckets, поэтому `TRAIN_PER_DEPTH=1` достаточно. Длины решений сравниваются только внутри одной метрики ходов.

## Что пишется в логи

Model-level:

- `MSE`, `MAE`, `Pearson`, `Spearman`;
- для `State Model`: локальное ранжирование good move, где good move является обратным последнему ходу random walk;
- для `Neighbour Model`: vector MSE/MAE, top-1/top-3/top-5 относительно teacher ranking, mean rank.

Solver-level:

- `success_rate`;
- mean/median solution length among solved;
- mean/median/p90 solve time;
- `expanded_states`;
- `generated_candidates`;
- `unique_candidates`;
- `model_inputs`;
- `model_flops`;
- `mean_model_flops`;
- `evaluated_moves_per_sec`;
- `model_inputs_per_sec`;
- `model_flops_per_sec`.

`scripts/08_beam_sweep.sh` пишет JSON для каждого `B`, CSV `logs/beam_sweep_<metric>.csv` и PNG `logs/beam_sweep_<metric>.png` с графиками `length~B`, `length~time`, `time~B`, `length~FLOPs`. Именно JSON/CSV из sweep, `logs/depth_*` и `logs/superflip_*` стоит использовать как основные таблицы для диплома.
