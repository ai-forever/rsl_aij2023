## Baseline решение
За основу предлагается взять подходы для решения задачи [Action Classification](https://paperswithcode.com/task/action-classification).
Для обучения baseline модели использовался фреймворк [mmaction2](https://github.com/open-mmlab/mmaction2)

## Запуск инференса
Чтобы воспроизвести бейзлайн, после установки зависимостей 
```bash
pip install requirements.txt
```

необходимо скачать [веса модели](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/aij2023/mvit32.2_small.pth)

прописать в скрипте solution.py пути к весам модели и к папке с видео
```
CHECKPOINT = <path_to_checkpoint>
DATASET_DIR = <path_to_dataset>
```

## Запуск обучения
Чтобы обучить модель с помощью фреймворка mmaction2 вам потребуется:

1. Следовать [инструкции по установке](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html)
2. Использовать за основу конфиги для задачи [Action Recognition](https://github.com/open-mmlab/mmaction2/tree/dev-1.x/configs/recognition)
3. Изменяемые параметры ```clip_len``` и ```frame_interval``` установить на нужное значение
4. Убрать аугментации MixupBlending и CutmixBlending, т. к. они не предназначены для задачи Распознавания Жесвтового Языка
4. Использовать датасет Slovo для обучения модели

