# Расчет метрики  📈

В качестве метрики качества для этого соревнования используется средняя метрика попадания или средняя точность — **Mean Accuracy (mAcc)**. mAcc равна средней точности оценок для каждого класса по всей выборке:

$$ mAcc = \dfrac{1}{N}\sum_{i=0}^{N-1}\dfrac{TP_i}{TN_i + FN_i} $$

В этой формуле: `mAcc` — средняя точность, `TP (True Positive)` — правильно предсказанные положительные ответы, `FN (False Negaive)` — ложноотрицательные ответы, `N` — число классов.
Высокое значение оценки показывает, что модель хорошо справляется с задачей классификации в среднем для всех классов. Значение метрики будет округлено до 5 знака после запятой. В случае одинакового значения метрики позиция в рейтинге будет выше у того решения, которое было раньше загружено на платформу.

Псевдокод расчета метрики:
```python
import numpy as np
from sklearn.metrics import confusion_matrix

def mean_class_accuracy(predicts: list[int], labels: list[int]) -> np.ndarray:
    """Calculate mean class accuracy.

    Args:
        predicts (list[int]): Prediction labels for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """

    conf_matrix = confusion_matrix(y_pred=predicts, y_true=labels)

    cls_cnt = conf_matrix.sum(axis=1) # all labels
    cls_hit = np.diag(conf_matrix) # true positives

    metrics = [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)]
    mean_class_acc = np.mean(metrics)

    return mean_class_acc
```