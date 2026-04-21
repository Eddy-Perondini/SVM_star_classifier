# Projeto | SVM_star_classifier

Nesse projeto, o objetivo principal é executar uma metodologia simples de classificação da base de dados pública (https://www.kaggle.com/datasets/deepu1109/star-dataset).

## Base de Dados | Star-Dataset 

A base de dados pública _star-dataset_ possui no total 240 estrelas, contendo um vetor de características de 6 dimensões (Temperatura ($K$), Raio ($R/R_{0}$), Luminosidade ($L/L_{0}$), Magnitude Absoluta ($M_{v}$), Cor da Estrela, Classe Espectral) e 6 classes (_Star type_) associadas à cada instância.

- Brown Dwarf -> Star Type = 0;
- Red Dwarf -> Star Type = 1;
- White Dwarf-> Star Type = 2;
- Main Sequence -> Star Type = 3;
- Supergiant -> Star Type = 4;
- Hypergiant -> Star Type = 5.

## Bibliotecas Utilizadas 

```python 

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd 
import kagglehub
```

## Performance Obtida

| Class | Precision | Recall | F1-score | Support |
|------|-----------|--------|---------|--------|
| 0 | 0.69 | 1.00 | 0.82 | 9 |
| 1 | 1.00 | 0.79 | 0.88 | 19 |
| 2 | 1.00 | 1.00 | 1.00 | 14 |
| 3 | 1.00 | 0.78 | 0.88 | 9 |
| 4 | 0.82 | 1.00 | 0.90 | 9 |
| 5 | 1.00 | 1.00 | 1.00 | 12 |

**Accuracy:** 0.92 (72 samples)

### Macro Average

| Metric | Score |
|------|------|
| Precision | 0.92 |
| Recall | 0.93 |
| F1-score | 0.91 |

### Weighted Average

| Metric | Score |
|------|------|
| Precision | 0.94 |
| Recall | 0.92 |
| F1-score | 0.92 |

**Accuracy Score:** `0.9167`

# Referências

- _star-dataset_: https://www.kaggle.com/datasets/deepu1109/star-dataset
