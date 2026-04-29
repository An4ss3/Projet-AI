# Prédiction des Tendances Boursières par NLP et Deep Learning

> Projet IA — ENSA de Fès | Module : Deep Learning & NLP | A.U. 2025/2026  
> Encadrant : Pr. Oussama EL GANNOUR

---

## Description

Pipeline hybride de prédiction des tendances boursières (GAFAM) combinant :
- **NLP** : VADER (baseline) → FinBERT (ProsusAI, modèle final)
- **Deep Learning** : LSTM → BiGRU (6 variantes expérimentales)
- **Machine Learning** : XGBoost (modèle final)
- **Feature Engineering** : RSI, SMA_14, SMA_50, MACD, Pression_News

**Résultat final** : XGBoost avec Pression_News → Accuracy **50,88%**  
**Conclusion** : Validation empirique de l'Hypothèse d'Efficience des Marchés (EMH)

---

## Structure du Projet

```
stock-nlp-prediction/
│
├── Projet_IA.ipynb          # Notebook principal (toutes les phases)
├── README.md                # Ce fichier
├── requirements.txt         # Dépendances Python
│
├── rapport_projet_NomEquipe.pdf   # Rapport final
│
└── images/                  # Figures générées pour le rapport
    ├── arch_lstm.png
    ├── arch_bigru.png
    ├── courbes_lstm_phase1.png
    ├── courbes_bigru_phase1.png
    ├── courbes_bigru_phase2.png
    ├── courbes_bigru_phase3.png
    ├── courbes_bigru_3jours.png
    ├── courbes_bigru_signaux.png
    ├── matrice_confusion_bigru.png
    ├── classification_report_bigru.png
    ├── feature_importance_xgb.png
    └── feature_importance_xgb_pression.png
```

---

## Comment Reproduire les Expériences

### 1. Prérequis

Python 3.10+ et un compte Google Colab (recommandé pour le GPU).

### 2. Installation des dépendances

```bash
pip install -r requirements.txt
```

Ou directement sur Google Colab, chaque cellule installe ses dépendances
automatiquement (`!pip install ...`).

### 3. Configuration de Kaggle (Phase 1 uniquement)

Pour télécharger le dataset Kaggle, vous avez besoin d'un fichier `kaggle.json` :

1. Créez un compte sur [kaggle.com](https://www.kaggle.com)
2. Allez dans **Account → API → Create New Token**
3. Téléchargez `kaggle.json`
4. Uploadez-le dans la première cellule du notebook

### 4. Exécution du Notebook

Ouvrez `Projet_IA.ipynb` dans Google Colab et **exécutez les cellules dans l'ordre**.

| Cellule | Phase | Description |
|---------|-------|-------------|
| 0–1  | Setup | Upload kaggle.json, téléchargement dataset |
| 2–6  | Phase 1 | Chargement, nettoyage, VADER sentiment |
| 7–10 | Phase 1 | Fusion prix yfinance, LSTM baseline |
| 11–13 | Phase 2 | BiGRU bidirectionnel |
| 14–16 | Setup Phase 3 | Indicateurs techniques + FinBERT |
| 17–20 | Phase 3 | BiGRU + FinBERT (dataset Kaggle) |
| 21–23 | Pivot | Scraping Google News + FinBERT massif |
| 24–26 | Scraping | Robot gnews (2021–2023, 5 actions) |
| 27–29 | Feature Eng. | Fusion, décalage sentiment, indicateurs |
| 30    | Phase 3 | BiGRU dataset massif (prédiction naïve) |
| 31    | Phase 4 | BiGRU tendance 3 jours |
| 32    | Phase 5 | BiGRU signaux forts filtrés |
| 33    | Phase 6 | XGBoost final |
| 34    | Phase 6+ | XGBoost + Pression_News |

> ⚠️ **Important** : Le scraping Google News (cellules 25–26) peut prendre
> **5 à 10 minutes**. L'analyse FinBERT (cellule 27) peut prendre
> **10 à 15 minutes** selon le GPU disponible.

### 5. Graine Aléatoire

Toutes les graines sont fixées pour la reproductibilité :
```python
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
```

### 6. Séparation Chronologique (Anti Data Leakage)

```python
# Décalage du sentiment d'un jour (aucune info future utilisée)
df['sentiment_lagged'] = df.groupby('stock')['sentiment_finbert'].shift(1)

# Séparation strictement chronologique (pas de shuffle)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
```

---

## Données Utilisées

| Source | Description | Volume |
|--------|-------------|--------|
| [Kaggle - Massive Stock News](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) | Headlines financières (Phase 1) | 1 407 328 articles |
| Google News (scraping gnews) | Articles GAFAM 2021–2023 | 3 011 articles |
| yfinance | Prix historiques GAFAM 2014–2024 | 12 580 entrées |

**Actions couvertes** : GOOGL, AAPL, MSFT, AMZN, META

---

## Résultats Principaux

| Phase | Modèle | NLP | Accuracy | Problème identifié |
|-------|--------|-----|----------|--------------------|
| 1 | LSTM | VADER | 0,65 | Prédiction naïve |
| 1 | BiGRU | VADER | 0,58 | Overfitting |
| 2 | BiGRU | FinBERT | 0,61 | Underfitting (10j seulement) |
| 3 | BiGRU | FinBERT | 0,53 | Prédiction naïve |
| 4 | BiGRU (3j) | FinBERT | 0,55 | Sous-apprentissage |
| 5 | BiGRU (filtré) | FinBERT | 0,50 | Overfitting (1699 lignes) |
| **6** | **XGBoost** | **FinBERT** | **0,5088** | — |

**Importance des variables (XGBoost final)** :
RSI (117) > Volume (105) > MACD (92) > ... > Pression_News (24) > sentiment_lagged (5)

---

## Environnement Technique

```
Python          3.12
TensorFlow      2.x
Transformers    5.0.0   (HuggingFace)
XGBoost         dernière stable
yfinance        dernière stable
gnews           0.4.3
scikit-learn    dernière stable
pandas          2.2.2
numpy           2.0.2
ta              0.11.0
matplotlib      dernière stable
seaborn         dernière stable
```

**Matériel** : Google Colab, GPU NVIDIA T4, RAM 12 Go

---

## Auteur

Mohammed Amine Dahmani numero 7 & Anasse Harki  numero 18 — ENSA de Fès, filière 3IACN, A.U. 2025/2026  
Encadrant : Pr. Oussama EL GANNOUR
