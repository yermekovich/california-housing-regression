```python
# =============================================
# ПРАКТИЧЕСКАЯ РАБОТА №9
# Множественная линейная регрессия
# Датасет: California Housing Prices
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import scipy.stats as stats

plt.style.use('seaborn-v0_8')

# ====================== 1. ЗАГРУЗКА ДАННЫХ ======================
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
display(df.head())

print("\nИнформация о данных:")
df.info()
```

    Размер датасета: (20640, 9)
    
    Первые 5 строк:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedHouseVal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
    </tr>
  </tbody>
</table>
</div>


    
    Информация о данных:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 9 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   MedInc       20640 non-null  float64
     1   HouseAge     20640 non-null  float64
     2   AveRooms     20640 non-null  float64
     3   AveBedrms    20640 non-null  float64
     4   Population   20640 non-null  float64
     5   AveOccup     20640 non-null  float64
     6   Latitude     20640 non-null  float64
     7   Longitude    20640 non-null  float64
     8   MedHouseVal  20640 non-null  float64
    dtypes: float64(9)
    memory usage: 1.4 MB
    


```python
# ====================== 2. EDA - Exploratory Data Analysis ======================

# 2.1 Корреляционная матрица
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Корреляционная матрица признаков')
plt.show()

# 2.2 Pairplot (ключевые признаки)
important_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
sns.pairplot(df[important_features], diag_kind='kde')
plt.suptitle('Pairplot ключевых признаков', y=1.02)
plt.show()
```


    
![png](prectice_files/prectice_1_0.png)
    



    
![png](prectice_files/prectice_1_1.png)
    



```python
# ====================== 2. EDA - Exploratory Data Analysis ======================

# 2.1 Корреляционная матрица
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Корреляционная матрица признаков')
plt.show()

# 2.2 Pairplot (ключевые признаки)
important_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
sns.pairplot(df[important_features], diag_kind='kde')
plt.suptitle('Pairplot ключевых признаков', y=1.02)
plt.show()
```


    
![png](prectice_files/prectice_2_0.png)
    



    
![png](prectice_files/prectice_2_1.png)
    



```python
# ====================== 3. Подготовка данных ======================
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
```

    Размер обучающей выборки: (16512, 8)
    Размер тестовой выборки: (4128, 8)
    


```python
# ====================== 4. Обучение модели ======================
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Метрики
print("=== Метрики качества модели ===")
print(f"R² score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
```

    === Метрики качества модели ===
    R² score: 0.5758
    RMSE: 0.7456
    MAE: 0.5332
    


```python
# ====================== 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ======================

# 5.1 Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Фактическая цена дома')
plt.ylabel('Предсказанная цена дома')
plt.title('Predicted vs Actual Values')
plt.show()

# 5.2 Residuals vs Fitted
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('Residuals vs Fitted')
plt.show()

# 5.3 Q-Q Plot
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()
```


    
![png](prectice_files/prectice_5_0.png)
    



    
![png](prectice_files/prectice_5_1.png)
    



    
![png](prectice_files/prectice_5_2.png)
    



```python
# ====================== 6. Анализ коэффициентов ======================
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nКоэффициенты модели (влияние признаков):")
display(coefficients)

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
sns.barplot(data=coefficients, x='Coefficient', y='Feature')
plt.title('Влияние признаков на цену дома')
plt.show()
```

    
    Коэффициенты модели (влияние признаков):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>AveBedrms</td>
      <td>0.783145</td>
    </tr>
    <tr>
      <th>0</th>
      <td>MedInc</td>
      <td>0.448675</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HouseAge</td>
      <td>0.009724</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Population</td>
      <td>-0.000002</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AveOccup</td>
      <td>-0.003526</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AveRooms</td>
      <td>-0.123323</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Latitude</td>
      <td>-0.419792</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Longitude</td>
      <td>-0.433708</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](prectice_files/prectice_6_2.png)
    

