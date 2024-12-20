[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2AJIyfdd)
## Dataset

`World Happiness Report 2024`

# import библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = '/content/World-happiness-report-2024.csv'  
data = pd.read_csv(data_path)

data.head()

# 1. Очистка данных
print("Количество пропущенных значений:")
print(data.isnull().sum())

num_cols = data.select_dtypes(include=[np.number]).columns
data[num_cols] = data[num_cols].apply(lambda x: x.fillna(x.median()), axis=0)

print(f"Количество дубликатов до удаления: {data.duplicated().sum()}")
data = data.drop_duplicates()
print(f"Количество дубликатов после удаления: {data.duplicated().sum()}")

data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# 2. Анализ EDA
print("Основные статистики для числовых переменных:")
print(data.describe())

cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"Частоты значений в колонке {col}:")
    print(data[col].value_counts())

plt.figure(figsize=(10, 6))
data['ladder_score'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Распределение Ladder Score')
plt.xlabel('Ladder Score')
plt.ylabel('Частота')
plt.show()

numeric_data = data.select_dtypes(include=[np.number]) 
corr_matrix = numeric_data.corr()  
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица для числовых данных')
plt.show()



# 3. Трансформация данных и создание признаков 
if 'upperwhisker' in data.columns and 'lowerwhisker' in data.columns:
    data['whisker_range'] = data['upperwhisker'] - data['lowerwhisker']


high_happiness = data[data['ladder_score'] > data['ladder_score'].median()]
low_happiness = data[data['ladder_score'] <= data['ladder_score'].median()]

# 4.Пользовательские функции и сводные таблицы
normalize = lambda x: (x - x.min()) / (x.max() - x.min())
data['normalized_ladder_score'] = normalize(data['ladder_score'])


pivot_table = data.pivot_table(values='ladder_score', index='regional_indicator', aggfunc=['mean', 'max', 'min'])
print("Сводная таблица по региональным индикаторам:")
print(pivot_table)

# 5. Визуализация топ стран с высоким и низким уровнем счастья
top_10_high = data.nlargest(10, 'ladder_score')
plt.figure(figsize=(12, 6))
sns.barplot(x='ladder_score', y='country_name', data=top_10_high, palette='viridis')
plt.title('Топ-10 стран с самым высоким Ladder Score')
plt.xlabel('Ladder Score')
plt.ylabel('Country')
plt.show()

top_10_low = data.nsmallest(10, 'ladder_score')
plt.figure(figsize=(12, 6))
sns.barplot(x='ladder_score', y='country_name', data=top_10_low, palette='magma')
plt.title('Топ-10 стран с самым низким Ladder Score')
plt.xlabel('Ladder Score')
plt.ylabel('Country')
plt.show()


important_factors = ['log_gdp_per_capita', 'social_support', 'healthy_life_expectancy', 'freedom_to_make_life_choices', 'generosity', 'perceptions_of_corruption']
plt.figure(figsize=(12, 8))
for factor in important_factors:
    sns.regplot(x=factor, y='ladder_score', data=data, label=factor, scatter_kws={'alpha': 0.5})
plt.legend()
plt.title('Сравнение факторов, влияющих на уровень счастья')
plt.xlabel('Факторы')
plt.ylabel('Ladder Score')
plt.show()

data.head()

output_path = '/content/World-happiness-report-2024.csv'
data.to_csv(output_path, index=False)
print(f"Обработанные данные сохранены в {output_path}")

