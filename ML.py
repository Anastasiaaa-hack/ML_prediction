

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from scipy.stats import bartlett
import seaborn as sns
import matplotlib.pyplot as plt
from kerastuner.tuners import BayesianOptimization
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import json
import matplotlib.pyplot as plt

data = pd.read_csv('', sep=';')
data = pd.DataFrame(data)
data = data.dropna()


df = data.drop(columns=['Date', 'Rank', 'Country'])
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
print(outliers.sum())
sns.boxplot(df["Climate Index"])

top_outliers = data.nlargest(10, 'Safety Index')[['Country', 'Date', 'Safety Index']]
print(top_outliers)

top_outliers = data.nsmallest(10, 'Climate Index')[['Country', 'Date', 'Climate Index']]
print(top_outliers)

# Замена на числовые коды
data['Country_Code'] = pd.factorize(data['Country'])[0]

# Преобразование в datetime и затем в timestamp
data['date_num'] = pd.to_datetime(data['Date']).astype('int64') // 10**9

#data.head()
data[data['Country']=='Germany']

#df = data.drop(columns=['Date', 'Country', 'Is_Fire', 'Is_Crisis'])
df = data.drop(columns=['Date', 'Country'])
test = data.drop(columns=['Date', 'Country', 'Rank', 'Country_Code', 'date_num'])

plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(test.corr(), annot=True, annot_kws={"size": 8},
                     cbar_kws={"shrink": 0.8}, fmt=".2f")
plt.xticks(rotation=45, ha='right')  # Поворот подписей оси X
plt.yticks(rotation=0)
plt.tight_layout()  # Автоматическая настройка отступов
plt.savefig("")

df_x = data.drop(columns=['Date', 'Country', 'Rank', 'Country_Code', 'date_num', 'Quality of Life Index'])

#проверка мультиколлинеарности
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["Признак"] = df_x.columns
vif_data["VIF"] = [variance_inflation_factor(df_x.values, i) for i in range(df_x.shape[1])]
print(vif_data)

df_x = df_x.drop(columns=['Health Care Index', 'Traffic Commute Time Index'])

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Признак"] = df_x.columns
vif_data["VIF"] = [variance_inflation_factor(df_x.values, i) for i in range(df_x.shape[1])]
print(vif_data)

df_x['Economic_Stress'] = df['Cost of Living Index'] / df['Purchasing Power Index']
df_x = df_x.drop(columns=['Cost of Living Index', 'Purchasing Power Index'])

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Признак"] = df_x.columns
vif_data["VIF"] = [variance_inflation_factor(df_x.values, i) for i in range(df_x.shape[1])]
print(vif_data)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
df_x['Safety_Pollution_PC'] = pca.fit_transform(df_x[['Safety Index', 'Pollution Index']])
df_x = df_x.drop(columns=['Safety Index', 'Pollution Index'])

print(pca.components_)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Признак"] = df_x.columns
vif_data["VIF"] = [variance_inflation_factor(df_x.values, i) for i in range(df_x.shape[1])]
print(vif_data)

plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(df_x.corr(), annot=True, annot_kws={"size": 10},
                     cbar_kws={"shrink": 0.8}, fmt=".2f")
plt.xticks(rotation=45, ha='right')  # Поворот подписей оси X
plt.yticks(rotation=0)
plt.tight_layout()  # Автоматическая настройка отступов
plt.savefig("")

df_final = data.drop(columns=['Date', 'Country', 'Rank', 'Health Care Index', 'Traffic Commute Time Index'])
df_final['Economic_Stress'] = data['Cost of Living Index'] / data['Purchasing Power Index']
df_final = df_final.drop(columns=['Cost of Living Index', 'Purchasing Power Index'])
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
df_final['Safety_Pollution_PC'] = pca.fit_transform(df_final[['Safety Index', 'Pollution Index']])
df_final = df_final.drop(columns=['Safety Index', 'Pollution Index'])
df_final.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_final)

#Критерий Кайзера-Мейера-Олкина (KMO) и Тест Бартлетта
#KMO > 0.6 и p-value < 0.05 — данные пригодны для анализа.
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
kmo_all, kmo_model = calculate_kmo(X_scaled)
bartlett, p_value = calculate_bartlett_sphericity(X_scaled)
print(f"KMO: {kmo_model}, Bartlett p-value: {p_value}")

df_final['Following QLI'] = None
for i in range(1589):
    current_country = df_final.loc[i, 'Country_Code']
    qli_list = [df_final.loc[i, 'Quality of Life Index']]
    temp = 1
    for j in range(i + 1, len(df_final)):
        if df_final.loc[j, 'Country_Code'] == current_country:
            qli_list.append(df_final.loc[j, 'Quality of Life Index'])
            temp += 1
        if temp == 3:
            break
    df_final.at[i, 'Following QLI'] = qli_list

df_final.head()

df_final = df_final.drop(columns=['Quality of Life Index'])

# Функция для создания отрезков
def create_segments(df, segment_length=5):
    segments = []
    labels = []

    # Убедимся, что данные отсортированы по стране и дате
    df = df.sort_values(by=['Country_Code', 'date_num'])

    for country in df['Country_Code'].unique():
        country_data = df[df['Country_Code'] == country]
        for i in range(len(country_data) - segment_length + 1):
            # Выбираем 5 строк подряд
            segment = country_data.iloc[i:i+segment_length]
            # Убираем ненужные столбцы (Following QLI, Country_Code, date_num)
            segment_data = segment.drop(columns=['Following QLI', 'Country_Code', 'date_num']).values
            # Проверяем, что Following QLI содержит ровно 3 значения
            following_qli = segment.iloc[-1]['Following QLI']
            if isinstance(following_qli, list) and len(following_qli) == 3:
                segments.append(segment_data)
                labels.append(following_qli)

    # Преобразуем в массивы NumPy
    segments = np.array(segments)
    labels = np.array(labels)

    return segments, labels

# Создание отрезков
X, y = create_segments(df_final)

df_final[df['Country_Code']==0]

X[0]

y[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# 1. Улучшенная нормализация (попробуем MinMax для выходных)
scaler_x = StandardScaler()
scaler_y = MinMaxScaler()

scaled_X_train = scaler_x.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
scaled_X_test = scaler_x.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Нормализуем выходные данные если они еще не нормализованы
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print(scaled_X_train[0])

print(y)
print(scaler_y.fit_transform(y))

print(scaler_y.fit_transform(y))

# 1. Проверка и преобразование формы данных
print("Original X_train shape:", X_train.shape)

# 1. Проверка данных
print("Форма данных до обработки:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# 1. Упрощенная и стабильная модель
def build_model_0(hp):
    model = keras.Sequential()

    # Простой LSTM слой
    model.add(layers.LSTM(
        units=hp.Int('units', min_value=16, max_value=64, step=16),
        input_shape=(scaled_X_train.shape[1], scaled_X_train.shape[2]),
        activation='tanh',  # Более стабильная активация
        return_sequences=False))

    # BatchNormalization для стабильности
    model.add(layers.BatchNormalization())

    # Один Dense слой
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=8, max_value=32, step=8),
        activation='relu'))

    # Выходной слой
    model.add(layers.Dense(20, activation='linear'))

    # Компиляция с меньшим learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('lr', min_value=1e-4, max_value=1e-3, sampling='log')),
        loss='mse',
        metrics=['mae'])

    return model

# 2. Улучшенная архитектура модели
def build_model(hp):
    model = keras.Sequential()

    # LSTM слой с регуляризацией
    model.add(layers.LSTM(
        units=hp.Int('units', 32, 128, step=16),
        input_shape=(scaled_X_train.shape[1], scaled_X_train.shape[2]),
        activation='tanh',
        recurrent_dropout=hp.Float('recurrent_dropout', 0.0, 0.3, step=0.1),
        return_sequences=False))

    model.add(layers.BatchNormalization())

    # Добавляем Dropout
    model.add(layers.Dropout(
        hp.Float('dropout_rate', 0.0, 0.5, step=0.1)))

    # Dense слои
    for i in range(hp.Int('num_dense', 1, 2)):
        model.add(layers.Dense(
            units=hp.Int(f'dense_units_{i}', 16, 64, step=16),
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(
                hp.Float('l2_reg', 1e-4, 1e-2, sampling='log'))))
        model.add(layers.BatchNormalization())

    # Выходной слой
    model.add(layers.Dense(20, activation='linear'))

    optimizer = keras.optimizers.Adam(
        learning_rate=hp.Float('lr', 1e-4, 1e-3, sampling='log'))

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae'])

    return model

# 3. Улучшенная архитектура модели
def build_model_3(hp):
    model = keras.Sequential()

    # LSTM слой с регуляризацией
    model.add(layers.LSTM(
        units=hp.Int('units', 32, 128, step=16),
        input_shape=(scaled_X_train.shape[1], scaled_X_train.shape[2]),
        activation='tanh',
        recurrent_dropout=hp.Float('recurrent_dropout', 0.0, 0.3, step=0.1),
        return_sequences=True))

    model.add(layers.BatchNormalization())

    # Dropout
    model.add(layers.Dropout(
        hp.Float('dropout_rate', 0.0, 0.5, step=0.1)))

    # LSTM 2 слой с регуляризацией
    model.add(layers.LSTM(
        units=hp.Int('units', 16, 64, step=16),
        input_shape=(scaled_X_train.shape[1], scaled_X_train.shape[2]),
        activation='tanh',
        recurrent_dropout=hp.Float('recurrent_dropout', 0.0, 0.3, step=0.1),
        return_sequences=False))

    model.add(layers.BatchNormalization())

    # Dropout
    model.add(layers.Dropout(
        hp.Float('dropout_rate', 0.0, 0.5, step=0.1)))

    # Dense слои
    for i in range(hp.Int('num_dense', 1, 2)):
        model.add(layers.Dense(
            units=hp.Int(f'dense_units_{i}', 16, 64, step=16),
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(
                hp.Float('l2_reg', 1e-4, 1e-2, sampling='log'))))
        model.add(layers.BatchNormalization())

    # Выходной слой
    model.add(layers.Dense(20, activation='linear'))

    optimizer = keras.optimizers.Adam(
        learning_rate=hp.Float('lr', 1e-4, 1e-3, sampling='log'))

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae'])

    return model

def build_model_4(hp):
    model = keras.Sequential()

    # Двунаправленная LSTM
    model.add(layers.Bidirectional(
        layers.LSTM(
            hp.Int('units', 32, 128, step=32),
            recurrent_dropout=hp.Float('rec_drop', 0.1, 0.3)),
        input_shape=(scaled_X_train.shape[1], scaled_X_train.shape[2])))

    # Регуляризация
    model.add(layers.Dropout(hp.Float('dropout', 0.2, 0.5)))

    # Dense-слои с L1/L2 регуляризацией
    for i in range(hp.Int('num_dense', 1, 2)):
        model.add(layers.Dense(
            hp.Int(f'dense_{i}_units', 16, 64, step=16),
            kernel_regularizer=keras.regularizers.l1_l2(
                l1=hp.Float(f'l1_{i}', 1e-5, 1e-3),
                l2=hp.Float(f'l2_{i}', 1e-5, 1e-3))))

    # Выходной слой
    model.add(layers.Dense(20, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.AdamW(
            hp.Float('lr', 1e-5, 1e-3),
            weight_decay=hp.Float('wd', 1e-6, 1e-4)),
        loss='mse',
        metrics=['mae'])
    return model

import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

# --- 1. Настройки моделей ---
MODELS = {
    "Base_LSTM": build_model_0,
    "Regularized_LSTM": build_model,
    "Stacked_LSTM": build_model_2,
    "Bidirectional_LSTM": build_model_4,
}

# --- 2. Функция для обучения и сбора результатов ---
def run_hyperparameter_tuning(models, scaled_X_train, y_train_scaled, scaled_X_test, y_test_scaled):
    results = {}  # Инициализация
    all_histories = {}  # Инициализация

    for model_name, hypermodel in models.items():
        print(f"\n🔧 Tuning {model_name}")

        tuner = BayesianOptimization(
            hypermodel,
            objective='val_loss',
            max_trials=15,
            num_initial_points=3,
            directory='tuning',
            project_name=model_name
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        tuner.search(
            scaled_X_train,
            y_train_scaled,
            epochs=100,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"\n✅ Best hyperparameters for {model_name}:")
        for param, value in best_hps.values.items():
            print(f"    {param}: {value}")

        best_model = tuner.hypermodel.build(best_hps)

        history = best_model.fit(
            scaled_X_train,
            y_train_scaled,
            epochs=100,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        eval_result = best_model.evaluate(scaled_X_test, y_test_scaled)
        print(f"\n📈 Test loss for {model_name}: {eval_result}")

        y_pred = best_model.predict(scaled_X_test)
        results[model_name] = {
            'best': {
                'mse': mean_squared_error(y_test_scaled, y_pred),
                'mae': mean_absolute_error(y_test_scaled, y_pred),
                'params': best_hps.values,
                'epochs': len(history.history['val_loss']),
            }
        }

        all_histories[model_name] = history.history['val_loss']
        plot_model_history(model_name, history)

    plot_combined_val_loss(all_histories)

    # Сохраняем все результаты в JSON
    with open('', 'w') as f:
        json.dump(results, f, indent=2)

    return results

# --- 3. Функция для визуализации одной модели ---
def plot_model_history(model_name, history):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(history.history['val_loss']) + 1)

    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.title(f'Изменение значения функции потерь по эпохам для {model_name}', pad=20)
    plt.xlabel('Эпоха', labelpad=10)
    plt.ylabel('Значение функции потерь (MSE)', labelpad=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'', dpi=120)
    plt.close()

# --- 4. Функция для общего сравнения моделей ---
def plot_combined_val_loss(histories):
    plt.figure(figsize=(12, 8))

    max_epochs = max(len(h) for h in histories.values())

    for model_name, val_loss in histories.items():
        epochs = range(1, len(val_loss) + 1)
        plt.plot(epochs, val_loss, label=model_name, linewidth=2)

    plt.title('Сравнение динамики валидационной ошибки (Loss) для разных моделей', pad=20)
    plt.xlabel('Эпоха', labelpad=10)
    plt.ylabel('Значение валидационной функции потерь (MSE)', labelpad=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(1, max_epochs)

    plt.axhline(y=min(min(h) for h in histories.values()),
                color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('',
                dpi=120, bbox_inches='tight')
    plt.close()

# --- 5. Сохраняем результаты в Excel ---
def save_results_to_excel(results):
    rows = []

    for model, data in results.items():
        rows.append({
            'Model': model,
            'MSE': data['best']['mse'],
            'MAE': data['best']['mae'],
            'Epochs': data['best']['epochs'],
            'Params': json.dumps(data['best']['params'])
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by='MSE')

    output_path = '/content/drive/MyDrive/8 семестр/Диплом/Вывод/3/final_results.xlsx'
    df.to_excel(output_path, index=False)

    print(f"\n✅ Excel файл сохранён по адресу: {output_path}")
    print("\n📊 TOP-3 моделей по MSE:")
    print(df.head(3))

# --- 6. Полный запуск пайплайна ---
final_results = run_hyperparameter_tuning(MODELS, scaled_X_train, y_train_scaled, scaled_X_test, y_test_scaled)

# Сохраняем Excel
save_results_to_excel(final_results)

print("\n🚀 Полный процесс завершён!")

data = pd.read_csv('', sep=';')
data = pd.DataFrame(data)
data = data.dropna()
# Замена на числовые коды
data['Country_Code'] = pd.factorize(data['Country'])[0]
# Преобразование в datetime и затем в timestamp
data['date_num'] = pd.to_datetime(data['Date']).astype('int64') // 10**9
df_final = data.drop(columns=['Date', 'Country', 'Rank', 'Health Care Index', 'Traffic Commute Time Index'])
df_final['Economic_Stress'] = data['Cost of Living Index'] / data['Purchasing Power Index']
df_final = df_final.drop(columns=['Cost of Living Index', 'Purchasing Power Index'])
pca = PCA(n_components=1)
df_final['Safety_Pollution_PC'] = pca.fit_transform(df_final[['Safety Index', 'Pollution Index']])
df_final = df_final.drop(columns=['Safety Index', 'Pollution Index'])
df_final.head()
df_final['Following QLI'] = None
for i in range(1422):
    current_country = df_final.loc[i, 'Country_Code']
    qli_list = [df_final.loc[i, 'Quality of Life Index']]
    temp = 1
    for j in range(i + 1, len(df_final)):
        if df_final.loc[j, 'Country_Code'] == current_country:
            qli_list.append(df_final.loc[j, 'Quality of Life Index'])
            temp += 1
        if temp == 3:
            break
    df_final.at[i, 'Following QLI'] = qli_list
df_final.head()
df_final = df_final.drop(columns=['Quality of Life Index'])
# Функция для создания отрезков
def create_segments(df, segment_length=5):
    segments = []
    labels = []

    # Убедимся, что данные отсортированы по стране и дате
    df = df.sort_values(by=['Country_Code', 'date_num'])

    for country in df['Country_Code'].unique():
        country_data = df[df['Country_Code'] == country]
        for i in range(len(country_data) - segment_length + 1):
            # Выбираем 5 строк подряд
            segment = country_data.iloc[i:i+segment_length]
            # Убираем ненужные столбцы (Following QLI, Country_Code, date_num)
            segment_data = segment.drop(columns=['Following QLI', 'Country_Code', 'date_num']).values
            # Проверяем, что Following QLI содержит ровно 5 значения
            following_qli = segment.iloc[-1]['Following QLI']
            if isinstance(following_qli, list) and len(following_qli) == 3:
                segments.append(segment_data)
                labels.append(following_qli)

    # Преобразуем в массивы NumPy
    segments = np.array(segments)
    labels = np.array(labels)

    return segments, labels

# Создание отрезков
X, y = create_segments(df_final)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Улучшенная нормализация (попробуем MinMax для выходных)
scaler_x = StandardScaler()
scaler_y = MinMaxScaler()

scaled_X_train = scaler_x.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
scaled_X_test = scaler_x.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Нормализуем выходные данные если они еще не нормализованы
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# входные данные уже подготовлены:
def create_lagged_features(data, n_lags):
    n_samples, n_timesteps, n_features = data.shape
    X_lagged = np.zeros((n_samples, n_lags * n_features))

    for i in range(n_samples):
        for lag in range(n_lags):
            start_idx = lag * n_features
            end_idx = (lag + 1) * n_features
            X_lagged[i, start_idx:end_idx] = data[i, - (lag + 1), :]  # Берём последние `n_lags` шагов

    return X_lagged

n_lags = 3  # Сколько последних шагов учитывать
X_train_rf = create_lagged_features(scaled_X_train, n_lags)
X_test_rf = create_lagged_features(scaled_X_test, n_lags)


# Определяем модели и их сетки гиперпараметров
model_params = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}  # Нет гиперпараметров для перебора
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    },
    'Lasso Regression': {
        'model': Lasso(),
        'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    },
    'Polynomial Regression': {
        'model': PolynomialFeatures(),
        'params': {
            'degree': [2, 3, 4]  # Степень полинома для регрессии
        }
    }
}

# Обучение моделей с подбором параметров
for name, mp in model_params.items():
    print(f'Подбор параметров для {name}...')

    if name == 'Polynomial Regression':
        # Для полиномиальной регрессии, преобразуем признаки перед линейной регрессией
        poly = PolynomialFeatures(degree=mp['params']['degree'][0])  # Используем первый уровень из параметров
        X_poly_train = poly.fit_transform(X_train_rf)
        X_poly_test = poly.transform(X_test_rf)

        model = LinearRegression()
        model.fit(X_poly_train, y_train_scaled)

        y_pred = model.predict(X_poly_test)
    else:
        grid = GridSearchCV(mp['model'], mp['params'], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_rf, y_train_scaled)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_rf)

    mse = mean_squared_error(y_test_scaled, y_pred)
    mae = mean_absolute_error(y_test_scaled, y_pred)

    print(f'{name}:')
    print(f'  Лучшие параметры: {grid.best_params_}')
    print(f'  MSE: {mse:.8f}')
    print(f'  MAE: {mae:.8f}\n')



import matplotlib.pyplot as plt
import random

indices = random.sample(range(len(y_test_scaled_10)), num_examples)

# Построим графики
for idx in indices:
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_scaled_10[idx], label='Реальные значения', linewidth=2, color='black')

    for name, y_pred in results.items():
        plt.plot(y_pred[idx], label=name)

    plt.title(f'Прогноз на 5 лет (пример {idx})')
    plt.xlabel('Горизонт прогноза (шаг)')
    plt.ylabel('Значение (норм.)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_lagged_features(data, n_lags):
    n_samples, n_timesteps, n_features = data.shape
    X_lagged = np.zeros((n_samples, n_lags * n_features))

    for i in range(n_samples):
        for lag in range(n_lags):
            start_idx = lag * n_features
            end_idx = (lag + 1) * n_features
            X_lagged[i, start_idx:end_idx] = data[i, - (lag + 1), :]  # Берём последние `n_lags` шагов

    return X_lagged

# Пример:
n_lags = 3  # Сколько последних шагов учитывать
X_train_rf_10 = create_lagged_features(scaled_X_train_10, n_lags)
X_test_rf_10 = create_lagged_features(scaled_X_test_10, n_lags)

print("X_train_rf shape:", X_train_rf.shape)  # (n_samples, n_lags * n_features)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Параметры для GridSearch
param_grid = {
      'n_estimators': [30, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [5, 10, 15],
}

# Поиск лучшей модели
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_rf_10, y_train_scaled_10)

# Лучшая модель
best_rf_10 = grid_search.best_estimator_
y_pred_rf_10 = best_rf_10.predict(X_test_rf_10)

# Оценка качества
mse_rf_10 = mean_squared_error(y_test_scaled_10, y_pred_rf_10)
mae_rf_10 = mean_absolute_error(y_test_scaled_10, y_pred_rf_10)

print(f"Random Forest: MSE = {mse_rf_10:.4f}, MAE = {mae_rf_10:.4f}")

# Построим графики
for idx in indices:
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_scaled_20[idx], label='Реальные значения', linewidth=2, color='black')

    for name, y_pred in results.items():
        plt.plot(y_pred[idx], label=name)

    plt.title(f'Прогноз на 10 лет (пример {idx})')
    plt.xlabel('Горизонт прогноза (шаг)')
    plt.ylabel('Значение (норм.)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_lagged_features(data, n_lags):
    n_samples, n_timesteps, n_features = data.shape
    X_lagged = np.zeros((n_samples, n_lags * n_features))

    for i in range(n_samples):
        for lag in range(n_lags):
            start_idx = lag * n_features
            end_idx = (lag + 1) * n_features
            X_lagged[i, start_idx:end_idx] = data[i, - (lag + 1), :]  # Берём последние `n_lags` шагов

    return X_lagged

# Пример:
n_lags = 3  # Сколько последних шагов учитывать
X_train_rf_20 = create_lagged_features(scaled_X_train_20, n_lags)
X_test_rf_20 = create_lagged_features(scaled_X_test_20, n_lags)

# Параметры для GridSearch
param_grid = {
      'n_estimators': [30, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [5, 10, 15],
}

# Поиск лучшей модели
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_rf_20, y_train_scaled_20)

# Лучшая модель
best_rf_20 = grid_search.best_estimator_
y_pred_rf_20 = best_rf_20.predict(X_test_rf_20)

# Оценка качества
mse_rf_20 = mean_squared_error(y_test_scaled_20, y_pred_rf_20)
mae_rf_20 = mean_absolute_error(y_test_scaled_20, y_pred_rf_20)

print(f"Random Forest: MSE = {mse_rf_20:.10f}, MAE = {mae_rf_20:.10f}")

print(f"Random Forest: MSE = {mse_rf:.10f}, MAE = {mae_rf:.10f}")
print(f"Random Forest 5: MSE = {mse_rf_5:.10f}, MAE = {mae_rf_5:.10f}")
print(f"Random Forest 10: MSE = {mse_rf_10:.10f}, MAE = {mae_rf_10:.10f}")
print(f"Random Forest 20: MSE = {mse_rf_20:.10f}, MAE = {mae_rf_20:.10f}")

from tabulate import tabulate

# Исходные данные моделей
models = [
    {
        "Model": "Base_LSTM",
        "units": 16,
        "dense_units": 32,
        "lr": 0.000463
    },
    {
        "Model": "Regularized_LSTM",
        "units": 96,
        "recurrent_dropout": 0.0,
        "dropout": 0.4,
        "dense_layers": 1,
        "dense_units": 16,
        "l2": 0.005287,
        "lr": 0.001
    },
    {
        "Model": "Stacked_LSTM",
        "units": 64,
        "recurrent_dropout": 0.1,
        "dropout": 0.2,
        "dense_layers": 1,
        "dense_units": 48,
        "l2": 0.0001,
        "lr": 0.000794
    },
    {
        "Model": "Bidirectional_LSTM",
        "units": 32,
        "recurrent_dropout": 0.1,
        "dropout": 0.5,
        "dense_layers": 1,
        "dense_units": 16,
        "l1": 0.00001,
        "l2": 0.000797,
        "lr": 0.001,
    }
]

# Создаем красивую таблицу
table_data = []
headers = [
    "Модель",
    "Кол-во нейронов\nв LSTM-слое",
    "Recurrent\ndropout",
    "Dropout на\nвыходах LSTM",
    "Кол-во нейронов\nв Dense слое",
    "L1-регуляризация",
    "L2-регуляризация",
    "Скорость\nобучения (lr)"
]

for model in models:
    row = [
        model["Model"],
        model.get("units", ""),
        model.get("recurrent_dropout", "-"),
        model.get("dropout", "-"),
        model.get("dense_units", "-"),
        model.get("l1", "-"),
        model.get("l2", "-"),
        model.get("lr", "-")
    ]
    table_data.append(row)

# Выводим стильную таблицу с настройками
print(tabulate(
    table_data,
    headers=headers,
    tablefmt="grid",
    numalign="center",
    stralign="center",
    floatfmt=".6f"
))

import matplotlib.pyplot as plt
import numpy as np

# Горизонт прогноза в периодах
horizons = [1.5, 2.5, 5, 10]

# Ошибки MSE для моделей
mse_data = {
    'Bidirectional_LSTM': [0.002640654, 0.003889505, 0.00660325, 0.011556428],
    'Base_LSTM':          [0.002964542, 0.009032848, 0.008671352, 0.031456055],
    'Regularized_LSTM':   [0.003725677, 0.005479769, 0.007314557, 0.020705814],
    'Stacked_LSTM':       [0.004086467, 0.004830991, 0.008041487, 0.039676043],
    'Lasso Regression':   [0.00867619,  0.01093151,  0.00733351,  0.01234252],
    #'Polynomial Regression': [2.22574551,  3.17753427,  0.01066679,  0.25810581],
    'Linear Regression':  [0.02175156,  0.02194493, 0.00744146,  0.01335177],
    'Ridge Regression':   [0.02174603,  0.02177307, 0.00763521,  0.01201772]
}

# Цвета для красоты
colors = {
    'Bidirectional_LSTM': '#1f77b4',
    'Base_LSTM':          '#aec7e8',
    'Regularized_LSTM':   '#ff7f0e',
    'Stacked_LSTM':       '#ffbb78',
    'Lasso Regression':      '#2ca02c',
    #'Polynomial Regression':            '#98df8a',
    'Linear Regression':  '#d62728',
    'Ridge Regression':   '#ff9896'
}

# Строим график
plt.figure(figsize=(12, 7))

for model_name, errors in mse_data.items():
    plt.plot(horizons, errors, marker='o', label=model_name, color=colors.get(model_name, None))

plt.title('Сравнение моделей по MSE в зависимости от горизонта прогноза', fontsize=16)
plt.xlabel('Горизонт прогноза (периоды)', fontsize=14)
plt.ylabel('Ошибка (MSE)', fontsize=14)
plt.grid(True)
plt.legend()
plt.xticks(horizons)
plt.savefig('/content/drive/MyDrive/8 семестр/Диплом/Вывод/mse_comparison.png',
                dpi=120, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Горизонт прогноза в периодах
horizons = [3, 5, 10, 20]

# Ошибки MSE для моделей
mse_data = {
    'Bidirectional_LSTM': [0.002640654, 0.003889505, 0.00660325, 0.011556428],
    'Base_LSTM':          [0.002964542, 0.009032848, 0.008671352, 0.031456055],
    'Regularized_LSTM':   [0.003725677, 0.005479769, 0.007314557, 0.020705814],
    'Stacked_LSTM':       [0.004086467, 0.004830991, 0.008041487, 0.039676043],
    'Lasso Regression':   [0.00867619,  0.01093151,  0.00733351,  0.01234252],
    #'Polynomial Regression': [2.22574551,  3.17753427,  0.01066679,  0.25810581],
    'Linear Regression':  [0.02175156,  0.02194493, 0.00744146,  0.01335177],
    'Ridge Regression':   [0.02174603,  0.02177307, 0.00763521,  0.01201772]
}

# Цвета для красоты
colors = {
    'Bidirectional_LSTM': '#1f77b4',
    'Base_LSTM':          '#1f77b4',
    'Regularized_LSTM':   '#1f77b4',
    'Stacked_LSTM':       '#1f77b4',
    'Lasso Regression':      '#2ca02c',
    #'Polynomial Regression':            '#98df8a',
    'Linear Regression':  '#2ca02c',
    'Ridge Regression':   '#2ca02c'
}

# Строим график
plt.figure(figsize=(12, 7))

for model_name, errors in mse_data.items():
    plt.plot(horizons, errors, marker='o', label=model_name, color=colors.get(model_name, None))

plt.title('Сравнение моделей по MSE в зависимости от горизонта прогноза', fontsize=16)
plt.xlabel('Горизонт прогноза (периоды)', fontsize=14)
plt.ylabel('Ошибка (MSE)', fontsize=14)
plt.grid(True)
plt.legend()
plt.xticks(horizons)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Горизонт прогноза в периодах
horizons = [1.5, 2.5, 5, 10]

# Ошибки MSE для моделей
mse_data = {
    'Двунаправленная LSTM': [6.9, 7.1, 8.1, 10.9],
    'Упрощенная базовая LSTM':          [7.1, 7.5, 9.4, 14.1],
    'Одноступенчатая LSTM с регуляризацией':   [7.3, 7.5, 8.7, 12.5],
    'Двуступенчатая глубокая LSTM':       [7.4, 7.6, 9.2, 13],
    'Лассо-регрессия':   [8, 10.5, 15.1, 21.9],
    'Линейная регрессия':  [8.7, 12.3, 16.1, 26.2],
    'Гребневая регрессия':   [8.3, 11.2, 15.3, 23.4]
}

# Цвета для красоты
colors = {
    'Двунаправленная LSTM': '#d62728',
    'Упрощенная базовая LSTM':          '#1f77b4',
    'Одноступенчатая LSTM с регуляризацией':   '#ff7f0e',
    'Двуступенчатая глубокая LSTM':       '#2ca02c',
    'Лассо-регрессия':      '#ff9896',
    #'Polynomial Regression':            '#98df8a',
    'Линейная регрессия':  '#ffbb78',
    'Гребневая регрессия':   '#98df8a'
}

# Строим график
plt.figure(figsize=(12, 7))

for model_name, errors in mse_data.items():
    plt.plot(horizons, errors, marker='o', label=model_name, color=colors.get(model_name, None))

plt.title('Сравнение моделей по MAPE в зависимости от горизонта прогноза', fontsize=16)
plt.xlabel('Горизонт прогноза (периоды)', fontsize=14)
plt.ylabel('Ошибка (MAPE), %', fontsize=14)
plt.grid(True)
plt.legend()
plt.xticks(horizons)
plt.savefig('/content/drive/MyDrive/8 семестр/Диплом/Вывод/mse_comparison_new.png',
                dpi=120, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt

# Входные данные
mse_data = {
    'Двунаправленная LSTM': [0.002640654, 0.003889505, 0.00660325, 0.011556428],
    'Упрощенная базовая LSTM': [0.002964542, 0.009032848, 0.008671352, 0.031456055],
    'Одноступенчатая LSTM с регуляризацией': [0.003725677, 0.005479769, 0.007314557, 0.020705814],
    'Двуступенчатая глубокая LSTM': [0.004086467, 0.004830991, 0.008041487, 0.039676043],
    'Лассо-регрессия': [0.00867619, 0.01093151, 0.00733351, 0.01234252],
    'Линейная регрессия': [0.02175156, 0.02194493, 0.00744146, 0.01335177],
    'Гребневая регрессия': [0.02174603, 0.02177307, 0.00763521, 0.01201772]
}

# Горизонты прогноза и соответствующее количество точек
forecast_horizons = [1.5, 2.5, 5, 10]
forecast_steps = [3, 5, 10, 20]

# Рассчитываем MSE на шаг прогноза
mse_per_step_data = {
    model: [mse / steps for mse, steps in zip(mse_list, forecast_steps)]
    for model, mse_list in mse_data.items()
}

colors = {
    'Двунаправленная LSTM': '#d62728',
    'Упрощенная базовая LSTM':          '#1f77b4',
    'Одноступенчатая LSTM с регуляризацией':   '#ff7f0e',
    'Двуступенчатая глубокая LSTM':       '#2ca02c',
    'Лассо-регрессия':      '#ff9896',
    #'Polynomial Regression':            '#98df8a',
    'Линейная регрессия':  '#ffbb78',
    'Гребневая регрессия':   '#98df8a'
}

# Строим график
plt.figure(figsize=(12, 7))
for model, mse_list in mse_per_step_data.items():
    plt.plot(forecast_horizons, mse_list, marker='o', label=model, color=colors.get(model, None))

plt.xlabel("Горизонт прогноза (лет)")
plt.ylabel("Средняя ошибка на шаг прогноза (MSE / точка)")
plt.title("Сравнение моделей по средней ошибке на шаг прогноза")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def get_user_data():
    indicators = []
    for i in range(8):
        while True:
            try:
                values_input = input(f"Введите 5 значений для показателя {i + 1} через пробел: ")
                values = list(map(float, values_input.split()))
                if len(values) != 5:
                    print("Ошибка: нужно ввести ровно 5 значений!")
                    continue
                indicators.append(values)
                break
            except ValueError:
                print("Ошибка: вводите только числа, разделенные пробелами!")
    return np.array(indicators)



def process_with_neural_network(data):
    # Здесь должна быть ваша нейросеть
    # В этом примере просто выводим данные
    print("\nДанные, переданные в нейросеть:")
    print(data)
    # Пример: return neural_network.predict(data)

if __name__ == "__main__":
    print("Введите значения следующих показателей для прогноза значения индекса качества жизни:\n1. Покупательная способность\n2. Безопасность\n3. Здравоохранение\n4. Стоимость жизи\n5. Доступность жилья\n6. Время в пути\n7. Загрязнение окружающей среды\n8. Климат")
    user_data = get_user_data()
    process_with_neural_network(user_data)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

def get_user_data():
    """Запрашивает у пользователя 5 значений для 8 показателей."""
    indicators = []
    print("Введите 5 значений для каждого из 8 показателей (разделяйте пробелом):")

    features = [
        "Purchasing Power Index",
        "Safety Index",
        "Health Care Index",
        "Cost of Living Index",
        "Property Price to Income Ratio",
        "Traffic Commute Time Index",
        "Pollution Index",
        "Climate Index"
    ]

    for i, feature in enumerate(features, 1):
        while True:
            try:
                values_input = input(f"{i}. {feature}: ")
                values = list(map(float, values_input.split()))
                if len(values) != 5:
                    print("Ошибка: нужно ввести ровно 5 значений!")
                    continue
                indicators.append(values)
                break
            except ValueError:
                print("Ошибка: вводите только числа, разделенные пробелами!")

    return np.array(indicators).T

def preprocess_user_data(data):
    """Предобработка введенных данных аналогично вашему датасету."""
    df = pd.DataFrame(data, columns=[
        "Purchasing Power Index",
        "Safety Index",
        "Health Care Index",
        "Cost of Living Index",
        "Property Price to Income Ratio",
        "Traffic Commute Time Index",
        "Pollution Index",
        "Climate Index"
    ])

    df['Economic_Stress'] = df['Cost of Living Index'] / df['Purchasing Power Index']

    pca = PCA(n_components=1)
    df['Safety_Pollution_PC'] = pca.fit_transform(df[['Safety Index', 'Pollution Index']])

    df_final = df.drop(columns=[
        'Cost of Living Index', 'Purchasing Power Index',
        'Safety Index', 'Pollution Index',
        'Traffic Commute Time Index', 'Health Care Index'
    ])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_final)

    return scaled_data.reshape(1, 5, -1)

if __name__ == "__main__":
    user_data = get_user_data()

    processed_data = preprocess_user_data(user_data)

    custom_objects = {
        'mse': MeanSquaredError(name='mse')
    }

    loaded_model = load_model(
        '/content/drive/MyDrive/8 семестр/Диплом/models/bidirectional_lstm.h5',
        custom_objects=custom_objects
    )

    y_pred_final = loaded_model.predict(processed_data)
    y_pred_original = scaler_y.inverse_transform(y_pred_final)
    print(y_pred_original)
    plt.plot(y_pred_original[0], marker='o')
    plt.title('Прогноз на 10 лет')
    plt.xlabel('Горизонт прогноза (шаг)')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print(y_pred_final[0])
print(y_pred_original[0])

plt.plot(y_pred_final[0], marker='o')
plt.title('Прогноз на 10 лет')
plt.xlabel('Горизонт прогноза (шаг)')
plt.ylabel('Значение')
plt.grid(True)
plt.tight_layout()
plt.show()

y_pred_final = loaded_model.predict(processed_data)
y_pred_original = scaler_y.inverse_transform(y_pred_final)
print(y_pred_original)
plt.plot(y_pred_original[0], marker='o')

plt.title('Прогноз на 10 лет')
plt.xlabel('Горизонт прогноза (шаг)')
plt.ylabel('Значение')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 1. Настройки моделей ---
MODELS = {
    "Base_LSTM": build_model_0,
    "Regularized_LSTM": build_model,
    "Stacked_LSTM": build_model_2,
    "Bidirectional_LSTM": build_model_4,
}

all_predictions = {}  # сохраняем все предсказания

# --- 2. Функция для обучения и сбора результатов ---
def run_hyperparameter_tuning(models, scaled_X_train, y_train_scaled, scaled_X_test, y_test_scaled):
    results = {}
    all_histories = {}
    trained_models = {}  # Новый словарь для сохранения обученных моделей

    for model_name, hypermodel in models.items():
        print(f"\n🔧 Tuning {model_name}")

        tuner = BayesianOptimization(
            hypermodel,
            objective='val_loss',
            max_trials=15,
            num_initial_points=3,
            directory='tuning',
            project_name=model_name
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        tuner.search(
            scaled_X_train,
            y_train_scaled,
            epochs=100,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)

        history = best_model.fit(
            scaled_X_train,
            y_train_scaled,
            epochs=100,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        # Сохраняем модель в словарь
        trained_models[model_name] = best_model

        # Остальной код остается без изменений...
        eval_result = best_model.evaluate(scaled_X_test, y_test_scaled)
        y_pred = best_model.predict(scaled_X_test)
        all_predictions[model_name] = y_pred

        results[model_name] = {
            'best': {
                'mse': mean_squared_error(y_test_scaled, y_pred),
                'mae': mean_absolute_error(y_test_scaled, y_pred),
                'params': best_hps.values,
                'epochs': len(history.history['val_loss']),
            }
        }
        all_histories[model_name] = history.history['val_loss']

    return results, trained_models  # Теперь возвращаем и результаты, и модели

# --- 3. Функция для визуализации одной модели ---
def plot_model_history(model_name, history):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(history.history['val_loss']) + 1)

    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.title(f'Изменение значения функции потерь по эпохам для {model_name}', pad=20)
    plt.xlabel('Эпоха', labelpad=10)
    plt.ylabel('Значение функции потерь (MSE)', labelpad=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'/content/drive/MyDrive/8 семестр/Диплом/Вывод/предсказание на 20/Апгрейд/{model_name}_history.png', dpi=120)
    plt.close()

# --- 4. Функция для общего сравнения моделей ---
def plot_combined_val_loss(histories):
    plt.figure(figsize=(12, 8))

    max_epochs = max(len(h) for h in histories.values())

    for model_name, val_loss in histories.items():
        epochs = range(1, len(val_loss) + 1)
        plt.plot(epochs, val_loss, label=model_name, linewidth=2)

    plt.title('Сравнение динамики валидационной ошибки (Loss) для разных моделей', pad=20)
    plt.xlabel('Эпоха', labelpad=10)
    plt.ylabel('Значение валидационной функции потерь (MSE)', labelpad=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(1, max_epochs)

    plt.axhline(y=min(min(h) for h in histories.values()),
                color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/8 семестр/Диплом/Вывод/предсказание на 20/Апгрейд/models_val_loss_comparison.png',
                dpi=120, bbox_inches='tight')
    plt.close()

# --- 5. Сохраняем результаты в Excel ---
def save_results_to_excel(results):
    rows = []

    for model, data in results.items():
        rows.append({
            'Model': model,
            'MSE': data['best']['mse'],
            'MAE': data['best']['mae'],
            'Epochs': data['best']['epochs'],
            'Params': json.dumps(data['best']['params'])
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by='MSE')

    output_path = '/content/drive/MyDrive/8 семестр/Диплом/Вывод/предсказание на 20/Апгрейд/final_results.xlsx'
    df.to_excel(output_path, index=False)

    print(f"\n✅ Excel файл сохранён по адресу: {output_path}")
    print("\n📊 TOP-3 моделей по MSE:")
    print(df.head(3))

# Запускаем обучение и получаем модели
final_results, trained_models = run_hyperparameter_tuning(
    MODELS,
    scaled_X_train_20,
    y_train_scaled_20,
    scaled_X_test_20,
    y_test_scaled_20
)

# Достаем Bidirectional LSTM
bidirectional_lstm_model = trained_models["Bidirectional_LSTM"]

trained_models

bidirectional_lstm_model.save('/content/drive/MyDrive/8 семестр/Диплом/models/bidirectional_lstm.h5')

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

# Указываем кастомные объекты (если они есть)
custom_objects = {
    'mse': MeanSquaredError(name='mse'),  # Явно регистрируем MSE
    # Добавьте другие кастомные метрики/слои здесь
}

loaded_model = load_model(
    '/content/drive/MyDrive/8 семестр/Диплом/models/bidirectional_lstm.h5',
    custom_objects=custom_objects
)

sample_input = scaled_X_test_20[0:1]  # Берем первый пример из тестовых данных
prediction = loaded_model.predict(sample_input)
print(y_pred_final)

data[data['Country']=='Luxembourg']

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

def get_user_data():
    """Запрашивает у пользователя 5 значений для 8 показателей."""
    indicators = []
    print("Введите 5 значений для каждого из 8 показателей (разделяйте пробелом):")

    features = [
        "Purchasing Power Index",
        "Safety Index",
        "Health Care Index",
        "Cost of Living Index",
        "Property Price to Income Ratio",
        "Traffic Commute Time Index",
        "Pollution Index",
        "Climate Index"
    ]

    for i, feature in enumerate(features, 1):
        while True:
            try:
                values_input = input(f"{i}. {feature}: ")
                values = list(map(float, values_input.split()))
                if len(values) != 5:
                    print("Ошибка: нужно ввести ровно 5 значений!")
                    continue
                indicators.append(values)
                break
            except ValueError:
                print("Ошибка: вводите только числа, разделенные пробелами!")

    return np.array(indicators).T

def preprocess_user_data(data):
    """Предобработка введенных данных аналогично вашему датасету."""
    df = pd.DataFrame(data, columns=[
        "Purchasing Power Index",
        "Safety Index",
        "Health Care Index",
        "Cost of Living Index",
        "Property Price to Income Ratio",
        "Traffic Commute Time Index",
        "Pollution Index",
        "Climate Index"
    ])

    df['Economic_Stress'] = df['Cost of Living Index'] / df['Purchasing Power Index']

    #pca = PCA(n_components=1)
    df['Safety_Pollution_PC'] = pca.transform(df[['Safety Index', 'Pollution Index']])

    df_final = df.drop(columns=[
        'Cost of Living Index', 'Purchasing Power Index',
        'Safety Index', 'Pollution Index',
        'Traffic Commute Time Index', 'Health Care Index'
    ])

    #scaler = StandardScaler()
    scaled_data = scaler_x.transform(df_final)

    return scaled_data.reshape(1, 5, -1)

if __name__ == "__main__":
    user_data = get_user_data()

    processed_data = preprocess_user_data(user_data)

    custom_objects = {
        'mse': MeanSquaredError(name='mse')
    }

    loaded_model = load_model(
        '/content/drive/MyDrive/8 семестр/Диплом/models/bidirectional_lstm.h5',
        custom_objects=custom_objects
    )

    y_pred_final = loaded_model.predict(processed_data)
    y_pred_original = scaler_y.inverse_transform(y_pred_final)
    print(y_pred_original)
    plt.plot(y_pred_original[0], marker='o')
    plt.title('Прогноз на 10 лет')
    plt.xlabel('Горизонт прогноза (шаг)')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

custom_objects = {
    'mse': MeanSquaredError(name='mse')
}

loaded_model = load_model(
   '/content/drive/MyDrive/8 семестр/Диплом/models/bidirectional_lstm.h5',
    custom_objects=custom_objects
)

# Найдём слой Bidirectional
bidir_layer = loaded_model.layers[0]

# Получим forward-слой из двунаправленного слоя
forward_lstm = bidir_layer.forward_layer

# Извлекаем веса из forward-части
forward_weights = forward_lstm.get_weights()

# Укажи нужные параметры вручную или по аналогии с hp
units = forward_lstm.units
#input_shape = forward_lstm.input_shape[1:]  # (timesteps, features)

# Создаём обычную forward-модель
forward_model = Sequential()
forward_model.add(layers.LSTM(units,
                       recurrent_dropout=forward_lstm.recurrent_dropout,
                       input_shape=(scaled_X_train.shape[1], scaled_X_train.shape[2])))
forward_model.add(Dropout(0.5))

# Добавь Dense-слои как в исходной модели
forward_model.add(Dense(16, kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=0.000797)))
forward_model.add(Dense(20, activation='linear'))  # выходной слой

# Компиляция
forward_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Устанавливаем веса только в первом LSTM-слое
forward_model.layers[0].set_weights(forward_weights)
forward_model.summary()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras import initializers

# Загружаем обученную BiLSTM-модель
loaded_model = keras.models.load_model(
    '/content/drive/MyDrive/8 семестр/Диплом/models/bidirectional_lstm.h5',
    custom_objects=custom_objects
)

# Forward LSTM
forward_lstm = loaded_model.layers[0].forward_layer

# Вход
input_layer = Input(shape=(scaled_X_train.shape[1], scaled_X_train.shape[2]))
forward_output = forward_lstm(input_layer)

# Dropout — можем использовать как есть
x = Dropout(0.5)(forward_output)

# Создаём Dense заново: вход теперь (None, 32) вместо (None, 64)
# Получим старый Dense-слой
old_dense = loaded_model.layers[2]

# Новый Dense с теми же параметрами, но input_dim=32
new_dense = Dense(
    old_dense.units,
    activation=old_dense.activation,
    kernel_regularizer=old_dense.kernel_regularizer,
    bias_regularizer=old_dense.bias_regularizer,
    kernel_initializer=initializers.Zeros(),  # потом перезапишем весами
    bias_initializer=initializers.Zeros()
)

x = new_dense(x)

# Выходной Dense (его можно взять как есть — он подключается к (None, 16))
output_dense = loaded_model.layers[3]
output = output_dense(x)

# Собираем forward-модель
forward_model = Model(inputs=input_layer, outputs=output)

# Устанавливаем веса: используем только первую половину весов (т.е. forward)
dense_weights = old_dense.get_weights()
# веса: [W, b] где W.shape = (64, 16), b.shape = (16,)
new_dense.set_weights([
    dense_weights[0][:32, :],  # только первая половина весов (forward)
    dense_weights[1]           # bias можно оставить без изменений
])

forward_model.summary()

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

def get_user_data():
    """Запрашивает у пользователя 5 значений для 8 показателей."""
    indicators = []
    print("Введите 5 значений для каждого из 8 показателей (разделяйте пробелом):")

    features = [
        "Purchasing Power Index",
        "Safety Index",
        "Health Care Index",
        "Cost of Living Index",
        "Property Price to Income Ratio",
        "Traffic Commute Time Index",
        "Pollution Index",
        "Climate Index"
    ]

    for i, feature in enumerate(features, 1):
        while True:
            try:
                values_input = input(f"{i}. {feature}: ")
                values = list(map(float, values_input.split()))
                if len(values) != 5:
                    print("Ошибка: нужно ввести ровно 5 значений!")
                    continue
                indicators.append(values)
                break
            except ValueError:
                print("Ошибка: вводите только числа, разделенные пробелами!")

    return np.array(indicators).T

def preprocess_user_data(data):
    """Предобработка введенных данных аналогично вашему датасету."""
    df = pd.DataFrame(data, columns=[
        "Purchasing Power Index",
        "Safety Index",
        "Health Care Index",
        "Cost of Living Index",
        "Property Price to Income Ratio",
        "Traffic Commute Time Index",
        "Pollution Index",
        "Climate Index"
    ])

    df['Economic_Stress'] = df['Cost of Living Index'] / df['Purchasing Power Index']

    #pca = PCA(n_components=1)
    df['Safety_Pollution_PC'] = pca.transform(df[['Safety Index', 'Pollution Index']])

    df_final = df.drop(columns=[
        'Cost of Living Index', 'Purchasing Power Index',
        'Safety Index', 'Pollution Index',
        'Traffic Commute Time Index', 'Health Care Index'
    ])

    #scaler = StandardScaler()
    scaled_data = scaler_x.transform(df_final)

    return scaled_data.reshape(1, 5, -1)

if __name__ == "__main__":
    user_data = get_user_data()

    processed_data = preprocess_user_data(user_data)

    custom_objects = {
        'mse': MeanSquaredError(name='mse')
    }

    loaded_model = forward_model

    y_pred_final = loaded_model.predict(processed_data)
    y_pred_original = scaler_y.inverse_transform(y_pred_final)
    print(y_pred_original)
    plt.plot(y_pred_original[0], marker='o')
    plt.title('Прогноз на 10 лет')
    plt.xlabel('Горизонт прогноза (шаг)')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def build_model_4(hp):
    model = keras.Sequential()

    # Двунаправленная LSTM
    model.add(layers.Bidirectional(
        layers.LSTM(
            hp.Int('units', 32, 128, step=32),
            recurrent_dropout=hp.Float('rec_drop', 0.1, 0.3)),
        input_shape=(scaled_X_train.shape[1], scaled_X_train.shape[2])))

    # Регуляризация
    model.add(layers.Dropout(hp.Float('dropout', 0.2, 0.5)))

    # Dense-слои с L1/L2 регуляризацией
    for i in range(hp.Int('num_dense', 1, 2)):
        model.add(layers.Dense(
            hp.Int(f'dense_{i}_units', 16, 64, step=16),
            kernel_regularizer=keras.regularizers.l1_l2(
                l1=hp.Float(f'l1_{i}', 1e-5, 1e-3),
                l2=hp.Float(f'l2_{i}', 1e-5, 1e-3))))

    # Выходной слой
    model.add(layers.Dense(20, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.AdamW(
            hp.Float('lr', 1e-5, 1e-3),
            weight_decay=hp.Float('wd', 1e-6, 1e-4)),
        loss='mse',
        metrics=['mae'])
    return model