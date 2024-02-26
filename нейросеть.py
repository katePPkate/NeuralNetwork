import pandas as pd
import numpy as np

Initial_Data = pd.read_csv('Initial_Data.txt',sep=';')
'''Структура файла, который мы прочитали:
<TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>;<OPENINT>'''

Initial_Data = Initial_Data.rename(columns=
        {'<DATE>':'date', '<CLOSE>':'close', '<OPEN>':'open',
                '<HIGH>':'high', '<LOW>':'low', '<VOL>':'vol'})
Initial_Data.date = pd.to_datetime(Initial_Data.date, format='%d/%m/%y')
Initial_Data.drop(columns= ["<TICKER>","<PER>",'<TIME>', '<OPENINT>'])

'''366 строк с информацией. Пусть наша нейросеть будет делать вывод о цене в строке N+10 по строкам с N по N+9'''

data = Initial_Data['close']
print (type(data))




import tensorflow as tf

# Генерация случайных данных для примера
prices = Initial_Data['close']  # Предположим, это динамика цен за 300 дней

# Подготовка данных для обучения
X = []
y = []
for i in range(len(prices) - 1):
    X.append(prices[i:i+365])  # Используем цены за предыдущие 300 дней
    y.append(prices[i+1])      # Предсказываем цену на следующий день

X = np.array(X)
y = np.array(y)

# Создание нейронной сети
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(365,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X, y, epochs=10, batch_size=32)

# Прогнозирование цены на 301 день
last_365_days = prices[-365:]  # Последние 300 дней
prediction = model.predict(np.array([last_365_days]))[0][0]

print("Прогноз цены на 366 день:", prediction)

