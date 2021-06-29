def data_information_printer(data):
    """Функция принимает на вход датасет и выводит основную информацию о нём"""
    display(data.head(10))
    print('-------- Информация о признаках --------')
    display(data.info())
    print('-------- Описание числовых признаков --------')
    display(data.describe())
    print('-------- Информация о пропусках --------')
    display(data.isna().sum())
    print('-------- Информация о дубликатах --------')
    display(data.duplicated().sum())


def make_features(data, max_lag, rolling_mean_size):
    """Функция принимает на вход датасет, количество шагов для отстающих значений и количество шагов 
    для сглаживания скользящим средним. Далее добавляет признаки дня, дня недели и месяца, 
    создаёт заданное количество отстающих значений и признак скользящего среднего"""
    
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)

    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()

def lemmatize(text):
    """Функция принимает текст и возвращает его в лемматизированном виде"""
    m = Mystem()
    lemm_list = m.lemmatize(text)
    lemm_text = "".join(lemm_list)
    return lemm_text

def clear_text(text):
    """Функция приниает текст и удаляет из него все символы, кроме кириллицы"""
    new_text = re.sub(r'[^а-яА-ЯёЁ ]', ' ', text)
    cleared_text = ' '.join(new_text.split())
    return cleared_text


def clear_text_eng(text):
    """Функция приниает текст и удаляет из него все символы, кроме английских букв"""
    new_text = re.sub(r'[^a-zA-Z]', ' ', text)
    cleared_text = ' '.join(lclear_text.split())
    return cleared_text


def upsampling(data, x_train, y_train, coefficient, target_column):
    """Функция работает для задачи классификации.
    Принимает общий датасет, x_train и y_train, коэффициент компенсации дисбаланса 
    и название столбца целевого признака.
    Возвращает x_train_upsampled и y_train_upsampled"""  
    # Определяем индексы строк, соответствующие разным классам целевого признака
    ex_0_index = y_train[y_train == 0].index
    ex_1_index = y_train[y_train != 0].index
    # Определяем степень дисбаланса
    # Присваиваем отдельные датафреймы с классом 0 и 1
    ex_0 = data.loc[ex_0_index]
    ex_1 = data.loc[ex_1_index]
    # Далее определяем, на какую степень будем увеличивать обучающую выборку. 
    difference = int(coefficient * (ex_0.shape[0] - ex_1.shape[0]))
    print(f'Увеличили на {difference} строк')
    up_data = ex_1.sample(difference, replace=True).drop([target_column], axis=1)
    up_y_data = y_train.loc[up_data.index]
    # Объединяем нарощенные данные с обучающей выборкой
    import numpy as np
    x_train_upsampled = pd.concat([x_train, up_data])
    y_train_upsampled = np.hstack((y_train.values, up_y_data.values))
    return x_train_upsampled, y_train_upsampled


def downsampling(data, y_train, target_column):
    """Функция работает для задачи классификации.
    Принимает общий датасет, y_train
    и название столбца целевого признака.
    Возвращает x_train_downsampled и y_train_downsampled"""
    # Определяем индексы строк, соответствующие разным классам целевого признака
    ex_0_index = y_train[y_train == 0].index
    ex_1_index = y_train[y_train != 0].index    
    # Присваиваем отдельные датафреймы с классом 0 и 1
    ex_0 = data.loc[ex_0_index]
    ex_1 = data.loc[ex_1_index]
    down_data = ex_0.sample(ex_1.shape[0]).drop([target_column], axis=1)
    down_y_data = y_train.loc[down_data.index]
    x_train_downsampled = pd.concat([down_data, ex_1]).drop([target_column], axis=1)
    y_train_downsampled = np.hstack((down_y_data.values, y_train[y_train != 0].values))  
    return x_train_downsampled, y_train_downsampled

def model_text_learning(model, x_train, y_train):
    """Функция принимает модель с заданными гиперпараметрами, x_train и y_train.
    Обрабатывает x_train в corpus, создаёт массив TFIDF.
    Возвращает среднее значение f1-меры по 5 батчам кросс-валидации."""    
    corpus = x_train.values.astype('U')
    count_tf_idf = TfidfVectorizer(stop_words = stopwords)
    tf_idf = count_tf_idf.fit_transform(corpus)
    score = cross_val_score(model, tf_idf, y_train, cv = 5, scoring = 'f1').mean()  
    return score

def arithmetical_rounding(x,y=0):
    '''Принимает число и количество требуемых знаков после запятой после округления.
    Округляет по обычным арифметическим правилам. Возвращает округлённое число'''
    m = int('1'+'0'*y) # multiplier - how many positions to the right
    q = x*m # shift to the right by multiplier
    c = int(q) # new number
    i = int( (q-c)*10 ) # indicator number on the right
    if i >= 5:
        c += 1
    return c/m

def unixtime_converter(row):
    '''Функция принимает строку датасета и приводит дату в формат timestamp unixtime'''
    dt = row['date']
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
    return timestamp

def high_corr_features(data, coefficient):
    """Функция принимает датасет и уровень кореляции и выводит на печать признаки, кореляция между которыми
    выше заданного коэффициента"""
    for column in data.columns:
        for column_2 in data.columns:
            if abs(data[column].corr(data[column_2])) > coefficient and column != column_2:
                print(column, column_2, data[column].corr(data[column_2]))

def upsample(features, target, repeat):
    """Простой апсемплер - принимает трэйн, таргет и сколько раз повторить"""
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled
            
def upsample_balanced(features, target, strings_to_add):
    """Простой апсемплер - принимает трэйн, таргет и сколько строчек единичного класса должно стать в итоге"""
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones.sample(n=strings_to_add, replace=True)])
    target_upsampled = pd.concat([target_zeros] + [target_ones.sample(n=strings_to_add, replace=True)])
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled

def probability(p, k, n):
    """Функция принимает вероятность события р, требуемое количество успешных
    опытов k и количество повторений эксперимента n. Печатает число комбинаций
    и искомую вероятность. Возвращает искомую вероятность."""
    # Считаем число комбинаций по формуле с факториалами
    C = (math.factorial(n)) / (math.factorial(k) * math.factorial(n - k))
    print(f'Combinations number = {C}')
    # Считаем искомую вероятность по формуле Бернулли
    probability = (C * (p**k)) * ((1 - p) ** (n - k))
    print(f'Probability         = {probability}')
    return probability

def high_corr_features(data, coefficient):
    """Функция принимает датасет и уровень кореляции и выводит на печать признаки, кореляция между которыми
    выше заданного коэффициента"""
    high_corr_columns = []
    for column in data.columns:
        for column_2 in data.columns:
            if abs(data[column].corr(data[column_2])) > coefficient and column != column_2:
                print(column, column_2, data[column].corr(data[column_2]))
                high_corr_columns.append(column)
    return high_corr_columns



