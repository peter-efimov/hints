# Убираем из строчек лишние символы и переводим в вещественный формат
data['Velocity'] = [float(x.replace('км/ч','')) for x in data['Velocity']]

# Петя, нашел как в юпитере можно конвертировать ipynb в py.
# Может пригодится)
!jupyter nbconvert name.ipynb —to python

# Получить первое или последнее значение в группе в датасете пандас
start_time = df_temp.sort_values('Время замера').groupby('key').first()['Время замера']
end_time = df_temp.sort_values('Время замера').groupby('key').last()['Время замера']

# получить в describe статистику и по категориальным признакам тоже
data.describe(include=['O'])

# Выключить все предупреждения
import warnings
warnings.filterwarnings("ignore")

# Извлечь титулы из имён регулярным выражением. Combine - датасет из Титаника
for row in combine:
    row['Title'] = row['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Только что сделал открытие: если у кого-то получаются мыльные рисунки в матплотлибе, добавьте в начале проекта строчку:
%config InlineBackend.figure_format='retina'

# Вот так можно показывать время срабатывания в каком-нибудь длинном цикле
from datetime import datetime
print('start: ', datetime.now())
...

# Magic command is added before string of code to estimate it's runtime
# -r2 - number of runs
# -n10 - number of loops
%timeit -r2 -n10 rand_nums = np.random.rand(1000)
% - one string, %% - whole cell mode

#magic command to estimate whole time
pip install line_profiler
%load_ext line_profiler
#then we can use
%lprun -f function function(parameters) # to estimate function we type its name, then its name including parameters

# memory profiler
pip install memory_profiler
%load_ext memory_profiler
%mp_run -f function function(parameters) # to estimate function we type its name, then its name including parameters
# but it works only with function in a separate file

# подсчёт элементов в списке и количество, сколько раз они встречаются
from collections import Counter
result = Counter(list_to_count)

# поменять столбцы местами в pandas можно командой
df = df.reindex(list_of_new_cols)

# Add a decorator that will make timer() a context manager
@contextlib.contextmanager
def timer():
  """Time the execution of a context block.
  Yields:
    None
  """
  start = time.time()
  # Send control back to the context block
  yield
  end = time.time()
  print('Elapsed: {:.2f}s'.format(end - start))
with timer():
  print('This should take approximately 0.25 seconds')
  time.sleep(0.25)

# make the pie circular by setting the aspect ratio to 1
plt.figure(figsize=plt.figaspect(1))
values = [3, 12, 5, 8] 
labels = ['a', 'b', 'c', 'd'] 
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct
plt.pie(values, labels=labels, autopct=make_autopct(values))
plt.show()

# Посчитат время исполнения операции
start_time = time.time()
# some operation
duration = time.time() - start_time

# показывать все столбы и строки в pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Добавить два столбца с помощью функции через apply в pandas
def myfunc1(row):
    C = row['A'] + 10
    D = row['A'] + 50
    return pd.Series([C, D])
df[['C', 'D']] = df.apply(myfunc1, axis=1)

# Чтобы расширить ноутбук на всю ширину экрана
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# Округление вверх до целого в пандасе и перевод в формат int64
df_ship['demand'] = (df_ship['loc_speed'] * region_safe_term).apply(np.ceil).astype('int64')

# Вытащить номер недели из даты в pandas
df['start_d'].dt.isocalendar().week