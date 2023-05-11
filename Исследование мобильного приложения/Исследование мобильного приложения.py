#!/usr/bin/env python
# coding: utf-8

# # Исследование мобильного приложения

# В нашем распоряжении логи мобильного приложения стартапа, который продает продукты питания.Необходимо:
# * Разобраться, как ведут себя пользователи мобильного приложения. Изучить воронку продаж. Узнать, как пользователи доходят до покупки. 
# * Дизайнеры захотели поменять шрифты во всём приложении, для принятия решения необходимо исследовать результаты A/A/B-эксперимента.

# **Описание данных**

# Путь к файлу:
# **/datasets/logs_exp.csv**

# Каждая запись в логе — это действие пользователя, или событие.
# * *EventName* - название события;
# * *DeviceIDHash* -  уникальный идентификатор пользователя;
# * *EventTimestamp* - время события;
# * *ExpId* - номер эксперимента: 246 и 247 — контрольные группы, а 248 — экспериментальная.

# ##  Изучение данных из файла

# In[1]:


# импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats as st
import seaborn as sns
from datetime import datetime, timedelta
from IPython.display import display_html
pd.set_option('display.max_columns', 30)
pd.set_option("display.float_format", "{:.2f}".format)
import warnings
warnings.simplefilter('ignore')
import os
import plotly
import plotly.graph_objs as go
import math as mth


# In[2]:


df = pd.read_csv('/datasets/logs_exp.csv',sep='\t')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


#проверим дубликаты
df.duplicated().sum()


# In[6]:


#проверим уникальные названия
df.ExpId.unique()


# 246 и 247 — контрольные группы, а 248 — экспериментальная.

# In[7]:


#проверим уникальные названия
df.EventName.unique()


# Для дальнейшей работы с данными необходимо:
#  * Переименовать названия столбцов для удобства при дальнейшей работе
#  * Столбец EventTimestamp имеет данные в секундах. Поэтому преобразуем столбец в формат data. Добавить отдельный столбец с датой
#  * Удалить дубликаты, они составляют 0,17% от всего количества событий. 
#  

# ## Подготовка данных

# In[8]:


#переименование столбцов
df = df.rename(columns={'EventName':'event_name', 'DeviceIDHash':'device_id', 'EventTimestamp':'event_time', 'ExpId':'group'})


# In[9]:


df.head(5)


# In[10]:


# преобразование времени
df['event_time'] = pd.to_datetime(df.event_time, unit='s')


# In[11]:


df['event_time'].head(5)


# In[12]:


# создание столбца с датой
df['event_date'] = df.event_time.dt.date


# In[13]:


df.head(2)


# In[14]:


# удаление дубликатов 
df = df.drop_duplicates().reset_index(drop=True)


# In[15]:


df.duplicated().sum()


# In[16]:


#проверка
df.info(5)


# ## Анализ данных

# Узнаем сколько всего событий и пользователей в логе. Посчитаем сколько в среднем событий приходится на одного пользователя. 

# In[17]:


display(f'Всего в логе событий {len(df)}.')


# In[18]:


display(f'Всего пользователей в логе {len(df.device_id.unique())}.')


# In[19]:


display(f'В среднем на пользователя приходится {int(len(df) / len(df.device_id.unique()))} события.')


# In[20]:


plt.figure(figsize=(15, 7))
sns.histplot(data=df.groupby('device_id')[['event_name']].count(), x='event_name', kde=True)
plt.title('Распределение событий по пользователям')
plt.xlabel('Количество событий')
plt.ylabel('Количество пользователей')
plt.xlim(0,201)
plt.show()


# Проверим, данными за какой преиод мы располагаем и как они распределены.

# In[21]:


#минимальная и максимальная дата в логе
display(f'Минимальная дата:', df['event_time'].max())
display(f'Максимальная дата:', df['event_time'].max());


# Мы имеем данные за период с 25 июля по 7 августа 2019г. Посмотрим как распределена активность событий во времени.

# In[22]:


#гистограммa по дате и времени
series = df.groupby('event_date')[['event_name']].count().reset_index()
plt.figure(figsize=(15, 7))
ax = sns.barplot(x='event_date', y='event_name', data=series, palette='crest')
for i, v in series.iterrows():
       ax.text(i, v[1]+500, str(v[1]), ha='center')
plt.title('Распределение событий по времени')
plt.xlabel('Дата')
plt.ylabel('Количество событий')
plt.show()


# In[23]:


# код ревьюера
df['event_time'].hist(bins=14*24, figsize=(14, 5));


# Резкое увеличение событий зафиксировано 1-го августа с 2030 до 36141. Такое может произойти по техническим причинам(например данные за июль выгрузились в первое число августа). Устаревшими данными примем период с 25 по 31 июля. Потому как за этот период маленькое количество событий, которое не даст полной картины происходящего в приложении. 

# In[24]:


#данные исходного лога
df.info()


# Всего в логе 243713 событий. 

# In[25]:


#выбран актуальный период
df_2w = df.query('event_date >= datetime(2019, 8, 1).date()')
df_2w.info()


# Осталось 240887 событий. Посчитаем это в процентах.

# In[26]:


#оставшаяся доля событий
display(f'{(len(df_2w) / len(df)):.2%} - доля событий c 01/08/19')


# Мы выбрали актуальный период и у нас осталось 98.84% событий. 

# In[27]:


#количество пользователей в исходном логе
dev_count_1=(len(df.device_id.unique()))
dev_count_1


# In[28]:


#количество пользователей в актуальном логе
dev_count_2=(len(df_2w.device_id.unique()))
dev_count_2


# In[29]:


#процент оставшихся пользователей
display(f'{(dev_count_2 / dev_count_1):.2%} - доля пользователей c 01/08/19')


# Почти все пользователи остались. Проверим распределение пользователей по группам.

# In[30]:


#количество пользователей по группам 246, 277 и 248
display(df_2w.groupby('group')['device_id'].nunique())


# Проверим не попали ли пользователи в одну из двух групп

# In[31]:


#проверим, есть ли значения, которые встречаются в группах 246 и 247
display('Количество одинаковых значений в группе 246 и 247:',len(np.intersect1d(df.query('group == 246')['device_id'].unique(), df.query('group == "247"')['device_id'].unique())));


# In[32]:


#проверим, есть ли значения, которые встречаются в группах 246 и 248
display('Количество одинаковых значений в группе 246 и 248:',len(np.intersect1d(df.query('group == 246')['device_id'].unique(), df.query('group == "248"')['device_id'].unique())));


# In[33]:


#проверим, есть ли значения, которые встречаются в группах 247 и 248
display('Количество одинаковых значений в группе 247 и 248:',len(np.intersect1d(df.query('group == 247')['device_id'].unique(), df.query('group == "248"')['device_id'].unique())));


# В актуальном периоде осталось 240887 событий, что составляет 99% от исходного количества событий. Большее количество пользователей из всех групп так же осталось в актуальном периоде - 99,77%. Актуальный период выбран верно. 

# ## Воронка событий

# ### Какие события есть в логах и как часто они меняются

# In[34]:


#уникальные события
display(df_2w['event_name'].unique())


# Всего имеется 5 событий:
# * Tutorial - инструкция к приложению
# * MainScreenAppear - просмотр главной страницы
# * OffersScreenAppear - просмотр страницы предложений
# * CartScreenAppear - просмотр корзины
# * PaymentScreenSuccessful - переход на страницу успешной оплаты

# ### Cколько пользователей совершали каждое из этих событий

# In[35]:


#распределение пользователей по событиям количественное
users_count = df_2w.groupby('event_name').agg({'device_id': 'nunique'}).sort_values(by='device_id', ascending=False)
print(users_count) 


# In[36]:


#распределение пользователей по событиям в %
users_count['%'] = users_count['device_id'] / df['device_id'].nunique() * 100
users_count


# ### В каком порядке происходят события

# Не многие захотели ознакомиться с инструкцией к приложению. Все остальные события идут последовательно согласно логике приобретения продукта. Для визуализации воронки уберем Tutorial, потому как его прошло слишком мало пользователей.

# In[37]:


#удаление Tutorial
users_count=users_count[:4]


# In[38]:


#визуализация
users_count['%'].plot(kind='pie', label="", subplots=True, autopct='%1.0f%%')
plt.legend(bbox_to_anchor=(2, 1), loc='upper center')
plt.title('Распределение событий');


# Не все пользователи просматривают главную страницу. Страница предложений так же не всем интересна. Те, кто заходят в корзину, наверняка произведут покупку, их доля примерно равна и составляет 18%. По воронке событий посчитаем, какая доля пользователей проходит на следующий шаг воронки (от числа пользователей на предыдущем).

# ### Какая доля пользователей проходит на следующий шаг воронки событий

# In[39]:


#выделим названия событий
funnel_events = [event for event in df_2w.event_name.unique() if 'Screen' in event]
funnel_events


# In[40]:


#сделаем отдельный датафрейм для воронки
logs_funnel = df_2w[df_2w.event_name.isin(funnel_events)]


# In[41]:


logs_funnel.event_name.value_counts()


# In[42]:


#количество уникальных пользователей для каждого события
logs_funnel = logs_funnel.groupby('event_name')['device_id'].nunique().reset_index()
logs_funnel.sort_values(by='device_id',ascending=False)


# In[43]:


#сформируем таблицу с конверсией
report = df_2w.groupby('event_name').agg({'device_id': 'nunique'}).sort_values(by='device_id', ascending=False)
report['conversion'] = report['device_id'] / df_2w['device_id'].nunique()
report['previous_step'] = report['device_id'].shift()
report['step_conversion'] = report['device_id'] / report['previous_step']
display(report)


# In[44]:


#построим график
report['conversion'].plot(kind = 'bar', title='Конверсия').set(xlabel='Название события', ylabel='Доля пользователей');


# Самая большая доля пользователей приходится на просмотр главной страницы.

# ### На каком шаге теряете больше всего пользователей

# In[45]:


steps = ['MainScreenAppear', 'OffersScreenAppear', 'CartScreenAppear', 
       'PaymentScreenSuccessful']

test_df_funnel = df_2w.groupby(['group', 'event_name'], as_index=False).agg({'device_id': 'nunique'}).query('event_name != "Tutorial"')


# In[46]:


fig = go.Figure()

fig.add_trace(go.Funnel(

    y = steps,
    x = test_df_funnel.query('group == 246').set_index('event_name').loc[steps]['device_id'],
    textinfo = "value+percent previous",name='246' ))

fig.add_trace(go.Funnel(

    y = steps,
    x = test_df_funnel.query('group == 247').set_index('event_name').loc[steps]['device_id'],
    textinfo = "value+percent previous", name='247'))

fig.add_trace(go.Funnel(

    y = steps,
    x = test_df_funnel.query('group == 248').set_index('event_name').loc[steps]['device_id'],
    textinfo = "value+percent previous", name='248' ))


fig.update_layout(title={'x': 0.5, 'text': 'Воронка из шага в шаг'})
fig.show()


# На графике видно что просмотр главной страницы 100%, просмотр страницы предложений 62%, просмотр корзины 80% и оплата 95%.Больше всего пользователей теряется на эране предложений(OffersScreenAppear), было потеряно 38%. Далее наблюдается повышение процента перехода пользователей из одно шага в другой. Необходимо экран OffersScreenAppear сделать немного привлекательнее. 

# Какова же доля пользователей доходит от первого события до оплаты?:
# * 38% терям на шаге от MainScreenAppear до OffersScreenAppear
# * 19% теряем на шаге от OffersScreenAppear до CartScreenAppear
# * 13% теряем на шаге от CartScreenAppear до PaymentScreenSuccessful
# * 48% теряем от изначального количества пользователей до факта оплаты.

# ## Изучение результатов эксперимента

# ### Сколько пользователей в каждой экспериментальной группе

# In[47]:


#посмотрим число уникальных пользователей в каждой группе
users = df_2w.groupby('group')['device_id'].nunique().reset_index()
users


# Во всех трех группах примерно одинаковое количество пользователей.

# ### Находят ли статистические критерии разницу между выборками 246 и 247.

# Критерии успешного A/A-теста: 
# * Количество пользователей в различных группах различается не более, чем на 1%;
# * Для всех групп фиксируют и отправляют в системы аналитики данные об одном и том же;
# * Различие ключевых метрик по группам не превышает 1% и не имеет статистической значимости;
# * Попавший в одну из групп посетитель остаётся в этой группе до конца теста.

# In[48]:


#разница между группами
for i in range(0, 1):
    gr_list = [246, 247, 248, 246]
    print(f'Разница между группой {gr_list[i]} и {gr_list[i+1]} составляет {1 - df_2w.query("group == @gr_list[@i]")["device_id"].nunique() / df_2w.query("group == @gr_list[@i+1]")["device_id"].nunique():.2%}')


# Первый пункт соответствует. Второй пункт мы проверить не можем. Четвертый пункт был проверен в разделе №3 и убедились что пользователи не попадают в разные группы одновременно. Необходимо проверить различие ключевых метрик по группам. 

# In[49]:


# разделим пользователей по группам и событиям
events_groups = df_2w.pivot_table(index='event_name', columns='group', values='device_id', aggfunc='nunique', margins=True).reset_index().sort_values('All', ascending=False).reset_index(drop=True)
events_groups = events_groups.reindex([1, 2, 3, 4, 5, 0]).reset_index(drop=True)
events_groups


# In[50]:


# функция для проверки различий
def check_stats(successes1, successes2, trials1, trials2):
    # пропорция успехов в первой группе:
    p1 = successes1/trials1
    # пропорция успехов во второй группе:
    p2 = successes2/trials2
    # пропорция успехов в комбинированном датасете:
    p_combined = (successes1 + successes2) / (trials1 + trials2)
    # разница пропорций в датасетах
    difference = p1 - p2 
    # считаем статистику в ст.отклонениях стандартного нормального распределения
    z_value = difference / (p_combined * (1 - p_combined) * (1/trials1 + 1/trials2)) ** 0.5
    # задаем стандартное нормальное распределение (среднее 0, ст.отклонение 1)
    distr = st.norm(0, 1)  
    p_value = (1 - distr.cdf(abs(z_value))) * 2
    return p_value


# In[51]:


# создадим пустой датафрейм для записи результатов
groups_stats = pd.DataFrame({'events' : ['MainScreenAppear', 'OffersScreenAppear', 'CartScreenAppear', 'PaymentScreenSuccessful', 'Tutorial'],
                            '246-247' : [0, 0, 0, 0, 0],
                            '247-248' : [0, 0, 0, 0, 0],
                            '248-246' : [0, 0, 0, 0, 0]})

# пройдемся циклом по всем событиям и по всем парам групп
# для каждой пары групп и для каждого события посчитаем вероятности получения таких различий
for i in range(0, 5):
   for j in range(1, 4):
       # для группы 248 парной должна побывать и 246
       k = j + 1 if j < 3 else 1
       # общее количество пользователей в группе в пятой строке All
       groups_stats.iloc[i, j] = check_stats(events_groups.iloc[i, j], 
                                             events_groups.iloc[i, k], 
                                             events_groups.iloc[5, j], 
                                             events_groups.iloc[5, k])

# посмотрим, что получилось
groups_stats.iloc[:, [0, 1]].style.highlight_max(color='red', axis=None)


# Для всех событий разница в пропорциях недостаточна, чтобы говорить о статистически значимом различии. То есть между группами 246 и 247 разницы нет. 

# ### Cамое популярное событие.

# In[52]:


df_2w.pivot_table(index='event_name', columns='group', values='device_id', aggfunc='nunique').style.highlight_max(color='red', axis=None)


# Самое популярное событие - MainScreenAppear. Тут большой разницы между количеством пользователей нет. 

# ### Сравнение результатов с каждой из контрольных групп в отдельности по каждому событию. Сравните результаты с объединённой контрольной группой.

# Гипотеза H0: доли уникальных посетителей, побывавших на этапе воронки, одинаковы.
# 
# 
# Гипотеза H1: между долями уникальных посетителей, побывавших на этапе воронки, есть значимая разница.

# In[53]:


groups_stats.iloc[:, [0, 2, 3]].style.highlight_min(color='red', axis=None)


# Добавим в таблицу events_group столбец для объединенной группы 246+247 и сравним ее с экспериментальной группой.

# In[54]:


events_groups


# In[55]:


# создаем столбец данных для объединенной группы
events_groups['246+247'] = events_groups[246] + events_groups[247]
events_groups


# In[56]:


# создаем столбец со статистическими показателями
groups_stats['246+247-248'] = 1

# заполняем новый столбец
for i in range(0, 5):
   groups_stats.iloc[i, 4] = check_stats(events_groups.iloc[i, 3], 
                                       events_groups.iloc[i, 5], 
                                       events_groups.iloc[5, 3], 
                                       events_groups.iloc[5, 5])


# In[57]:


groups_stats.style.highlight_min(color='red', axis=None)


# ### Какой уровень значимости вы выбрали при проверке статистических гипотез выше? Посчитайте, сколько проверок статистических гипотез вы сделали. При уровне значимости 0.1 каждый десятый раз можно получать ложный результат. Какой уровень значимости стоит применить? Если вы хотите изменить его, проделайте предыдущие пункты и проверьте свои выводы.

# Так как в нашем эксперименте 4 пары групп и 4 этапа в воронке, число гипотез равно 16, и используя поправку Бонферрони, значение p-value следовало бы уменьшить до 0,003.Однако, для всех событий и всех пар групп p-value значительно больше 0.05 - минимальное значение 0,078.Следовательно, нулевую гипотезу не удалось отвергнуть, и шрифт не влияет на взаимодействие пользователя с приложением.

# ## Вывод

# Мы имеем данные за период с 25 июля по 7 августа 2019г. но часть данных была неполной. После удаления мы получили данные за период 01.08.2019 по 08.08.2019 (т.е. неделя).При изучении воронки событий обнаружили, что событие Tutorial самое малочисленное, т.е. большинство пользователей его игнорируют. От первого события(главный экран) до последнего (успешная оплата) доходит 47% пользователей.Больше всего пользователей мы теряем при переходе с главного экрана к ассортименту - 38%. По результатам А/А/В-теста не было выявлено различий между контрольными и эксперементальной группами, что говорит о том, что замена шрифтов никак не повлияла на пользователей ни в худшую, ни в лучшую сторону.
# 
# Рекомендации:
# * Больше всего пользователей "отваливается" на этапе главного экрана, возможно, его стоит доработать (сделать более привлекательным или сделать интерфейс более удобным). 
# * Если есть возможность, то можно повторить тестирование по шрифтам и провести его в течение 2 недель, чтобы убедиться, что различий по конверсии в разных группах действительно нет 
