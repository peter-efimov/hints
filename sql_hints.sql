-- запустить запрос на параллельных машинах
/* PARALLEL(8) */ 

-- открыть новое окно запроса в oracle, вроде как могут даже параллельно работать
ctrl + shift + N

--функция LISTAGG, чтобы написать несколько записей в одно поле одной строки
LISTAGG(***, '; ' ON OVERFLOW TRUNCATE) OVER (PARTITION BY ***) AS ***

--В мегафоне при наличии в таблицах полей start_date & end_date возможно нужно их ограничивать
AND start_date <= to_date ('12.11.2018', 'dd.mm.yyyy')
AND end_date > to_date ('12.11.2018', 'dd.mm.yyyy')
-- 12.11.2018 - дата, на которую нам нужно получить данные

-- Сервисные таблицы с системной информацией
ALL_TABLES, ALL_TABS_COLS, ALL_VIEWS