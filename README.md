# Хакатон Glowbyte: Предсказание общего потребления электроэнергии

## Быстрый старт

Запуск ноутбука `ml_model.ipynb` и скрипта `main.py` производится в виртуальной среде на основе `requirements.txt`:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Расчет предсказаний

После запуска `ml_model.ipynb` сохраняется pickle экспериментов. Расчет предсказаний производится на основе лучшей модели в наборе экспериментов.

Запуск расчета предсказаний:

```
python main.py <path_to_data>
```

`path_to_data` - путь к входным данным формате CSV. Результат расчета сохраняется в файл `prediction_team21.csv` и содержит колонки date - дата и predict - предсказание суммарного потребления электроэнергии в этот день.

## Структура проекта

## Описание данных о погоде
https://rp5.ru/%D0%90%D1%80%D1%85%D0%B8%D0%B2_%D0%BF%D0%BE%D0%B3%D0%BE%D0%B4%D1%8B_%D0%B2_%D0%9A%D0%B0%D0%BB%D0%B8%D0%BD%D0%B8%D0%BD%D0%B3%D1%80%D0%B0%D0%B4%D0%B5

> _Прогноз погоды в исходных данных совпадает с этим датасетом_ 


T - 'Температура воздуха (градусы Цельсия) на высоте 2 метра над поверхностью земли'

Po - 'Атмосферное давление на уровне станции (миллиметры ртутного столба)'

U - 'Относительная влажность (%) на высоте 2 метра над поверхностью земли'

DD - 'Направление ветра (румбы) на высоте 10-12 метров над земной поверхностью, осредненное за 10-минутный период, непосредственно предшествовавший сроку наблюдения'

Ff - 'Cкорость ветра на высоте 10-12 метров над земной поверхностью, осредненная за 10-минутный период, непосредственно предшествовавший сроку наблюдения (метры в секунду)'

N - 'Общая облачность'

WW - 'Текущая погода, сообщаемая с метеорологической станции'

## Описание данных о праздниках
https://xmlcalendar.ru/

day - день (формат ММ.ДД)

type - тип дня: 1 - выходной день, 2 - рабочий и сокращенный (может быть использован для любого дня недели), 3 - рабочий день (суббота/воскресенье)

holiday - номер праздника (ссылка на атрибут id тэга holiday)

from - дата с которой был перенесен выходной день
суббота и воскресенье считаются выходными, если нет тегов day с атрибутом t=2 и t=3 за этот день