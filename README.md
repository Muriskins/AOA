# Analysis Of Activitis 
(временное название)
## О проекте
Это мой первый проект для студенческой производтсвенной практики в котрой мне предстоит реализовать aнализ трудовой 
деятельности персонала по видеоданным.
### Нужно реализовать:
1) работа с видеопотоком.
2) распознавание фрагментов кадра с людьми (YOLO) либо скелетным моделей людей в кадре (MediaPipe).
3) классификация состояний объектов с использованием дообученных моделей.
4) трекинг перемещения объектов в зоне видимости.
### Реализованно:
#### Обработка видео файла или видеопотока веб камеры через консольный ввод
Я использовал OpenCV для обработки каждого фрейма видеопотока и последующей записи в новый mp4 файл.