# About Project
Этот проект о распознавании эмоций в режиме реального времени используя CNN.
Мотивацией для написания этого проекта послужило мое желание глубже познакомиться с машшиным обучением, глубокими нейронными сетями и потребность написать мою выпускную работу в университете.

Человеческие эмоции можно разделить на 7 базовых эмоций: злость, отвращение, страх, радость, грусть, удивление и нейтральность.
Для решения данной задачи существуют разные подходы, но в своем решении я решил использовать СНН, потому что этот вид сетей отлично справляется с задачи распознавания и классификации изображений.

Проект состоит из двух частей:
1. Первая часть содержит в код подготовки данных и обучения сети, а так же код распознавания эмоций по изозображению с вебкамеры в режиме реального времени (этот репозиторий)
2. Вторая часть - веб приложение, которое предоставяет возможность определения лиц и эмоций на загруженном изображении. (https://github.com/FiseD/web_emotion_recognition)

Пожалуйста, дайте мне знать если Вы захотите использовать моей проект или его части в своей разработке. Так же я обязательно постараюсь ответить на все вопросы, которые могут возникнуть при ознакомлении с моим проектом.
Весь проект является личной разработкой.

# About Dataset
В моем проекте используется датасет FER-2013 и несколько других значительно меньших по размеру датасетов.
FER-2013 содержит в себе ~36к картинок размером 48х48 в градация серого, которые содержат одну из 7 эмоций описаных мной в части About Project.
Но много изображений содержат лицо повернутое боком, закрытое руками и тд. На мой взгляд данные на которые обучается сеть должны быть как можно более чистыми, поэтому на этапе обработки данных перед тренировкой сети я отчищаю датасет от изображений на которых не может быть найдено лицо. В результат данной операции остается ~11k изображений.
Есть возможность добавления дополнительных изображений в тренировочкую и проверочную выборки, куда я добавил данные из датасетов CK+, Jaffe, ..., что в итоге дало мне датасет размером 15к.

# About CNN
Сверточные нейронные сети или CNN часто используются в обработке изображений и имеет схему работы подобно нашему глазу. Этот типо сетей отлично подходит для выделения характерных признаковых на изображении и последующей классификации по этим признакам. Расписывать подробнее о сверточных нейронных сетях я не буду, потому что в интернете есть много хорошей информации по данному типу нейросетей.

Структура моей сети и всех ее гиперпараметров была подобрана в результате большого кол-ва экспериментов.
Начальный вид моей сети и ее результаты:

Конечный вид и ее результаты:

# Results
В результате обучения на последней эмохе точность моей сети составила 70%. На первый взгляд этого может показаться мало, но как я уже говорил эмоции имеют разное кол-во изображений, поэтому некоторые эмоций в точности уступают другим. К тому же не всегда можно определить эмоцию по одному лишь изображению, без знания условий.

Чаще всего если эмоций определена не правильно, но правильный вариант стоит на втором месте по вероятности

# Instructions
Точкой входа в приложение является файл app.py
python app.py train - тренировка сети
python app train --extra - тренировка сети с ипользованием дополнительных данных(убедитесь что создана категория extra-images)
python app start <device> (device - номер вашего устройства вебкамеры в системе)
  
Для остановки приложения нажмити комбинацию CTRL + C. После этого запуститься Dash приложение с визуализацией полученных данных за весь период использования.

# Dependencies
- python 3
- numpy
- opencv
- pandas
- tensorflow
- tflearn
- dash(with components)

Вы можете установить все зависимости из файла requirements.txt

# Paper
todo
