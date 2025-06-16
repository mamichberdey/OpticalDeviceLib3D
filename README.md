# OpticalDeviceLib3D 🛠️🔦

**OpticalDeviceLib3D** — проект для моделирования оптических схем с использованием *синхротронного рентгеновского излучения*.

Возможности проекта:
- Точечные и протяжённые источники с заданной энергией или длиной волны  
- Отверстия и апертуры  
- Параболические [CRL-линзы](https://en.wikipedia.org/wiki/Compound_refractive_lens)  
- Построение схем в 2D и 3D  

---

## 📚 Библиотеки

В проекте реализованы две основные Python-библиотеки:

- [`od_2d`](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/opticaldevicelib_1d.py) — построение 2D-схем  
- [`od_3d`](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/opticaldevicelib_2d.py) — моделирование в 3D  

Примеры использования — в ноутбуках:  
- [2D ноутбук](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/test_1d.ipynb)  
- [3D ноутбук](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/test_2d.ipynb)

---

## 🔍 Применение

Проект использовался для анализа формы отклонений поверхности линзы по экспериментальным данным и сравнения с результатами численного моделирования.

### 🔧 Оптическая схема:

<img src="https://github.com/user-attachments/assets/1ff41fb2-f32c-4329-9779-a010486bfca5" width="800"/>

---

## 🔬 Обработка экспериментальных данных

1. **Оценка PSF**  
   [Модуль PSF-оценивания](https://github.com/mamichberdey/OpticalDeviceLib3D/tree/main/PSF_ESTIMATOR)  
   <img src="https://github.com/user-attachments/assets/4463d0fe-d073-431a-b575-0755c654a37d" width="800"/>

2. **Сшивка и деконволюция**  
   [Кадровая сшивка и деконволюция Ричардсона–Люси](https://github.com/mamichberdey/OpticalDeviceLib3D/tree/main/FRAMES_PREPROCESSING)  
   <img src="https://github.com/user-attachments/assets/e2a2bf4f-baa7-4451-b7eb-17b7bb4b3278" width="800"/>

3. **Численное моделирование и сравнение с экспериментом**  
   - Построение модели с дефектами линз  
   - Сравнительный анализ с результатами наблюдений

---

## 🤖 Автоматизация с помощью нейросетей

Для автоматического оценивания формы дефектов линз применена **сверточная нейросеть (CNN)**:  
[Подробнее о CNN-модуле](https://github.com/mamichberdey/OpticalDeviceLib3D/tree/main/CNN_PROJECT)  
<img src="https://github.com/user-attachments/assets/54a63a14-30ef-4673-b7c9-40d82ee50eb4" width="800"/>
