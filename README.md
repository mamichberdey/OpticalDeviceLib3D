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

- [`od_2d`](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/opticaldevicelib_1d.py) — моделирование в 2D  
- [`od_3d`](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/opticaldevicelib_2d.py) — моделирование в 3D  

Примеры использования — в ноутбуках:  
- [2D ноутбук](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/test_1d.ipynb)  
- [3D ноутбук](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/test_2d.ipynb)

---

## 🔍 Применение

Проект использовался для оценки формы отклонений поверхности линзы по экспериментальным данным и сравнения с результатами численного моделирования.

### 🔧 Оптическая схема:

<div align="center"><img src="https://github.com/user-attachments/assets/1ff41fb2-f32c-4329-9779-a010486bfca5" width="800"/></div>

---

## 🔬 Обработка экспериментальных данных

1. **Оценка PSF**  
   [Модуль PSF-оценивания](https://github.com/mamichberdey/OpticalDeviceLib3D/tree/main/PSF_ESTIMATOR)
     
   <div align="center"><img src="https://github.com/user-attachments/assets/4463d0fe-d073-431a-b575-0755c654a37d" width="800"/></div>

3. **Обработка экспериментальных данных**  
   [Кадровая сшивка и деконволюция Ричардсона–Люси](https://github.com/mamichberdey/OpticalDeviceLib3D/tree/main/FRAMES_PREPROCESSING)
     
   <div align="center"><img src="https://github.com/user-attachments/assets/699a930a-962b-47ac-88bc-df3350f7f9c0" width="800"/></div>

4. **Численное моделирование и сравнение с экспериментом**  
   - Построение модели с дефектами линз: [2D ноутбук](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/test_1d.ipynb), [3D ноутбук](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/test_2d.ipynb)
   - Сравнительный анализ с результатами наблюдений  
 <div align="center"><img src="https://github.com/user-attachments/assets/cf12e208-4b8e-483d-b8ae-7ad207612393" width="800"/></div>

---

## 🤖 Автоматизация с помощью нейросетей

Для автоматического оценивания формы дефектов линз применена **сверточная нейросеть (CNN)**:  
[Примеры оценивания с использованием CNN в ноутбуках: ](https://github.com/mamichberdey/OpticalDeviceLib3D/tree/main/CNN_PROJECT)
  
<div align="center"><img src="https://github.com/user-attachments/assets/135716ba-07d4-4155-8dec-cc684b7b8943" width="800"/></div>

