# OpticalDeviceLib3D 🛠️🔦

Проект для моделирования оптических схем с использованием синхротронного рентгеновского излучения.
В схеме могут использоваться: точечный или протяженный источник с заданной энергией или длиной волны, отверстия, параболические [CRL линзы](https://en.wikipedia.org/wiki/Compound_refractive_lens) и т.д.  

Для проекта построены две Python-библиотеки:
[od_2d](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/opticaldevicelib_1d.py) и 
[od_3d](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/opticaldevicelib_2d.py)
для построения оптических схем в 2D и 3D пространствах, соответственно. Непосредственное построение оптических схем проводится в
[notebook_2d](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/test_1d.ipynb) и [notebook_3d](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/test_2d.ipynb).

В частности, проект использовался для оценивания формы отклонений поверхности линзы по экспериментальным данным и сравнением с результатами численного моделирования.  
Оптическая схема:
![image](https://github.com/user-attachments/assets/1ff41fb2-f32c-4329-9779-a010486bfca5)



Для реализации поставленной задачи, были проведены:  
Качественная обработка результатов экспериментальных данных. 

Обработка проводилась с [оцениванием PSF](https://github.com/mamichberdey/OpticalDeviceLib3D/tree/main/PSF_ESTIMATOR).
![image](https://github.com/user-attachments/assets/4463d0fe-d073-431a-b575-0755c654a37d)

[Сшивка кадров экспериментальных данных и деконволюция Ричардсона-Люси](https://github.com/mamichberdey/OpticalDeviceLib3D/tree/main/FRAMES_PREPROCESSING). 
Построение оптической схемы 
![image](https://github.com/user-attachments/assets/e2a2bf4f-baa7-4451-b7eb-17b7bb4b3278)

Построение численной оптической схемы, с заданной формой дефектов линз, и сравнение с экспериментальными данными.

Для автоматизации оценивания формы дефектов линз, была применена [CNN](https://github.com/mamichberdey/OpticalDeviceLib3D/tree/main/CNN_PROJECT).
![image](https://github.com/user-attachments/assets/54a63a14-30ef-4673-b7c9-40d82ee50eb4)

