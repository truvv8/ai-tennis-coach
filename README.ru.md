# AI Tennis Coach — Jetson Nano (starter)

[English version](README.md)

Пайплайн: камера → trt_pose (TensorRT) → углы суставов → скелет и подсказки
поверх видео. Смотреть можно в браузере с ноутбука (MJPEG) — монитор к Jetson
не нужен, всё запускается по SSH.

## Файлы

| Файл | Что делает |
|---|---|
| `main.py` | основной цикл: захват, поза, углы, фидбек, стрим |
| `pose_estimator.py` | обёртка над trt_pose (инференс + отрисовка скелета) |
| `angles.py` | углы локтя/плеча/колена/бёдер, скорость запястья |
| `smoothing.py` | EMA-сглаживание keypoints (без него углы дрожат на ±10°) |
| `analyze.py` | офлайн-разбор CSV: детект ударов, фазы, углы в контакте, оценка по эталону |
| `train_reference.py` | обучение эталонного шаблона на CSV с хорошими ударами |
| `build_engine.py` | одноразовая конвертация модели в TensorRT (на Jetson) |
| `streamer.py` | MJPEG-сервер без зависимостей, просмотр в браузере |
| `setup.sh` | вся установка на Jetson одним скриптом (шаги 2–5 ниже) |

## Развёртывание на Jetson Nano по SSH

### 0. Прошивка (один раз, если Nano чистый)
Скачай **JetPack 4.6.x** (последний для Nano) — образ SD-карты с
developer.nvidia.com, запиши через balenaEtcher. Первую загрузку можно пройти
headless: воткни microUSB в ноут и `screen /dev/tty.usbmodem* 115200`, создай
пользователя, подключи Wi-Fi/Ethernet.

### 1. Подключение
```bash
ssh <user>@jetson.local        # или по IP из роутера
head -n1 /etc/nv_tegra_release # проверить версию L4T (нужна R32.x)
```

> Шаги 2–5 можно не делать руками: `git clone` репозитория на Jetson и `bash setup.sh` —
> скрипт идемпотентный, при обрыве просто запусти ещё раз.

### 2. Подготовка системы
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks   # максимальная производительность

# swap 4GB — без него сборки падают по памяти
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

sudo apt update
sudo apt install -y python3-pip libopenblas-base libjpeg-dev zlib1g-dev git
```

### 3. PyTorch + torchvision (сборки NVIDIA под Jetson, не pip-овские!)
Возьми wheel **torch 1.10.0 для JetPack 4.6** со страницы
"PyTorch for Jetson" на forums.developer.nvidia.com:
```bash
wget <ссылка на torch-1.10.0-cp36-cp36m-linux_aarch64.whl>
pip3 install numpy torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# torchvision собирается из исходников под версию torch (~30-60 мин)
git clone --branch v0.11.1 https://github.com/pytorch/vision
cd vision && export BUILD_VERSION=0.11.1
python3 setup.py install --user && cd ..

python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# должно вывести: 1.10.0 True
```

### 4. torch2trt и trt_pose
```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt && sudo python3 setup.py install --plugins && cd ..

git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose && sudo python3 setup.py install && cd ..
```

### 5. Модель и проект
```bash
# с ноутбука:
rsync -av tennis-coach/ <user>@<jetson-ip>:~/tennis-coach/

# на Jetson:
cd ~/tennis-coach
cp ~/trt_pose/tasks/human_pose/human_pose.json .
# скачай resnet18_baseline_att_224x224_A_epoch_249.pth
# (ссылка в README trt_pose -> tasks/human_pose) и положи сюда же

python3 build_engine.py   # один раз, ~5-10 минут -> model_trt.pth
```

### 6. Запуск
```bash
python3 main.py --source 0 --stream 8080     # USB-камера
python3 main.py --source csi --stream 8080   # CSI-камера
```
Открой `http://<jetson-ip>:8080/` в браузере на ноутбуке — там видео со
скелетом, углами и подсказками. Ctrl+C останавливает.

Разбор записанного клипа с логом углов в CSV:
```bash
python3 main.py --source forehand.mp4 --csv forehand.csv --stream 0
```

Разбор сессии по CSV (работает где угодно, Jetson не нужен):
```bash
python3 analyze.py session.csv --reference good_forehand.csv
```
Найдёт удары по пикам скорости запястья, разрежет на фазы (замах → контакт →
проводка), напечатает углы в момент контакта и DTW-оценку похожести на
эталонный удар. Запиши один удар, которым доволен, — и каждая тренировка
будет оцениваться относительно него.

Ещё лучше — обучить эталон на многих хороших ударах (тренер, твой лучший
день). Шаблон запоминает среднюю траекторию каждого сустава и естественный
разброс, поэтому пишет конкретно: какой сустав, на сколько градусов и в какой
фазе отклонился:
```bash
python3 train_reference.py good1.csv good2.csv coach.csv -o forehand.json
python3 analyze.py session.csv --reference forehand.json
# ->  37%  <- elbow -66 deg vs ref in backswing
```
20–50 хороших ударов дают устойчивый шаблон.

Флаг `--record out.mp4` у main.py пишет размеченное видео в файл — удобно для
демо и разбора.

Чтобы не умирало при разрыве SSH: `tmux new -s coach`, внутри запускай main.py,
отцепляйся `Ctrl+B D`, возвращайся `tmux attach -t coach`.

## Что дальше
- Настроить пороги в `FEEDBACK_RULES` (main.py) по своим эталонным клипам
- Правила для подачи и бэкхенда
- Демо-гифка в README, звуковые подсказки между розыгрышами
