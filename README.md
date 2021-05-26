# itr-image-classification

## Group Name

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;เดี๋ยวมันก็ผ่านพ้น

## Group Members

| No. | First Name | Last Name    | ID       |     |
| --- | ---------- | ------------ | -------- | --- |
| 1   | Satawat    | Thitisupakul | 60070093 | IT  |
| 2   | Sopoat     | Iamcharoen   | 60070101 | IT  |
| 3   | Nichapat   | Kachacheewa  | 61070059 | IT  |
| 4   | Wipawapat  | Hongsing     | 61070208 | IT  |

## Installation

### Packages

- os
- json
- pickle
- dotenv
- sklearn
- numpy
- openCV
- tensorflow

```install pickle
pip install pickle-mixin
```

```install sklearm
pip install -U scikit-learn
```

```install dotenv
pip install python-dotenv
```

```install numpy
pip install numpy
```

```install opencv
pip install opencv-python
```

```install tensorflow
pip install --upgrade tensorflow
```

## Getting Started

### Hand Craft Base

1. run python file [start_hand_craft.py](./start_hand_craft.py) for create model.
2. run python file [start_hand_craft.py](./start_hand_craft.py) for test accuracy.

### Learning Base

1. run python file [start_learning_cnn.py](./start_learning_cnn.py) for create model.

2. run python file [start_learning_cnn.py](./start_learning_cnn.py) for test accuracy.

### Change Dataset

- in [dataset](./dataset) folder have class folder.

```example
|example
├── dataset
│  └── 0 # Class Name
│      └── Image.jpg # Image file
│      └── ...
```

- in [datatest](./datatest) folder have class folder too.

```example
|example
├── dataset
│  └── 0 # Class Name
│      └── Image.jpg # Image file
│      └── ...
```

### Model

- Keep model in folder [model](./model)

- Keep features and lables in [model](./model) file name [train_handcraft_based.sav](./model/train_handcraft_based.sav)
- Keep model hand craft base in [model](./model) file name [handcraft_model.sav](./handcraft_model.sav)
- Keep model learing base in [model](./model) file name [train_learning_based.h5](./model/train_handcraft_based.h5)

### Configuration

- You can change config in [.env](./.env) file such as dataset directory, file name for save model.
