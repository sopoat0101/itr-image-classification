# itr-image-classification

## Group Name

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;เดี๋ยวมันก็ผ่านพ้น

## Group Members

| First Name | Last Name    | ID       |
| ---------- | ------------ | -------- |
| Satawat    | Thitisupakul | 60070093 |
| Sopoat     | Iamcharoen   | 60070101 |
| Nichapat   | Kachacheewa  | 61070059 |
| Wipawapat  | Hongsing     | 61070208 |

## Installation

### Packages

- os
- json
- pickle
- dotenv
- sklearn
- numpy
- openCV

```pip install pickle-mixin```

```pip install -U scikit-learn```

```pip install python-dotenv```

```pip install numpy```

```pip install opencv-python```

## Getting Started

1. run python file [learning_base.py](./learning_base.py)
2. run python file [handcraft_base.py](./handcraft_base.py)

### Change Dataset

- in [dataset](./dataset) folder have class folder.

```example
|example
├── dataset
│  └── 1 # Class Name
│      └── Image.jpg # Image file
│      └── ...
```

- in [datatest](./datatest) folder have json [test.json](./datatest/test.json) file.

```example
{
    "data":[
       {"imagePath":"./dataset/1/4.jpg","label":"1"},
       {"imagePath":"./dataset/1/4.jpg","label":"1"},
       ...
    ]
}
```

### Model

- Keep model in [model](./model) file name [train_learning_based.sav](./model/train_learning_based.sav)

- Keep features and lables in [model](./model) file name [train_handcraft_based.sav](./model/train_handcraft_based.sav)
