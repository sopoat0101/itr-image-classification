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

### Configuration

- You can change config in [.env](./.env) file such as dataset directory, file name for save model.
