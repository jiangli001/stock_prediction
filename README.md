# 命名格式规范
1. 文件命名：`小驼峰命名`
   - 例:  `sample`, `sampleProcessing`, `model`, `loadModel`
2. 类命名：`大驼峰命名`(大写字母开头，不同单词首字母大写区分)
   - 例: `class SampleProcessing(object):`
3. 函数，变量命名: `下划线命名法`
   - 例: `read_data(xxx,xxx)`, `stock_return = xxx`
4. 常量命名: `全大写 + 下划线`
   - 例: `EPOCHS = 10`


# 数据
考虑到数据文件一般较大，不适合用使用git上传。因此数据文件放在项目外面
1. 原始数据: `/root/rawData/`
   - 文件命名: `股票代码.csv`
2. 样本数据: `/root/sample/`
   - 文件命名: `股票代码_窗口大小.csv`

# 目录结构说明
```
├──root
    └── rawData                   # 项目外的文件, 原始数据文件文件
        └── xxxx.csv
    └── sample                    # 项目外的文件，样本文件
        └── xxxx.csv
    └── stock-prediction
        └── README.md
        └── conf                  # 配置文件
        └── strategies            # 各种策略模型
        └── util                  # 工具库
            └── sampleProcessing.py
```
