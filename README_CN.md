# Boston Housing 波士顿房价预测

基于 C++ 的线性回归实现，用于预测波士顿房价。

## 描述

该项目在不依赖外部 ML 框架的情况下，使用纯 C++ 实现了一个完整的机器学习流程，包括 CSV 数据加载、特征归一化和线性回归训练。

## 构建

```bash
g++ -std=c++17 -O2 main.cpp -o boston_housing
```

## 结构

- `main.cpp` — 核心实现，包含 DataFrame、归一化、回归等模块
