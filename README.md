# Boston Housing Price Prediction

A C++ implementation of linear regression for predicting Boston housing prices.

## Description

This project implements a basic machine learning pipeline in pure C++ without external ML frameworks, featuring data loading via CSV, feature normalization, and linear regression training.

## Build

```bash
g++ -std=c++17 -O2 main.cpp -o boston_housing
```

## Structure

- `main.cpp` — Core implementation including DataFrame, normalization, and regression
