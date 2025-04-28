#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iomanip>
#include <map>
#include <cassert>

using namespace std;

struct DataFrame {
    vector<vector<double>> data;
    vector<string> column_names;
    map<string, size_t> column_index;
    size_t rows = 0;

    void add_column(const vector<double>& col, const string& name) {
        if (rows == 0) {
            rows = col.size();
        }
        else if (col.size() != rows) {
            throw runtime_error("Column size mismatch");
        }
        column_index[name] = data.size();
        data.push_back(col);
        column_names.push_back(name);
    }

    vector<double>& operator[](const string& name) {
        if (column_index.find(name) == column_index.end()) {
            throw runtime_error("Column not found: " + name);
        }
        return data[column_index[name]];
    }

    const vector<double>& operator[](const string& name) const {
        if (column_index.find(name) == column_index.end()) {
            throw runtime_error("Column not found: " + name);
        }
        return data.at(column_index.at(name));
    }

    size_t get_column_index(const string& name) const {
        return column_index.at(name);
    }
};

// split
struct TrainTestSplit {
    DataFrame train;
    DataFrame test;
    double test_size;
};

TrainTestSplit train_test_split(const DataFrame& df, double test_size = 0.2, unsigned seed = 42) {
    if (test_size <= 0 || test_size >= 1) {
        throw runtime_error("test_size must be between 0 and 1");
    }

    TrainTestSplit result;
    result.test_size = test_size;

    // shuffle index
    vector<size_t> indices(df.rows);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), mt19937(seed));

    size_t split_point = static_cast<size_t>(df.rows * (1 - test_size));

    for (const auto& name : df.column_names) {
        result.train.add_column({}, name);
        result.test.add_column({}, name);
    }

    // Fill
    for (size_t i = 0; i < df.rows; ++i) {
        size_t col_idx = 0;
        for (const auto& name : df.column_names) {
            if (i < split_point) {
                result.train.data[col_idx].push_back(df[name][indices[i]]);
            }
            else {
                result.test.data[col_idx].push_back(df[name][indices[i]]);
            }
            col_idx++;
        }
    }

    result.test.rows = result.test.data[0].size();
    result.train.rows = result.train.data[0].size();

    return result;
}

// Aggregation
double mean(const vector<double>& v) {
    if (v.empty()) throw runtime_error("Cannot compute mean of empty vector");
    return accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double covariance(const vector<double>& x, const vector<double>& y, double mean_x, double mean_y) {
    if (x.size() != y.size()) {
        throw runtime_error("Vectors must be of same size for covariance");
    }
    if (x.empty()) return 0.0;

    double cov = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        cov += (x[i] - mean_x) * (y[i] - mean_y);
    }
    return cov / x.size();
}

double pearson_correlation(const vector<double>& x, const vector<double>& y) {
    if (x.size() != y.size()) {
        throw runtime_error("Vectors must be of same size for correlation");
    }
    if (x.empty()) return 0.0;

    double mean_x = mean(x);
    double mean_y = mean(y);
    double cov = covariance(x, y, mean_x, mean_y);
    double var_x = covariance(x, x, mean_x, mean_x);
    double var_y = covariance(y, y, mean_y, mean_y);

    if (var_x == 0 || var_y == 0) {
        return 0.0;
    }

    return cov / sqrt(var_x * var_y);
}

// StandardScaler
void standard_scale(DataFrame& df, const vector<string>& features) {
    for (const auto& col : features) {
        vector<double>& vec = df[col];
        if (vec.empty()) continue;

        double m = mean(vec);
        double sq_sum = inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
        double s = sqrt(sq_sum / vec.size() - m * m);

        if (s == 0) s = 1;

        for (auto& val : vec) {
            val = (val - m) / s;
        }
    }
}

// Matrix
vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
    if (matrix.empty()) return {};

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    vector<vector<double>> result(cols, vector<double>(rows));

    for (size_t i = 0; i < rows; ++i) {
        if (matrix[i].size() != cols) {
            throw runtime_error("Inconsistent matrix dimensions in transpose");
        }
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

vector<vector<double>> matrix_multiply(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    if (a.empty() || b.empty()) return {};

    size_t a_rows = a.size();
    size_t a_cols = a[0].size();
    size_t b_rows = b.size();
    size_t b_cols = b[0].size();

    if (a_cols != b_rows) {
        throw runtime_error("Matrix dimensions mismatch for multiplication");
    }

    vector<vector<double>> result(a_rows, vector<double>(b_cols, 0));
    for (size_t i = 0; i < a_rows; ++i) {
        if (a[i].size() != a_cols) {
            throw runtime_error("Inconsistent matrix dimensions in multiplication");
        }
        for (size_t j = 0; j < b_cols; ++j) {
            for (size_t k = 0; k < a_cols; ++k) {
                if (b[k].size() != b_cols) {
                    throw runtime_error("Inconsistent matrix dimensions in multiplication");
                }
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

vector<vector<double>> matrix_inverse(const vector<vector<double>>& matrix) {
    if (matrix.empty()) return {};

    size_t n = matrix.size();
    if (n != matrix[0].size()) {
        throw runtime_error("Matrix must be square for inversion");
    }

    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw runtime_error("Inconsistent matrix dimensions in inversion");
        }
    }

    vector<vector<double>> inverse(n, vector<double>(n, 0));
    for (size_t i = 0; i < n; ++i) {
        inverse[i][i] = 1;
    }

    vector<vector<double>> temp = matrix;

    // 高斯-约当消元法
    for (size_t col = 0; col < n; ++col) {
        // 找到主元行
        size_t max_row = col;
        for (size_t row = col + 1; row < n; ++row) {
            if (fabs(temp[row][col]) > fabs(temp[max_row][col])) {
                max_row = row;
            }
        }

        // 检查奇异矩阵
        if (fabs(temp[max_row][col]) < 1e-10) {
            throw runtime_error("Matrix is singular or nearly singular");
        }

        // 交换行
        if (max_row != col) {
            swap(temp[col], temp[max_row]);
            swap(inverse[col], inverse[max_row]);
        }

        // 归一化主元行
        double pivot = temp[col][col];
        for (size_t j = col; j < n; ++j) {
            temp[col][j] /= pivot;
        }
        for (size_t j = 0; j < n; ++j) {
            inverse[col][j] /= pivot;
        }

        // 消去其他行
        for (size_t row = 0; row < n; ++row) {
            if (row != col && fabs(temp[row][col]) > 1e-10) {
                double factor = temp[row][col];
                for (size_t j = col; j < n; ++j) {
                    temp[row][j] -= factor * temp[col][j];
                }
                for (size_t j = 0; j < n; ++j) {
                    inverse[row][j] -= factor * inverse[col][j];
                }
            }
        }
    }

    return inverse;
}

class LinearRegression {
private:
    vector<double> coefficients;
    double intercept;

public:
    void fit(const vector<vector<double>>& X, const vector<double>& y) {
        if (X.empty()) {
            throw runtime_error("Input matrix X is empty");
        }
        if (X.size() != y.size()) {
            throw runtime_error("X and y must have same number of samples");
        }

        size_t n_samples = X.size();
        size_t n_features = X[0].size();

        for (const auto& row : X) {
            if (row.size() != n_features) {
                throw runtime_error("Inconsistent feature dimensions in X");
            }
        }

        // X = [1, X]
        vector<vector<double>> X_aug(n_samples, vector<double>(n_features + 1, 1.0));
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                X_aug[i][j + 1] = X[i][j];
            }
        }

        try {
            // (X^T X)^-1 * X^T * y
            auto XT = transpose(X_aug);
            auto XTX = matrix_multiply(XT, X_aug);
            auto XTX_inv = matrix_inverse(XTX);
            auto XTX_inv_XT = matrix_multiply(XTX_inv, XT);

            vector<vector<double>> y_mat(y.size(), vector<double>(1));
            for (size_t i = 0; i < y.size(); ++i) {
                y_mat[i][0] = y[i];
            }

            auto beta = matrix_multiply(XTX_inv_XT, y_mat);

            intercept = beta[0][0];
            coefficients.resize(n_features);
            for (size_t i = 0; i < n_features; ++i) {
                coefficients[i] = beta[i + 1][0];
            }
        }
        catch (const exception& e) {
            throw runtime_error("Linear regression failed: " + string(e.what()));
        }
    }

    vector<double> predict(const vector<vector<double>>& X) const {
        if (X.empty()) return {};

        vector<double> y_pred;
        for (const auto& x : X) {
            if (x.size() != coefficients.size()) {
                throw runtime_error("Input feature size mismatch in prediction");
            }

            double pred = intercept;
            for (size_t i = 0; i < coefficients.size(); ++i) {
                pred += coefficients[i] * x[i];
            }
            y_pred.push_back(pred);
        }
        return y_pred;
    }

    const vector<double>& get_coefficients() const { return coefficients; }
    double get_intercept() const { return intercept; }
};

// Evaluation
double mean_squared_error(const vector<double>& y_true, const vector<double>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw runtime_error("Vectors must be of same size for MSE");
    }
    if (y_true.empty()) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        sum += pow(y_true[i] - y_pred[i], 2);
    }
    return sum / y_true.size();
}

double r2_score(const vector<double>& y_true, const vector<double>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw runtime_error("Vectors must be of same size for R2 score");
    }
    if (y_true.empty()) return 0.0;

    double y_mean = mean(y_true);
    double ss_total = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_total += pow(y_true[i] - y_mean, 2);
        ss_res += pow(y_true[i] - y_pred[i], 2);
    }

    if (ss_total == 0) return 1.0;
    return 1.0 - (ss_res / ss_total);
}

DataFrame load_data(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    DataFrame df;
    string line;

    bool first = true;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<double> row;
        double val;
        while (iss >> val) {
            row.push_back(val);
        }

        if (first) {
            for (size_t i = 0; i < row.size(); ++i) {
                df.add_column({}, "Column_" + to_string(i));
            }
            first = false;
        }

        if (row.size() != df.column_names.size()) {
            throw runtime_error("Inconsistent number of columns in data file");
        }

        for (size_t i = 0; i < row.size(); ++i) {
            df.data[i].push_back(row[i]);
        }
    }

    if (df.column_names.size() == 14) {
        df.column_names = { "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
                          "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV" };
        df.column_index.clear();
        for (size_t i = 0; i < df.column_names.size(); ++i) {
            df.column_index[df.column_names[i]] = i;
        }
    }

    df.rows = df.data[0].size();

    return df;
}

int main() {
    try {
        // Load
        DataFrame data = load_data("C:/Users/Summer/Desktop/111/2/housing.txt");

        cout << "Data overview:" << endl;
        cout << "Number of samples: " << data.rows << endl;
        cout << "Number of features: " << data.column_names.size() << endl;

        // Split
        auto split = train_test_split(data, 0.2);
        DataFrame& train_data = split.train;
        DataFrame& test_data = split.test;

        cout << "\nTrain set size: " << train_data.rows << endl;
        cout << "Test set size: " << test_data.rows << endl;

        cout << "\nPearson correlation (on train set):" << endl;
        vector<pair<string, double>> correlations;
        for (const auto& col : train_data.column_names) {
            if (col != "MEDV") {
                try {
                    double corr = pearson_correlation(train_data[col], train_data["MEDV"]);
                    correlations.emplace_back(col, corr);
                    cout << col << ": " << fixed << setprecision(3) << corr << endl;
                }
                catch (const exception& e) {
                    cerr << "Error calculating correlation for " << col << ": " << e.what() << endl;
                }
            }
        }

        sort(correlations.begin(), correlations.end(),
            [](const auto& a, const auto& b) { return fabs(a.second) > fabs(b.second); });

        // Standard scale
        vector<string> features_to_scale = { "CRIM", "ZN", "INDUS", "NOX", "RM",
                                          "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT" };

        DataFrame train_scaled = train_data;
        standard_scale(train_scaled, features_to_scale);
        vector<double> y_train = train_scaled["MEDV"];

        DataFrame test_scaled = test_data;
        for (const auto& col : features_to_scale) {
            const auto& train_col = train_data[col];
            double m = mean(train_col);
            double sq_sum = inner_product(train_col.begin(), train_col.end(), train_col.begin(), 0.0);
            double s = sqrt(sq_sum / train_col.size() - m * m);
            if (s == 0) s = 1;

            for (auto& val : test_scaled[col]) {
                val = (val - m) / s;
            }
        }
        vector<double> y_test = test_scaled["MEDV"];

        // LSTAT -> MEDV
        vector<vector<double>> X_train_lstat;
        for (double val : train_scaled["LSTAT"]) {
            X_train_lstat.push_back({ val });
        }

        vector<vector<double>> X_test_lstat;
        for (double val : test_scaled["LSTAT"]) {
            X_test_lstat.push_back({ val });
        }

        try {
            LinearRegression model_simple;
            model_simple.fit(X_train_lstat, y_train);

            vector<double> y_train_pred_simple = model_simple.predict(X_train_lstat);
            double train_r2_simple = r2_score(y_train, y_train_pred_simple);

            vector<double> y_test_pred_simple = model_simple.predict(X_test_lstat);
            double test_r2_simple = r2_score(y_test, y_test_pred_simple);

            cout << "\nLSTAT -> MEDV:" << endl;
            cout << "Regression function: MEDV = " << model_simple.get_coefficients()[0]
                << " * LSTAT + " << model_simple.get_intercept() << endl;
            cout << "mean squared error: " << mean_squared_error(y_test, y_test_pred_simple) << endl;
            cout << "Train R^2 score: " << train_r2_simple << endl;
            cout << "Test R^2 score: " << test_r2_simple << endl;
        }
        catch (const exception& e) {
            cerr << "Error LSTAT -> MEDV: " << e.what() << endl;
        }

        // All features -> MEDV
        vector<vector<double>> X_train_multi;
        for (size_t i = 0; i < train_scaled.rows; ++i) {
            vector<double> row;
            for (const auto& col : features_to_scale) {
                row.push_back(train_scaled[col][i]);
            }
            X_train_multi.push_back(row);
        }

        vector<vector<double>> X_test_multi;
        for (size_t i = 0; i < test_scaled.rows; ++i) {
            vector<double> row;
            for (const auto& col : features_to_scale) {
                row.push_back(test_scaled[col][i]);
            }
            X_test_multi.push_back(row);
        }

        try {
            LinearRegression model_multi;
            model_multi.fit(X_train_multi, y_train);

            vector<double> y_train_pred_multi = model_multi.predict(X_train_multi);
            double train_r2_multi = r2_score(y_train, y_train_pred_multi);

            vector<double> y_test_pred_multi = model_multi.predict(X_test_multi);
            double test_r2_multi = r2_score(y_test, y_test_pred_multi);

            cout << "\nAll features -> MEDV:" << endl;
            cout << "Coefficients:" << endl;
            for (size_t i = 0; i < features_to_scale.size(); ++i) {
                cout << features_to_scale[i] << ": " << model_multi.get_coefficients()[i] << endl;
            }
            cout << "intercept: " << model_multi.get_intercept() << endl;
            cout << "mean squared error: " << mean_squared_error(y_test, y_test_pred_multi) << endl;
            cout << "Train R^2 score: " << train_r2_multi << endl;
            cout << "Test R^2 score: " << test_r2_multi << endl;
        }
        catch (const exception& e) {
            cerr << "Error All features -> MEDV: " << e.what() << endl;
        }

        // First 4 features -> MEDV
        vector<string> top_features;
        for (size_t i = 0; i < 4 && i < correlations.size(); ++i) {
            top_features.push_back(correlations[i].first);
        }

        vector<vector<double>> X_train_top;
        for (size_t i = 0; i < train_scaled.rows; ++i) {
            vector<double> row;
            for (const auto& col : top_features) {
                row.push_back(train_scaled[col][i]);
            }
            X_train_top.push_back(row);
        }

        vector<vector<double>> X_test_top;
        for (size_t i = 0; i < test_scaled.rows; ++i) {
            vector<double> row;
            for (const auto& col : top_features) {
                row.push_back(test_scaled[col][i]);
            }
            X_test_top.push_back(row);
        }

        try {
            LinearRegression model_top;
            model_top.fit(X_train_top, y_train);

            vector<double> y_train_pred_top = model_top.predict(X_train_top);
            double train_r2_top = r2_score(y_train, y_train_pred_top);

            vector<double> y_test_pred_top = model_top.predict(X_test_top);
            double test_r2_top = r2_score(y_test, y_test_pred_top);

            cout << "\nFirst 4 features -> MEDV:" << endl;
            cout << "Used features: ";
            for (const auto& col : top_features) {
                cout << col << " ";
            }
            cout << endl;
            cout << "mean squared error: " << mean_squared_error(y_test, y_test_pred_top) << endl;
            cout << "Train R^2 score: " << train_r2_top << endl;
            cout << "Test R^2 score: " << test_r2_top << endl;
        }
        catch (const exception& e) {
            cerr << "Error First 4 features -> MEDV: " << e.what() << endl;
        }

    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}