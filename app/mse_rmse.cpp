#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

// Función para calcular el MSE entre dos vectores
double mean_squared_error(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Los vectores deben tener el mismo tamaño.");
    }

    double mse = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_pred[i];
        mse += diff * diff;
    }
    mse /= y_true.size();
    return mse;
}

// Función para calcular el RMSE entre dos vectores
double root_mean_squared_error(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    return std::sqrt(mean_squared_error(y_true, y_pred));
}

PYBIND11_MODULE(mse_rmse, m) {
    m.def("mean_squared_error", &mean_squared_error, "Calculate Mean Squared Error (MSE)",
          py::arg("y_true"), py::arg("y_pred"));
    m.def("root_mean_squared_error", &root_mean_squared_error, "Calculate Root Mean Squared Error (RMSE)",
          py::arg("y_true"), py::arg("y_pred"));
}
