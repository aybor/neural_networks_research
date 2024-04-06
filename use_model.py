from pathlib import Path

import torch
import matplotlib.pyplot as plt

from linear_regressions import SimpleLinearRegressionNN, EnhancedLinearRegressionNN, ComplexLinearRegressionNN
from utils import find_best_fit_line, generate_points_around_line


def visualize_model_predictions(points, model, title):
    data = torch.tensor([val for point in points for val in point], dtype=torch.float32)
    data = data.view(1, -1)  # Подготовка вектора входных данных для модели

    model.eval()  # Переключение модели в режим оценки
    with torch.no_grad():
        predicted_a, predicted_b = model(data)[0].tolist()

    # Расчет коэффициентов методом наименьших квадратов для сравнения
    real_a, real_b = find_best_fit_line(points)

    x_values, y_values = zip(*points)

    # Рассчитываем y для математически рассчитанной линии и для линии, полученной моделью
    real_line_y = [real_a * x + real_b for x in x_values]
    model_line_y = [predicted_a * x + predicted_b for x in x_values]

    plt.figure(figsize=(10, 6))

    # Отображаем точки
    plt.scatter(x_values, y_values, color='blue', label='Random Points')

    # Отображаем математически рассчитанную линию
    plt.plot(x_values, real_line_y, color='green', label='Real Best Fit Line')

    # Отображаем линию, полученную моделью
    plt.plot(x_values, model_line_y, color='red', linestyle='--', label='Model Best Fit Line')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    base_dir = Path(__file__).parent

    input_size = 128
    simple_model = SimpleLinearRegressionNN(input_size)
    complex_model = ComplexLinearRegressionNN(input_size)
    enhanced_model = EnhancedLinearRegressionNN(input_size)

    simple_model.load_state_dict(torch.load(base_dir / 'models' / 'simple_model.pth'))
    complex_model.load_state_dict(torch.load(base_dir / 'models' / 'complex_model.pth'))
    enhanced_model.load_state_dict(torch.load(base_dir / 'models' / 'enhanced_model.pth'))

    rp, line_a, line_b = generate_points_around_line()

    visualize_model_predictions(rp, simple_model, 'Simple model')
    visualize_model_predictions(rp, complex_model, 'Complex model')
    visualize_model_predictions(rp, enhanced_model, 'Enhanced model')
