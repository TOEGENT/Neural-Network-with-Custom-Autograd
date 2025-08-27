import numpy as np

np.random.seed(4245)


class Param:
    """
    Класс Param хранит:
    - value: текущее значение параметра (вес или bias, либо промежуточное вычисление)
    - history: множество Param, от которых он зависит
    - chain: словарь для хранения "цепочки производных" (ключ — Param, значение — производная)

    Идея: при каждой операции над Param создаётся новый объект Param,
    а в chain предыдущих сохраняется, как к этому новому результату протекает градиент.
    """

    def __init__(self, value, raw_params=0):
        self.value = value
        self.chain = {self: 1}  # изначально dself/dself = 1
        if raw_params:
            # Если это исходный вес или bias, то он сам себе предок
            self.history = set()
            self.history.add(self)
        else:
            # Если Param получен в результате вычислений, history будет дополняться позже
            self.history = set()

    def __add__(self, other):
        """
        Реализация сложения Param (+ число).
        При сложении двух Param объединяем их history.
        В chain добавляем правило: d(result)/dp = d(self)/dp + d(other)/dp.
        """
        if isinstance(other, Param):
            result = Param(self.value + other.value)
            result.history = self.history | other.history
            for p in result.history:
                p.chain[result] = p.chain.get(result, 0) + p.chain.get(other, 0) + p.chain.get(self, 0)
        else:  # если other обычное число
            result = Param(self.value + other)
            result.history = self.history
            for p in result.history:
                p.chain[result] = p.chain.get(result, 0) + p.chain.get(self, 0)
        return result

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """
        Вычитание Param или числа.
        Аналогично сложению, только добавляем знак «-» для второго аргумента.
        """
        if isinstance(other, Param):
            result = Param(self.value - other.value)
            result.history = self.history | other.history
            for p in result.history:
                p.chain[result] = p.chain.get(result, 0) + p.chain.get(self, 0) + p.chain.get(other, 0) * -1
        else:
            result = Param(self.value - other)
            result.history = self.history
            for p in result.history:
                p.chain[result] = p.chain.get(result, 0) + p.chain.get(self, 0)
        return result

    def __rsub__(self, other):
        """
        Обратное вычитание: число - Param или Param - Param.
        """
        if isinstance(other, Param):
            result = Param(other.value - self.value)
            result.history = self.history | other.history
            for p in result.history:
                p.chain[result] = p.chain.get(result, 0) + p.chain.get(self, 0) * -1 + p.chain.get(other, 0)
        else:
            result = Param(other - self.value)
            result.history = self.history
            for p in self.history:
                p.chain[result] = p.chain.get(result, 0) + p.chain.get(self, 0) * -1
        return result

    def __pow__(self, power, modulo=None):
        """
        Возведение Param в степень.
        d(x^n)/dx = n * x^(n-1).
        """
        result = Param(self.value ** power)
        result.history = self.history
        for p in result.history:
            p.chain[result] = p.chain.get(result, 0) + p.chain.get(self, 0) * power * self.value ** (power - 1)
        return result

    def __mul__(self, other):
        """
        Умножение Param на Param или Param на число.
        d(x*y)/dx = y ; d(x*y)/dy = x
        """
        if isinstance(other, Param):
            result = Param(self.value * other.value)
            result.history = self.history | other.history
            for p in result.history:
                p.chain[result] = p.chain.get(result, 0) + p.chain.get(self, 0) * other.value + p.chain.get(other,
                                                                                                            0) * self.value
        else:  # умножение на число
            result = Param(self.value * other)
            result.history = self.history
            for p in result.history:
                p.chain[result] = p.chain.get(result, 0) + other * p.chain[self]
        return result

    def __truediv__(self, other):
        """
        Деление Param на Param или на число.
        d(x/y)/dx = 1/y ; d(x/y)/dy = -x/y^2
        """
        if isinstance(other, Param):
            result = Param(self.value / other.value)
            result.history = self.history | other.history
            for p in result.history:
                p.chain[result] = p.chain.get(result, 0) + p.chain.get(self) * (1 / other.value) + p.chain.get(
                    other) * (-self.value / other.value ** 2)
        else:
            result = Param(self.value / other)
            result.history = self.history
            for p in result.history:
                p.chain[result] = p.chain.get(result, 0) + p.chain.get(self) * 1 / other
        return result

    def __neg__(self):
        return self * (-1)


def der_act(act, alpha=0.01):
    """ Производная активации ReLU/LeakyReLU. """
    if act > 0:
        return 1
    else:
        return alpha


def act(param: Param, alpha=0.01):
    """
    Активация (LeakyReLU).
    Если param > 0 → пропускаем как есть.
    Если param <= 0 → умножаем на alpha.

    В chain сохраняем: d(act(param))/dp = der_act(param.value) * d(param)/dp
    """
    if isinstance(param, Param):
        if param.value > 0:
            activated = Param(param.value)
        else:
            activated = Param(param.value * alpha)
        activated.history = param.history
        for p in activated.history:
            p.chain[activated] = p.chain.get(param, 0) * der_act(param.value)
        return activated
    else:
        return param if param > 0 else param * alpha


from random import uniform


def structurise(ws, bs, architecture):
    """
    Преобразует "плоский" список весов и bias в структурированное представление,
    чтобы удобно работать при прямом проходе.

    architecture = [кол-во входов, нейронов в слое 1, нейронов в слое 2, ..., выходов]

    Пример:
        architecture = [2, 3, 2]
        ws (длина 6) -> [[w11,w12],[w21,w22],[w31,w32]]
        bs (длина 3) -> [[b1,b2,b3]]

    Возвращает:
        (structured_ws, structured_bs)
        structured_ws: список слоёв, где каждый слой = список нейронов,
                       а каждый нейрон = список его весов
        structured_bs: список слоёв, где каждый слой = список bias
    """
    ws_per_layer = [architecture[i] * architecture[i + 1] for i in range(len(architecture) - 1)]
    structured_ws = [[] for _ in range(len(architecture) - 1)]
    architecture_structured = [[architecture[i], architecture[i + 1]] for i in range(len(architecture) - 1)]

    start = 0
    for i in range(len(architecture_structured)):
        slice = ws[start:start + ws_per_layer[i]]
        # Разбиваем веса для каждого нейрона слоя
        structured_ws[i] = [slice[j:j + architecture_structured[i][0]] for j in
                            range(0, len(slice), architecture_structured[i][0])]
        start += ws_per_layer[i]

    step = 0
    structured_bs = []
    for b_num in architecture[1:]:  # bias только для скрытых и выходных слоёв
        slice = bs[step: b_num + step]
        structured_bs.append(slice)
        step += b_num

    return (structured_ws, structured_bs)


def init_params(architecture):
    """
    Инициализация весов и bias для сети с заданной архитектурой.

    - Веса инициализируются случайно из U(-1, 1).
    - Bias инициализируются нулями.

    Возвращает:
        (ws, bs) — два списка "плоских" Param, которые потом структурируются.
    """
    ws_per_layer = [architecture[i] * architecture[i + 1] for i in range(len(architecture) - 1)]

    ws = []
    for _ in range(sum(ws_per_layer)):
        ws.append(Param(np.random.uniform(-1, 1), 1))  # raw_params=1, т.к. это обучаемые параметры

    bs = []
    for b_num in architecture[1:]:
        for _ in range(b_num):
            bs.append(Param(0, 1))  # bias инициализируем нулями

    return (ws, bs)


def forward(xs, ws, bs):
    """
    Прямое распространение сигнала (forward pass) с вычислением градиентов.
    xs: входной вектор (список чисел или Param).
    ws: структурированные веса.
    bs: структурированные bias.

    На каждом слое:
        y = act(sum(w_i * x_i) + b)

    Возвращает:
        список выходных Param текущего слоя (последний слой = выход сети).
    """
    current_xs = xs
    for i in range(len(ws)):  # по каждому слою
        new_xs = []
        for m in range(len(ws[i])):  # по каждому нейрону в слое
            # линейная комбинация входов + bias → активация
            new_xs.append(act(sum([ws[i][m][xi] * current_xs[xi] for xi in range(len(ws[i][m]))]) + bs[i][m]))
        current_xs = new_xs  # выход слоя становится входом для следующего
    return current_xs


def forward_no_grad(xs, ws, bs, xs_min=None, xs_max=None, ys_min=None, ys_max=None, norm=True):
    """
    Прямой проход БЕЗ накопления градиентов.
    Используется для оценки работы сети после обучения.

    Поддерживает нормализацию входов и денормализацию выходов.
    """
    if norm:
        current_xs = normalize_data(xs, xs_min, xs_max)
    else:
        current_xs = xs

    for i in range(len(ws)):
        new_xs = []
        for m in range(len(ws[i])):
            new_xs.append(act(sum([ws[i][m][xi] * current_xs[xi] for xi in range(len(ws[i][m]))]) + bs[i][m]))
        current_xs = new_xs

    if norm:
        current_xs = denormalize_data(current_xs, ys_min, ys_max)

    return current_xs
def normalize_data(data,data_min,data_max):
    result=[]
    for xi in range(len(data)):
        result.append((data[xi]-data_min)/(data_max-data_min))
    return result
def denormalize_data(data,data_min,data_max):
    result=[]
    for xi in range(len(data)):
        result.append(data[xi]*(data_max-data_min)+data_min)
    return result
# Подготовка данных
r = [i for i in range(-10, 11)]  # входы
XS = [i for i in r]
YS = [(i)**3 for i in r]  # целевая функция y = x^3

# Нормализация данных
xs_min, xs_max = min(XS), max(XS)
ys_min, ys_max = min(YS), max(YS)

norm_xs_no_structure = normalize_data(XS, xs_min, xs_max)
nxs = [[x] for x in norm_xs_no_structure]
norm_ys_no_structure = normalize_data(YS, ys_min, ys_max)
nys = [[y] for y in norm_ys_no_structure]

ys = [[y] for y in YS]
xs = [[x] for x in XS]

# Архитектура сети: [1 вход → 6 скрытых нейронов → 1 выход]
arch = [1, 8,8, 1]
epochs = 500
lr_w = 0.1
lr_b = 0.1

all_iterations_data = []

for I in range(15):  # несколько запусков обучения
    ws, bs = init_params(arch)  # инициализация
    structured_ws, structured_bs = structurise(ws, bs, arch)

    for epoch in range(epochs):
        total_loss = 0
        for xindex in range(len(r)):
            # Прямой проход
            y_pred = forward(nxs[xindex], structured_ws, structured_bs)[0]
            y_true = nys[xindex][0]

            # Функция ошибки (MSE)
            loss = (y_true - y_pred) ** 2
            total_loss += loss.value

            # Обновление весов и bias через градиентный спуск
            for w in ws:
                grad = w.chain[list(w.chain.keys())[-1]]
                w.value -= lr_w * grad
                w.chain = {w: 1}
                w.history = {w}
            for b in bs:
                grad = b.chain[list(b.chain.keys())[-1]]
                b.value -= lr_b * grad
                b.chain = {b: 1}
                b.history = {b}

    # Сохраняем результаты после обучения
    well_done_ws = [i.value for i in ws]
    well_done_bs = [i.value for i in bs]
    structured_well = structurise(well_done_ws, well_done_bs, arch)

    forward_xs = []
    for i in r:
        forward_xs.append(forward_no_grad([i], structured_well[0], structured_well[1], xs_min, xs_max, ys_min, ys_max,True))

    print(I, "loss", total_loss)

    all_iterations_data.append({
        'iteration': I,
        'true_values': XS,
        'predicted_values': forward_xs,
        'difference': [abs(true_val - pred_val[0]) for true_val, pred_val in zip(XS, forward_xs)]
    })

import matplotlib.pyplot as plt

def plot_combined_overlaid(data_list, x_range):
    """
    Рисует аппроксимацию y = x^3 для всех итераций обучения.
    Черная пунктирная линия — истинная функция.
    Цветные линии — предсказания сети.
    """
    plt.figure(figsize=(12, 6))

    true_y = [x**3 for x in x_range]
    plt.plot(x_range, true_y, 'k--', linewidth=2, label='y = x^3 (true)')

    for data in data_list:
        y_preds = [p[0] for p in data['predicted_values']]
        plt.plot(x_range, y_preds, alpha=0.6)

    plt.title(f"Аппроксимация y = x^3 нейросетью {arch}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_combined_overlaid(all_iterations_data, r)
