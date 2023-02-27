import pandas as pd
import numpy as np

# add corresponding dict components from d2 to d1
def sum_components(d1, d2):
    for k, v in d2.items():
        d1[k] = d1.get(k, 0) + v

# creates a predictor function built with the given weights
def predictor(weights):
    def predict(x):
        denominator = (weights['w0']**2 - x**2)**2 + 4 * x**2 * weights['beta']**2
        return weights['f0']**2 / denominator
    return predict

# gradient of the predictor function
def grad_predictor(weights, x):
    f0, w0, beta = weights.values()
    denominator = (w0**2 - x**2)**2 + 4 * x**2 * beta**2

    d_f0 = 2 * f0 / denominator
    d_w0 = - f0**2 * (4 * (w0**2 - x**2) * w0) / denominator**2
    d_beta = - f0**2 * (8 * x**2 * beta) / denominator**2

    return {'f0': d_f0, 'w0': d_w0, 'beta': d_beta}

# squared difference loss
def loss(predict, x, y):
    return (y - predict(x)) ** 2

# gradient of the loss
def grad_loss(weights, x, y):
    predict = predictor(weights)
    grad = {
        w: 2 * (predict(x) - y) * g
        for w, g in grad_predictor(weights, x).items()
    }
    return grad

# perform gradient descent on the given samples and weights
def gradient_descent(weights, batch, step):
    def penalty(val):
        if val < 0:
            return val * 1.5
        return 0
    g_loss = {'f0': 0, 'w0': 0, 'beta': 0}
    for x, y in batch:
        sum_components(g_loss, grad_loss(weights, x, y))
    for w, val in weights.items():
        weights[w] = val - step * (g_loss[w] + penalty(val)) / len(batch)

# perform gradient descent on training data to optimize weights
def train_predictor(train_data, step_size, iterations, batch_size=1, initial_w=None, record=None, quiet=False):
    if record is not None:
        history = []
    weights = initial_w or {'f0': 1, 'w0': 1, 'beta': 1}
    for i in range(iterations):
        for batch in batcher(train_data, batch_size):
            gradient_descent(weights, batch, step_size)

        if record is not None:
            history.append([*weights.values(), eval_weights(weights, train_data)])

        if not quiet and (i+1) % 50 == 0:
            print(f'Epoch {i+1}:')
            print_weights(weights)
            print(f'<Loss>: {eval_weights(weights, train_data):.03f}\n')
    if record is not None:
        out = pd.DataFrame(columns=['f0', 'w0', 'beta', 'error'], data=history)
        out.to_csv(record)
    return weights

# generates batches from the given dataset
def batcher(data, size):
    index = 0
    while index <= len(data):
        if index + size < len(data):
            yield data[index:index+size]
            index += size
        else:
            yield data[index:len(data)]
            break

# calculates the average squared error loss across the dataset
def eval_weights(weights, data, nparr=False):
    predict = predictor(weights)
    if nparr:
        sum_loss = np.sum(loss(predict, data[0], data[1]))
        return sum_loss / len(data[0])
    else:
        sum_loss = 0
        for x, y in data:
            sum_loss += loss(predict, x, y)
        return sum_loss / len(data)

def print_weights(weights):
    print(f'f0: {weights["f0"]:.03f}    w0: {weights["w0"]:.03f}    beta: {weights["beta"]:.03f}')
