"""CPU NumPy fitting for the two linear heads; no inference-time training."""
from collections import Counter
import numpy as np


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def prompt_weights(prompt_ids):
    counts = Counter(prompt_ids)
    weights = np.array([1 / counts[key] for key in prompt_ids])
    return weights / weights.sum()


def fit_classifier(features, labels, prompt_ids, classes, regularization, steps=500):
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), .1)
    standardized = (features - mean) / scale
    target = np.eye(classes)[labels]
    sample_weights = prompt_weights(prompt_ids)
    matrix = np.zeros((standardized.shape[1], classes))
    bias = np.zeros(classes)
    parameters = [matrix, bias]
    first_moment = [np.zeros_like(value) for value in parameters]
    second_moment = [value.copy() for value in first_moment]
    for step in range(1, steps + 1):
        probabilities = softmax(standardized @ matrix + bias)
        residual = (probabilities - target) * sample_weights[:, None]
        gradients = [standardized.T @ residual + regularization * matrix, residual.sum(axis=0)]
        for index, gradient in enumerate(gradients):
            first_moment[index] = .9 * first_moment[index] + .1 * gradient
            second_moment[index] = .999 * second_moment[index] + .001 * gradient * gradient
            corrected_mean = first_moment[index] / (1 - .9**step)
            corrected_variance = second_moment[index] / (1 - .999**step)
            parameters[index] -= .03 * corrected_mean / (np.sqrt(corrected_variance) + 1e-8)
    return {'mean': mean, 'scale': scale, 'p0': matrix, 'p1': bias}


def predict(model, features):
    normalized = (features - model['mean']) / model['scale']
    return softmax(normalized @ model['p0'] + model['p1'])
