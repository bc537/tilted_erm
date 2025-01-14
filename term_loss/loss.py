import numpy as np
from qiskit_machine_learning.utils.loss_functions.loss_functions import Loss

class L1Loss_ERM(Loss):
    def __init__(self):
        super().__init__

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)
        N = len(predict)
        if len(predict.shape) <= 1:
            return np.abs(predict - target)
        else:
            return np.linalg.norm(predict - target, ord=1, axis=tuple(range(1, len(predict.shape))))

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)
        N = len(predict)
        return np.sign(predict - target)

class L1Loss_TERM(Loss):
    def __init__(self, t: float = 1.0):
        super().__init__
        self.t = t

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        t = self.t
        self._validate_shapes(predict, target)
        N = len(predict)
        if len(predict.shape) <= 1:
            return 1/t * np.log(1/N * np.exp(t *  np.abs(predict - target)))
        else:
            return 1/t * np.log(1/N * np.exp(t * np.linalg.norm(predict - target, ord=1, axis=tuple(range(1, len(predict.shape))))))

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        t = self.t
        self._validate_shapes(predict, target)
        E = np.exp(t * np.abs(predict - target))
        d = np.sum(E)
        return (np.sign(predict - target) * E) / d
    
class L2Loss_ERM(Loss):

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)
        N = len(predict)
        if len(predict.shape) <= 1:
            return (predict - target) ** 2
        else:
            return np.linalg.norm(predict - target, axis=tuple(range(1, len(predict.shape)))) ** 2

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)
        N = len(predict)
        return 2 * (predict - target)

class L2Loss_TERM(Loss):
    def __init__(self, t: float = 1.0):
        super().__init__
        self.t = t

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        t = self.t
        self._validate_shapes(predict, target)
        N = len(predict)
        if len(predict.shape) <= 1:
            return 1/t * np.log(1/N * np.exp(t *  (predict - target) ** 2))
        else:
            return 1/t * np.log(1/N * np.exp(t * np.linalg.norm(predict - target, axis=tuple(range(1, len(predict.shape)))) ** 2))

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        t = self.t
        self._validate_shapes(predict, target)
        E = np.exp(t * (predict - target) ** 2)
        d = np.sum(np.exp(t * (predict - target) ** 2))
        return 2 * (predict - target) * E / d
    
class CrossEntropyLoss_ERM(Loss):
    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)
        if len(predict.shape) == 1:
            predict = predict.reshape(1, -1)
            target = target.reshape(1, -1)
        val = -np.einsum(
            "ij,ij->i", target, np.log(np.clip(predict, a_min=1e-10, a_max=None))
        ).reshape(-1, 1)
        return val

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Assume softmax is used, and target vector may or may not be one-hot encoding"""

        self._validate_shapes(predict, target)
        if len(predict.shape) == 1:
            predict = predict.reshape(1, -1)
            target = target.reshape(1, -1)

        # sum up target along rows, then multiply predict by this sum element wise,
        # then subtract target
        grad = np.einsum("ij,i->ij", predict, np.sum(target, axis=1)) - target

        return grad

class CrossEntropyLoss_TERM(Loss):
    def __init__(self, t: float = 1.0):
        super().__init__
        self.t = t

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        t = self.t
        self._validate_shapes(predict, target)
        N = len(predict)
        if len(predict.shape) == 1:
            predict = predict.reshape(1, -1)
            target = target.reshape(1, -1)
        val = -np.einsum(
            "ij,ij->i", target, np.log(np.clip(predict, a_min=1e-10, a_max=None))
        ).reshape(-1, 1)
        return 1/t * np.log(1/N * np.exp(t * val))

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        t = self.t
        self._validate_shapes(predict, target)
        N = len(predict)
        if len(predict.shape) == 1:
            predict = predict.reshape(1, -1)
            target = target.reshape(1, -1)
        val = -np.einsum(
            "ij,ij->i", target, np.log(np.clip(predict, a_min=1e-10, a_max=None))
        ).reshape(-1, 1)
        grad = np.einsum("ij,i->ij", predict, np.sum(target, axis=1)) - target
        E = np.exp(t * val)
        d = np.sum(E)
        return grad * E / d