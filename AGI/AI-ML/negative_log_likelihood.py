import math

class NegativeLogLikelihoodLossError(Exception):
    pass

def neg_log_likelihood(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise NegativeLogLikelihoodLossError("InputsHaveDifferentLength")
    if len(y_pred) == 0:
        raise NegativeLogLikelihoodLossError("EmptyInputs")
    for v in y_true + y_pred:
        if v < 0.0 or v > 1.0:
            raise NegativeLogLikelihoodLossError("InvalidValues")

    total = 0.0
    for p, a in zip(y_pred, y_true):
        total += -a * math.log(p) - (1.0 - a) * math.log(1.0 - p)
    return total / len(y_pred)


if __name__ == "__main__":
    assert abs(neg_log_likelihood([1.0, 0.0, 1.0], [0.9, 0.1, 0.8]) - 0.14462152754328741) < 1e-10
    print("neg_log_likelihood: OK")
