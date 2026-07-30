def hng_loss(y_true, y_pred):
    total = sum(max(0.0, 1.0 - a * p) for p, a in zip(y_pred, y_true))
    return total / len(y_pred)


if __name__ == "__main__":
    p = [-1.0, 1.0, 1.0]
    a = [-1.0, -1.0, 1.0]
    assert abs(hng_loss(a, p) - 0.6666666666666666) < 1e-10
    print("hinge_loss: OK")
