def mse_loss(predicted, actual):
    total = sum((p - a) ** 2 for p, a in zip(predicted, actual))
    return total / len(predicted)


if __name__ == "__main__":
    p = [1.0, 2.0, 3.0, 4.0]
    a = [1.0, 3.0, 3.5, 4.5]
    assert mse_loss(p, a) == 0.375
    print("mse_loss: OK")
