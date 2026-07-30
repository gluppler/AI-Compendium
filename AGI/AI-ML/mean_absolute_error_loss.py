def mae_loss(predicted, actual):
    total = sum(abs(p - a) for p, a in zip(predicted, actual))
    return total / len(predicted)


if __name__ == "__main__":
    p = [1.0, 2.0, 3.0, 4.0]
    a = [1.0, 3.0, 3.5, 4.5]
    assert mae_loss(p, a) == 0.5
    print("mae_loss: OK")
