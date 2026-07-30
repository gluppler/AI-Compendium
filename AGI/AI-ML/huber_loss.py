def huber_loss(y_true, y_pred, delta):
    if len(y_true) != len(y_pred) or len(y_pred) == 0:
        return None
    loss = 0.0
    for t, p in zip(y_true, y_pred):
        r = abs(t - p)
        if r <= delta:
            loss += 0.5 * r ** 2
        else:
            loss += delta * r - 0.5 * delta ** 2
    return loss / len(y_pred)


if __name__ == "__main__":
    assert huber_loss([10.0, 8.0, 12.0], [9.0, 7.0, 11.0], 1.0) == 0.5
    assert abs(huber_loss([3.0, 5.0, 7.0], [2.0, 4.0, 8.0], 0.5) - 0.375) < 1e-10
    assert huber_loss([10.0, 8.0, 12.0], [7.0, 6.0], 1.0) is None
    assert huber_loss([10.0, 8.0, 12.0], [], 1.0) is None
    print("huber_loss: OK")
