def relu(array):
    for i in range(len(array)):
        if array[i] <= 0:
            array[i] = 0.0
    return array


if __name__ == "__main__":
    test = [1.0, 0.5, -1.0, 0.0, 0.3]
    result = relu(test[:])
    assert result == [1.0, 0.5, 0.0, 0.0, 0.3]
    print("relu: OK")
