class MarginalRankingLossError(Exception):
    pass

def average_margin_ranking_loss(x_first, x_second, margin, y_true):
    if len(x_first) != len(x_second):
        raise MarginalRankingLossError("InputsHaveDifferentLength")
    if len(x_first) == 0:
        raise MarginalRankingLossError("EmptyInputs")
    if margin < 0.0:
        raise MarginalRankingLossError("NegativeMargin")
    if y_true not in (1.0, -1.0):
        raise MarginalRankingLossError("InvalidValues")

    total = sum(max(0.0, margin - y_true * (f - s)) for f, s in zip(x_first, x_second))
    return total / len(x_first)


if __name__ == "__main__":
    result = average_margin_ranking_loss([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], 1.0, -1.0)
    assert result == 0.0

    result = average_margin_ranking_loss([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], 1.0, 1.0)
    assert result == 2.0

    print("margin_ranking_loss: OK")
