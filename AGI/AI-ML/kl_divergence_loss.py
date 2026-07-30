import math

def kld_loss(actual, predicted):
    eps = 0.00001
    loss = 0.0
    for a, p in zip(actual, predicted):
        loss += (a + eps) * math.log((a + eps) / (p + eps))
    return loss


if __name__ == "__main__":
    a = [1.346112, 1.337432, 1.246655]
    p = [1.033836, 1.082015, 1.117323]
    assert abs(kld_loss(a, p) - 0.7752789394328498) < 1e-10
    print("kl_divergence_loss: OK")
