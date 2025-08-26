import numpy as np

def cosine_sim(a, b, eps=1e-8):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0   # or skip / handle specially
    return np.dot(a, b) / (na * nb)