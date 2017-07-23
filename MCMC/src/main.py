import numpy as np


alphabet = np.loadtxt('../alphabet.csv', delimiter=',', dtype=np.str)
cipher_func = np.loadtxt('../cipher_function.csv', delimiter=',', dtype=np.str)
P = np.loadtxt('../letter_probabilities.csv', delimiter=',')
M = np.loadtxt('../letter_transition_matrix.csv', delimiter=',')
with open('../ciphertext.txt', 'r') as f:
    cipher_text = f.read()
with open('../plaintext.txt', 'r') as f:
    plain_text = f.read()


def process_text(text):
    ind = []
    for l in text:
        z = np.where(alphabet == l)[0][0]
        ind.append(z)
    return np.array(ind)


def Pinv(f):
    inv = np.arange(f.size)
    # TODO:: make inverse with numpy functionality
    for i in range(f.size):
        inv[f[i]] = i
    return inv


def logPyf(y, f):
    inv = Pinv(f)
    return np.log(P[inv[y[0]]]) + np.sum(np.log(M[inv[y[1:]], inv[y[:-1]]]))


def V(f):
    g = np.copy(f)
    indx = np.random.randint(0, f.size, size=2)
    if indx[0] == indx[1]: indx[0] = (indx[0] + 1) % f.size
    g[indx[0]], g[indx[1]] = f[indx[1]], f[indx[0]]
    return g


def MH(y, steps=100):
    bag = []
    f = f0(y)  #process_text(cipher_func) #np.random.permutation(alphabet.size)
    best_f = None
    best_p = -np.inf
    for _ in range(steps):
        tp = logPyf(y, f)
        if best_p < tp:
            best_p = tp
            best_f = f

        fp = V(f)

        log_diff = logPyf(y, fp) - logPyf(y, f)
        if np.isnan(log_diff): log_diff = -np.inf
        alpha = np.exp(np.minimum(0, log_diff))
        if np.random.rand() < alpha:
            f = fp
        bag.append(f)
    print best_f
    return best_f


def f0(y):
    for _ in range(1000):
        f = np.random.permutation(alphabet.size)
        fp = V(f)
        log_diff = logPyf(y, fp) - logPyf(y, f)
        if not np.isnan(log_diff):
            return f


def main():
    text = process_text(cipher_text)
    orig = process_text(plain_text)
    MH(text, steps=100000)
    print process_text(cipher_func)


if __name__ == '__main__':
    main()
