import numpy as np


def read_result(path):
    result = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            value = float(line)
            result.append(value)
    return np.array(result)


def compare():
    result1 = read_result('final.txt')
    # result1 = read_result('output/i29w1d1n1/result.txt')
    # result2 = read_result('output/i29w8d2n1/result.txt')
    # result2 = read_result('output/i29w32d2v3/result.txt')
    # result2 = read_result('predict.txt')
    delt = np.abs(result1 - result2)
    mean = np.mean(delt)
    m = np.max(delt)
    print('mean', mean, 'max', m)


if __name__ == '__main__':
    compare()

