import numpy as np
import math
import sys

def read_file():
    filename = sys.argv[1]
    file = open(filename).read().splitlines()
    value = []
    index = []
    for i, n in enumerate(file):
        index.append(i)
        value.append(int(n))
    return (index, value)


def calc_variance(data_set, mean):
    k = 0
    n = 0
    for value in data_set:
        k = (value-mean)**2
        n += k
    return (1/len(data_set))*n


def calc_covariance(data_x, data_y, m_x, m_y):
    k = 0
    for i in range(len(data_x)):
        k += (data_x[i]-m_x)*(data_y[i]-m_y)
    return (1/len(data_x))*k

# Calcul pearson correlation coefficient
def pcc(cov, v_x, v_y):
    return cov / (math.sqrt(v_x * v_y))

def calc_coef(x, y, m_x, m_y):

    n = np.size(x)

    # calcul cross deviation
    cross_dev = np.sum(y*x) - n*m_y*m_x
    
    # calcul deviation around x
    dev_x = np.sum(x*x)-n*m_x*m_x

    # define regression coeff
    a = cross_dev/dev_x
    b = m_y - a*m_x
    
    return (a, b)


def main():
    index, value = read_file()
    x = np.array([index])
    y = np.array([value])

    n = np.size(x)
    m_x = np.sum(index)/n
    m_y = np.sum(value)/n

    a, b = calc_coef(x, y, m_x, m_y)

    cov = calc_covariance(x, y, m_x, m_y)
    var_x = calc_variance(x, m_x)
    var_y = calc_variance(y, m_y)

    p = pcc(cov, m_x, m_y)

    print('Linear Regression Line: y = {:,.6f}x + {:,.6f}'.format(round(a,6), round(b, 6)))
    print('Pearson Correlation Coefficient: {:,.10f}'.format(round(p, 10)))

if __name__ == '__main__':
    main()
