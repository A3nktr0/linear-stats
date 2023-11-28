import sys
import numpy as np
from scipy.stats import pearsonr

def read_file():
    filename = sys.argv[1]
    file = open(filename).read().splitlines()

    value = []
    index = []

    for i, n in enumerate(file):
        index.append(i)
        value.append(int(n))
    
    return (index, value)


def coef(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    
    # Calcul cross deviation (x, y)
    cross_dev = np.sum(y*x) - n*m_y*m_x

    # Calcul deviation around x
    dev_x = np.sum(x*x) - n*m_x*m_x
    
    # Define regression coefficients (y = [a]x + [b])
    a = cross_dev / dev_x
    b = m_y - a*m_x
    
    return (a, b)

def main():
    index, value = read_file()
    x = np.array([index])
    y = np.array([value])
    
    a, b = coef(x,y)

    # Calcul PCC (Pearson's Correlation Coefficient)
    # [covariance(x, y) / sqrt(variance(x) * variance(y))]
    p, _ = pearsonr(index, value)
    
    print('Linear Regression Line: y = {:.6f}x + {:.6f}'.format(round(a, 6),round(b, 6) ))
    print('Pearson Correlation Coefficient: {:,.10f}'.format(round(p, 10)))

if __name__ == "__main__":
    main()
