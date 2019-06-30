import numpy as np
import scipy.stats as sps


def doLineFunc(orthog, p, x0, y0):
    mul = 1 if orthog else -1  # (x-x0)y/x + y0  or   -x(x-x0)/y + y0
    scale = mul * (p[0] ** -mul) * (p[1] ** mul)
    scaledX0fromY0 = scale * x0 + y0
    return lambda x: x * scale - scaledX0fromY0


def rotate(p, point):
    zoff = 25
    x0 = point[0] + zoff * p[0] / p[2]
    y0 = point[1] + zoff * p[1] / p[2]
    return x0, y0


def doComputeAssym(img, line_func, momentum, orthog, xx, yy, sumImg):
    idx = np.where(yy - line_func(xx) > 0) \
        if not orthog and momentum[1] < 0 \
        else np.where(yy - line_func(xx) < 0)
    imsize = img.shape[0]
    zz = np.ones((imsize, imsize))
    zz[idx] = 0
    return (np.sum(img * zz) - np.sum(img * (1 - zz))) / sumImg


def computeWidth(bb, rescaledX, x_, y_):
    sum0 = 0
    sum1 = 0
    sum2 = 0
    for i in range(100):
        ww = bb(x_[i], y_[i])
        if ww < 0: continue
        sum0 += ww
        rescaledXi = rescaledX[i]
        sum1 += rescaledXi * ww
        sum2 += rescaledXi * rescaledXi * ww
    sum1 = sum1 / sum0
    sum2 = sum2 / sum0
    if sum2 > sum1 * sum1:
        sigma = np.sqrt(sum2 - sum1 * sum1)
        spread = sigma[0]
    else:
        spread = 0
    return spread


def computeMsRatio2(alpha, img, sumImg):
    imsize = img.shape[0]
    ms_ = sumImg * alpha
    num = np.sum((img >= ms_))
    return num / imsize**2

#Distances between histograms
def chiSquare(observed, expected):
    return sps.chisquare(observed, expected)

def l2norm(observed, expected):
    return np.linalg.norm(observed - expected)

def l1norm(observed, expected):
    return np.linalg.norm(observed - expected, ord=1)