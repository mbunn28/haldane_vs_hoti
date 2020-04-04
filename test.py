import scipy.optimize as opt

def f(x):
    return (x-1)

result = None
k = -1
while result is None:
    try:
        result = opt.brentq(f,-2,k)
    except ValueError:
        k = k + 0.05
        pass

print(k)
print(result)