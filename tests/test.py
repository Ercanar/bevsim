#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from functools import cache
import lmaofit
import sys

xs = np.arange(10)

main = plt.gcf().subfigures(2, 1)
top  = main[0].subplots(1, 2)
bot  = main[1].subplots(1, 1)
help(bot)

top[0].scatter(xs, xs)
top[1].scatter(xs, xs)
bot.scatter(xs, xs)

plt.show()

exit(0)

def lazy(f):
    return lambda *args, **kwargs: lambda _: f(*args, **kwargs)

@lazy
def foo(bar):
    print(f"foo called with {bar}")

print(foo(42)(()))

exit(0)

pds = np.array([])
pys = np.array([])

for pd in np.arange(0.00000, 0.05001, 0.00005):
#for pd in [0.006232393]:
    @cache
    def step(n):
        return 0 if n <= 0 else pd + (1 - pd) * step(n - 1)

    xs = list(range(361))
    ys = np.array([step(n) for n in xs])

    plt.plot(xs, ys, label = str(pd))
    pds = np.append(pds, pd)
    pys = np.append(pys, step(361))
#plt.legend()
# plt.show()
plt.clf()

def f(x):
    return 250 * x / (90000 - 49887 * x)
    #return x

# print(pds, pys)
ix = filter(lambda x: x[1] >= 0.75, enumerate(pys)).__next__()[0]
iz = filter(lambda x: x[1] >= 0.998, enumerate(pys)).__next__()[0]
newPy = pys[ix:iz]
newPd = pds[ix:iz]
print(len(newPd), len(newPy))

@lmaofit.wrap
def fitfunc(x, y, a, b, c):
    res = a * np.log(x + b) + c - y
    res = list(map(lambda x: 1e6 if np.isnan(x) else x, res))
    # print(res)
    return res

res = lmaofit.fit(fitfunc, newPd, newPy,)


lmaofit.plot(fitfunc, res, newPd, color = "black")
#plt.plot(pds)
#plt.plot(pds, f(pys) - pys)
#plt.plot(pds, pds * 360 - pys)
#plt.show()

plt.plot(pds, pys)
plt.plot(f(pys), pys)
#plt.plot(pds, np.ones(1001))
#plt.show()


def p_d(p_y):
    if p_y == 0:
        return 0

    if p_y <= 0.75:
        return 250 * p_y / (90000 - 49887 * p_y)
    elif p_y >= 0.998:
        return 0.05
    else:
        return np.exp( 20 / 721 * (500 * p_y - 663)) + 181 / 50000


plt.plot(list(map(p_d, pys)), pys)
# plt.plot(np.vectorize(p_d)(pys), pys)
plt.xlabel("p_d")
plt.ylabel("p_y")
plt.show()

#plt.plot(pds, f(pys), label = "my py")
#plt.plot(pds, pds, color = "red", label = "real")
#plt.legend()
#plt.show()

# f(x+2) = f(x) + f(x+1)
# In [9]: golden = (1+sqrt(5))/2
#
# In [10]: silver = 1 - golden
#
# In [11]: def fib(x):
#     ...:     return (golden ** x - silver ** x)/(sqrt(5))

# class Util:
#     @staticmethod
#     def do_shit(arr):
#         # arr *= 2
#         arr = arr * 2
#         print(arr)
#
# class Foo:
#     def __init__(self):
#         self.bar = np.array([1, 2, 3])
#
# foo = Foo()
# print(foo.bar)
# Util.do_shit(foo.bar)
# print(foo.bar)

#plt.ion()
#
#fig, ax = plt.subplots(1, 1)
#
#_manager = plt.get_current_fig_manager()
#assert _manager != None
#_canvas = _manager.canvas
#
#def update():
#    _canvas.flush_events()
#
#while True:
#    print("here")
#    x = range(10)
#    y = np.random.rand(10)
#    ax.scatter(x, y)
#    update()
