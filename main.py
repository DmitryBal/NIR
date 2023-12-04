import numpy as np
from scipy.interpolate import UnivariateSpline
import sympy as sp
import math as m
import matplotlib.pyplot as plt


sign = lambda x: m.copysign(1, x)


def f(y):
    return sp.expand(-y*y)


def _f(x):
    return 1.0/(1.0 + x)


def f2(u, v, mu=1.0):
    return sp.expand(-u + mu*(1.0 - u*u)*v)


def CA(arr, index, _h):
    c = arr[index]
    s = sign(c)
    abs_c = abs(c)
    mod = abs_c % _h
    if mod == 0.0:
        arr[index] = s*_h
    else:
        arr[index] = s*mod
    div = abs_c//_h
    arr[index - 1] += s*div
    return arr


def no_rank(y0, h, m):
    Y_tay = np.zeros(m+1)
    Y_tay2 = np.zeros(m + 1)
    Y_tay3 = np.zeros(m + 1)
    Y_tay[0] = y0
    Y_tay2[0] = y0
    Y_tay3[0] = y0
    for i in range(m):
        n = i+1
        Y_tay[i+1] = 1.0 - n*h
        Y_tay2[i+1] = Y_tay[i+1] + n * (n - 1) * h ** 2
        Y_tay3[i+1] = Y_tay2[i+1] - 0.5*n*(2.0*n**2 - 5.0*n + 3.0)*pow(h, 3)
    return Y_tay, Y_tay2, Y_tay3


def test2(y0, h, m):
    Y = np.zeros(m+1)
    Y[0] = y0
    for i in range(m):
        Y[i+1] = Y[i] + h*f(Y[i])
    return Y


def test_func(_x, t, l, hh):
    y = sp.series(_x, t, 0, l + 1).removeO()
    a = [y.coeff(t, i) for i in range(l + 1)]
    for j in range(l):
        ind = -(j + 1)
        if abs(a[ind]) > hh:
            a = CA(a, ind, hh)
    y = 0.0
    if a[0] > hh:
        a[0] = sign(a[0]) * (abs(a[0]) % hh)
    for k in range(l + 1):
        y = sp.expand(y + a[k] * t ** k)
    return a, y


def lorenc(x, y, z, h, m, l):
    hh = 1.0 / h
    sigma = 3.0
    r = 15.0
    v = 1.0
    print('h^-1', hh)
    X = np.zeros(m + 1)
    X[0] = x
    Y = np.zeros(m + 1)
    Y[0] = y
    Z = np.zeros(m + 1)
    Z[0] = z
    tay = np.zeros(l + 1)
    tay[0] = 1.0
    for j in range(1, l + 1):
        tay[j] = h * tay[j - 1]
    t = sp.symbols('t')
    A_x = []
    A_x.append([3.0, 2.0, 15.0])
    for i in range(m):
        print(f'N = {i}')
        _x = sp.expand(x + sigma * (y - x) * t)
        _y = sp.expand(y + (x * (r - z) - y) * t)
        _z = sp.expand(z + (x * y - v * z) * t)

        a_x, x = test_func(_x, t, l, hh)
        a_y, y = test_func(_y, t, l, hh)
        a_z, z = test_func(_z, t, l, hh)

        X[i+1] = sum(tay * a_x)
        Y[i+1] = sum(tay * a_y)
        Z[i+1] = sum(tay * a_z)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(x, y, z)
        print(a_x, a_y, a_z)
        print('------------------------------------')
        A_x.append([a_x, a_y, a_z])
    return X, Y, Z, A_x


def vanderpol(u, v, h, m, l):
    hh = 1.0 / h
    print('h^-1', hh)
    U = np.zeros(m + 1)
    U[0] = u
    V = np.zeros(m + 1)
    V[0] = v
    tay = np.zeros(l + 1)
    tay[0] = 1.0
    for j in range(1, l + 1):
        tay[j] = h * tay[j - 1]
    t = sp.symbols('t')
    for i in range(m):
        print(f'N = {i}')
        u = sp.expand(u + v*t)
        u = sp.series(u, t, 0, l + 1).removeO()
        v = sp.expand(v + f2(u, v)*t)
        v = sp.series(v, t, 0, l + 1).removeO()
        a_u = [u.coeff(t, i) for i in range(l + 1)]
        a_v = [v.coeff(t, i) for i in range(l + 1)]
        print(u, v)
        for j in range(l):
            ind = -(j + 1)
            if abs(a_u[ind]) > hh:
                a_u = CA(a_u, ind, hh)
            if abs(a_v[ind]) > hh:
                a_v = CA(a_v, ind, hh)
        u = 0.0
        v = 0.0
        a_v[0] = sign(a_v[0]) * (abs(a_v[0]) % hh)
        a_u[0] = sign(a_u[0]) * (abs(a_u[0]) % hh)
        for k in range(l + 1):
            u = sp.expand(u + a_u[k] * t ** k)
            v = sp.expand(v + a_v[k] * t ** k)
        U[i + 1] = sum(tay * a_u)
        V[i + 1] = sum(tay * a_v)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(u, v)
        print('------------------------------------')
    return U, V


def eiler(y0, h, m, l):
    hh = 1.0 / h
    Y = np.zeros(m + 1)
    Y[0] = y0
    tay = np.zeros(l+1)
    tay[0] = 1.0
    for j in range(1, l+1):
        tay[j] = h*tay[j-1]
    t = sp.symbols('t')
    y = y0
    for i in range(m):
        y = sp.expand(y + f(y) * t)
        y = sp.series(y, t, 0, l+1).removeO()
        a = [y.coeff(t, i) for i in range(l+1)]
        print(y)
        for j in range(l):
            ind = -(j + 1)
            if abs(a[ind]) > hh:
                a = CA(a, ind, hh)
        y = 0.0
        if a[0] > hh:
            a[0] = sign(a[0]) * (abs(a[0]) % hh)
        for k in range(l+1):
            y = sp.expand(y+a[k]*t**k)
        Y[i + 1] = sum(tay*a)
    return Y


def plot(x, y, y1, y2, y3, _y):
    plt.plot(x, y, label='exact')
    plt.scatter(x, y1, label='CA 3', color='red')
    plt.scatter(x, y2, label='CA 5', color='green')
    plt.scatter(x, y3, label='CA 8', color='gray')
    plt.plot(x, _y, label='Eiler')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_values(x, y, z, flag, sigma=3.0, rho=15.0, v=1.0, time=0.01, mu = 1.0):
    if flag:
        x_point = time * (sigma * (y - x))
        y_point = time * (x * (rho - z) - y)
        z_point = time * ((x * y) - (v * z))
    else:
        x_point = time * y
        y_point = time * (-x - mu*(1.0 - x*x)*v)
    return x_point, y_point, z_point


# Points is number of points graphed
def generate_points(points):
    xs, ys, zs = [3.0], [2.0], [15.0]
    for i in range(points):
        new_x, new_y, new_z = get_values(xs[i], ys[i], zs[i], True)
        xs.append(xs[i] + new_x)
        ys.append(ys[i] + new_y)
        zs.append(zs[i] + new_z)
    return xs, ys, zs


def main(xs, ys, zs, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    plt.axis('on')
    ax.plot(xs, ys, zs)
    ax.scatter(x, y, z)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.show()


def main_eiler():
    x_0 = 0.0
    y_0 = 1.0
    h = 0.1
    x_n = 1.0
    n = int((x_n - x_0) / h)
    X = np.linspace(x_0, x_n, n+1)
    Y = [y_0]
    for x in X[1::]:
        Y.append(_f(x))
    _Y = test2(y_0, h, n)
    _y1 = eiler(y_0, h, n, 3)
    _y2 = eiler(y_0, h, n, 5)
    _y3 = eiler(y_0, h, n, 8)
    Y_test1, Y_test2, Y_test3 = no_rank(y_0, h, n)
    plot(X, Y, _y1, _y2, _y3, _Y)


def plot_vp(v, u, v1, u1):
    plt.plot(v, u, label='p = 1')
    plt.plot(v1, u1, label='p = 4')
    plt.legend()
    plt.xlabel('V')
    plt.ylabel('U')
    plt.grid(True)
    plt.show()


def main_vanderpole():
    x_0 = 0.0
    u_0 = 1.0
    v_0 = 1.0
    h = 0.1
    x_n = 1.0
    n = int((x_n - x_0) / h + 1)
    _U, _V = vanderpol(u_0, v_0, h, n, 1)
    _U1, _V1 = vanderpol(u_0, v_0, h, n, 4)
    plot_vp(_V, _U, _V1, _U1)


def main_lorenc():
    time_0 = 0.0
    x_0 = 3.0
    y_0 = 2.0
    z_0 = 15.0
    time_n = 10.0
    _h = 0.01
    n = int((time_n - time_0) / _h)
    _X, _Y, _Z, _A= lorenc(x_0, y_0, z_0, _h, n, 1)
    return _X, _Y, _Z, _A, n



if __name__ == "__main__":
   # main_eiler()
    #main_vanderpole()
    ca_x, ca_y, ca_z, A,  N = main_lorenc()


    # Начальное состояние и временной интервал
    initial_state = [3.0, 2.0, 15.0]
    t = np.arange(0.0, 10.01, 0.01)
    xs, ys, zs = generate_points(N)

    # Создание сплайна
    spl1 = UnivariateSpline(t, ca_x)
    spl2 = UnivariateSpline(t, ca_y)
    spl3 = UnivariateSpline(t, ca_z)

    new_ax = []
    time_ = []
    new_ay = []
    new_az = []
    i = 1
    for i in range(2, len(ca_x)):
        if (np.round(ca_x[i] - ca_x[i-1], 12) != np.round(ca_x[i-1] - ca_x[i-2], 12) or np.round(ca_y[i] - ca_y[i-1], 12) != np.round(ca_y[i-1] - ca_y[i-2], 12) or np.round(ca_z[i] - ca_z[i - 1], 12) != np.round(ca_z[i - 1] - ca_z[i - 2], 12)):
            new_ax.append(ca_x[i-1])
            new_ay.append(ca_y[i - 1])
            new_az.append(ca_z[i - 1])
            time_.append(t[i-1])
            '''
        if ():
            new_ay.append(ca_y[i-1])
            time_y.append(t[i - 1])
        if ():
            new_az.append(ca_z[i - 1])
            time_z.append(t[i - 1])
        '''
        i += 1
    main(ca_x, ca_y, ca_z, new_ax, new_ay, new_az)
    plt.figure(figsize=(10, 7))
    #plt.plot(t, xs, label='exact x(t)')
    plt.plot(t, ca_x, label='CA x(t)')
    plt.scatter(time_, new_ax, color = 'g')
    print(ca_x)
    #plt.plot(t, spl1(t), 'g', lw=3, label='quadratic spline')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.grid(True)
    plt.show()
    #plt.plot(t, ys, label='exact y(t)')
    plt.plot(t, ca_y, label='CA y(t)')
    plt.scatter(time_, new_ay, color='g')
    print(ca_y)
    #plt.plot(t, spl2(t), 'g', lw=3, label='quadratic spline')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.grid(True)
    plt.show()
    #plt.plot(t, zs, label='exact z(t)')
    plt.plot(t, ca_z, label='CA z(t)')
    plt.scatter(time_, new_az, color='g')
    print(ca_z)
    #plt.plot(t, spl3(t), 'g', lw=3, label='quadratic spline')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.grid(True)
    plt.show()



'''
import numpy as np
from scipy import integrate
from scipy.interpolate import UnivariateSpline
import sympy as sp
import math as m
import matplotlib.pyplot as plt


sign = lambda x: m.copysign(1, x)


def f(y):
    return sp.expand(-y*y)


def _f(x):
    return 1.0/(1.0 + x)


def f2(u, v, mu=1.0):
    return sp.expand(-u + mu*(1.0 - u*u)*v)


def lorenz(p, t):
    x,y,z = p.tolist()
    # Текущее положение безмассовой точки (x, y, z)
    print("x,y,z,t:", x, y, z, t)
    return y, x - 0.81 * y - x * z, (-0.375 * z + x * x) # Вернуться к dx / dt, dy / dt, dz / dt





def CA(arr, index, _h):
    c = arr[index]
    s = sign(c)
    abs_c = abs(c)
    mod = abs_c % _h
    if mod == 0.0:
        arr[index] = s*_h
    else:
        arr[index] = s*mod
    div = abs_c//_h
    arr[index - 1] += s*div
    return arr


def no_rank(y0, h, m):
    Y_tay = np.zeros(m+1)
    Y_tay2 = np.zeros(m + 1)
    Y_tay3 = np.zeros(m + 1)
    Y_tay[0] = y0
    Y_tay2[0] = y0
    Y_tay3[0] = y0
    for i in range(m):
        n = i+1
        Y_tay[i+1] = 1.0 - n*h
        Y_tay2[i+1] = Y_tay[i+1] + n * (n - 1) * h ** 2
        Y_tay3[i+1] = Y_tay2[i+1] - 0.5*n*(2.0*n**2 - 5.0*n + 3.0)*pow(h, 3)
    return Y_tay, Y_tay2, Y_tay3


def test2(y0, h, m):
    Y = np.zeros(m+1)
    Y[0] = y0
    for i in range(m):
        Y[i+1] = Y[i] + h*f(Y[i])
    return Y


def test_func(_x, t, l, hh):
    y = sp.series(_x, t, 0, l + 1).removeO()
    a = [y.coeff(t, i) for i in range(l + 1)]
    for j in range(l):
        ind = -(j + 1)
        if abs(a[ind]) > hh:
            a = CA(a, ind, hh)
    y = 0.0
    if a[0] > hh:
        a[0] = sign(a[0]) * (abs(a[0]) % hh)
    for k in range(l + 1):
        y = sp.expand(y + a[k] * t ** k)
    return a, y


def lorenc(x, y, z, h, m, l):
    hh = 1.0 / h
    sigma = 3.0
    r = 15.0
    v = 1.0
    print('h^-1', hh)
    X = np.zeros(m + 1)
    X[0] = x
    Y = np.zeros(m + 1)
    Y[0] = y
    Z = np.zeros(m + 1)
    Z[0] = z
    tay = np.zeros(l + 1)
    tay[0] = 1.0
    for j in range(1, l + 1):
        tay[j] = h * tay[j - 1]
    t = sp.symbols('t')
    for i in range(m):
        print(f'N = {i}')
        _x = sp.expand(x + sigma * (y - x) * t)
        _y = sp.expand(y + (x * (r - z) - y) * t)
        _z = sp.expand(z + (x * y - v * z) * t)

        a_x, x = test_func(_x, t, l, hh)
        a_y, y = test_func(_y, t, l, hh)
        a_z, z = test_func(_z, t, l, hh)

        X[i+1] = sum(tay * a_x)
        Y[i+1] = sum(tay * a_y)
        Z[i+1] = sum(tay * a_z)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(u, v)
        print('------------------------------------')
    return X, Y, Z


def vanderpol(u, v, h, m, l):
    hh = 1.0 / h
    print('h^-1', hh)
    U = np.zeros(m + 1)
    U[0] = u
    V = np.zeros(m + 1)
    V[0] = v
    tay = np.zeros(l + 1)
    tay[0] = 1.0
    for j in range(1, l + 1):
        tay[j] = h * tay[j - 1]
    t = sp.symbols('t')
    for i in range(m):
        print(f'N = {i}')
        u = sp.expand(u + v*t)
        u = sp.series(u, t, 0, l + 1).removeO()
        v = sp.expand(v + f2(u, v)*t)
        v = sp.series(v, t, 0, l + 1).removeO()
        a_u = [u.coeff(t, i) for i in range(l + 1)]
        a_v = [v.coeff(t, i) for i in range(l + 1)]
        print(u, v)
        for j in range(l):
            ind = -(j + 1)
            if abs(a_u[ind]) > hh:
                a_u = CA(a_u, ind, hh)
            if abs(a_v[ind]) > hh:
                a_v = CA(a_v, ind, hh)
        u = 0.0
        v = 0.0
        a_v[0] = sign(a_v[0]) * (abs(a_v[0]) % hh)
        a_u[0] = sign(a_u[0]) * (abs(a_u[0]) % hh)
        for k in range(l + 1):
            u = sp.expand(u + a_u[k] * t ** k)
            v = sp.expand(v + a_v[k] * t ** k)
        U[i + 1] = sum(tay * a_u)
        V[i + 1] = sum(tay * a_v)
        #U[i+1] = (U[i] + V[i]*h)*x[i]
        #V[i+1] = (V[i] + f2(U[i], V[i])*h)*x[i]
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(u, v)
        print('------------------------------------')
    return U, V


def eiler(y0, h, m, l):
    hh = 1.0 / h
    Y = np.zeros(m + 1)
    Y[0] = y0
    tay = np.zeros(l+1)
    tay[0] = 1.0
    for j in range(1, l+1):
        tay[j] = h*tay[j-1]
    t = sp.symbols('t')
    y = y0
    for i in range(m):
        y = sp.expand(y + f(y) * t)
        y = sp.series(y, t, 0, l+1).removeO()
        a = [y.coeff(t, i) for i in range(l+1)]
        print(y)
        for j in range(l):
            ind = -(j + 1)
            if abs(a[ind]) > hh:
                a = CA(a, ind, hh)

        y = 0.0
        if a[0] > hh:
            a[0] = sign(a[0]) * (abs(a[0]) % hh)
        for k in range(l+1):
            y = sp.expand(y+a[k]*t**k)
        Y[i + 1] = sum(tay*a)
        print(y)
    return Y


def plot(x, y, y1, y2, y3, _y):
    plt.plot(x, y, label='exact')
    plt.scatter(x, y1, label='CA 3', color='red')
    plt.scatter(x, y2, label='CA 5', color='green')
    plt.scatter(x, y3, label='CA 8', color='gray')
    plt.plot(x, _y, label='Eiler')
    plt.legend()
    plt.grid(True)
    plt.show()


# Change the values of sigma, rho, beta, and/or time in the default param value or function
# call to manipulate the appearance of the Lorenz Attractor
def get_values(x, y, z, flag, sigma=3.0, rho=15.0, v=1.0, time=0.01, mu = 0.81, alpha = 0.375):
    if flag:
        x_point = time * (sigma * (y - x))
        y_point = time * (x * (rho - z) - y)
        z_point = time * ((x * y) - (v * z))
    else:
        x_point = time * y
        y_point = time * (x - mu*y - x*z)
        z_point = time * (-alpha*z + x*x)
    return x_point, y_point, z_point


# Points is number of points graphed
def generate_points(points=400):
    xs, ys, zs = [15.0], [2.0], [3.0]
    for i in range(points):
        new_x, new_y, new_z = get_values(xs[i], ys[i], zs[i], False)
        xs.append(xs[i] + new_x)
        ys.append(ys[i] + new_y)
        zs.append(zs[i] + new_z)
    return xs, ys, zs


def main(xs, ys, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    plt.axis('on')
    ax.plot(xs, ys, zs)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.show()


def main_eiler():
    x_0 = 0.0
    y_0 = 1.0
    h = 0.1
    x_n = 1.0
    n = int((x_n - x_0) / h)
    X = np.linspace(x_0, x_n, n+1)
    Y = [y_0]
    for x in X[1::]:
        Y.append(_f(x))
    _Y = test2(y_0, h, n)
    _y1 = eiler(y_0, h, n, 3)
    _y2 = eiler(y_0, h, n, 5)
    _y3 = eiler(y_0, h, n, 8)
    Y_test1, Y_test2, Y_test3 = no_rank(y_0, h, n)
    plot(X, Y, _y1, _y2, _y3, _Y)


def plot_vp(v, u, v1, u1):
    plt.plot(v, u, label='p = 1')
    plt.plot(v1, u1, label='p = 4')
    plt.legend()
    plt.xlabel('V')
    plt.ylabel('U')
    plt.grid(True)
    plt.show()


def main_vanderpole():
    x_0 = 0.0
    u_0 = 1.0
    v_0 = 1.0
    h = 0.1
    x_n = 20.0
    n = int((x_n - x_0) / h + 1)
    _U, _V = vanderpol(u_0, v_0, h, n, 1)
    _U1, _V1 = vanderpol(u_0, v_0, h, n, 4)
    plot_vp(_V, _U, _V1, _U1)


def main_lorenc():
    time_0 = 0.0
    x_0 = 3.0
    y_0 = 2.0
    z_0 = 15.0
    time_n = 1.0
    _h = 0.01
    n = int((time_n - time_0) / _h)
    _X, _Y, _Z = lorenc(x_0, y_0, z_0, _h, n, 4)
    return _X, _Y, _Z, n


def SM_test(time, ca, no_ca, exact):
    plt.figure(figsize=(10, 7))
    plt.plot(time, exact, label='exact x(t)')
    plt.plot(time, no_ca, label='no CA x(t)')
    plt.plot(time, ca, label='CA x(t)', color='g')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.grid(True)
    plt.show()
    count_t = 0
    count_f = 0
    count = 0
    for i in range(N):
        a = exact[i] - ca[i]
        b = exact[i] - no_ca[i]
        if abs(a) > abs(b):
            TF = False
            count_f += 1
        elif abs(a) == abs(b):
            TF = 'equel'
            count += 1
        else:
            TF = True
            count_t += 1
        print(i, exact[i], a, b, TF)
    print('true:', count_t, '==:', count, 'false:', count_f)


if __name__ == "__main__":
    #main_eiler()
    #main_vanderpole()
    ca_x, ca_y, ca_z, N = main_lorenc()
    main(ca_x, ca_y, ca_z)
    x, y, z = generate_points()
    main(x, y, z)
    t = np.arange(0, 1.01, 0.01)
    track1 = integrate.odeint(lorenz, (3.0, 2.0, 15.0), t)
    print("type(track1):", type(track1), "track1.shape:", track1.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    plt.axis('on')
    ax.plot(track1[:, 0], track1[:, 1], track1[:, 2], lw=1.0, color='r')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.show()
    SM_test(t, ca_x, x, track1[:, 0])
    SM_test(t, ca_y, y, track1[:, 1])
    SM_test(t, ca_z, z, track1[:, 2])

    
    # Начальное состояние и временной интервал
    initial_state = [0.375, 0.0, 1.0]
    t = np.arange(0.0, 10.01, 0.01)
    xs, ys, zs = generate_points(N)

    # Создание сплайна
    spl1 = UnivariateSpline(t, ca_x)
    spl2 = UnivariateSpline(t, ca_y)
    spl3 = UnivariateSpline(t, ca_z)

    plt.figure(figsize=(10, 7))
    plt.plot(t, xs, label='exact x(t)')
    plt.plot(t, ca_x, label='CA x(t)')
    #plt.plot(t, spl1(t), 'g', lw=3, label='quadratic spline')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.grid(True)
    plt.show()
    plt.plot(t, ys, label='exact y(t)')
    plt.plot(t, ca_y, label='CA y(t)')
    #plt.plot(t, spl2(t), 'g', lw=3, label='quadratic spline')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.grid(True)
    plt.show()
    plt.plot(t, zs, label='exact z(t)')
    plt.plot(t, ca_z, label='CA z(t)')
    #plt.plot(t, spl3(t), 'g', lw=3, label='quadratic spline')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.grid(True)
    plt.show()
    

'''