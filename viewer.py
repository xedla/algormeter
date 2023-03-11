#!/usr/bin/env python3
# type: ignore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path
from algormeter.libs import *

levels = 30 
mpl.rc('figure', max_open_warning = 0)


def plotRect(f, data):
    c = f.optimumPoint
    s = f.XStart
    dMax = data.max(axis=0)
    dMin = data.min(axis=0)
    xMax = max(dMax[0],s[0],c[0])
    yMax = max(dMax[1],s[1],c[1])
    xMin = min(dMin[0],s[0],c[0])
    yMin = min(dMin[1],s[1],c[1])
    lx = (xMax - xMin)*1.2
    ly = (yMax - yMin)*1.2
    s = max(lx,ly)
    r = [[c[0]-s,c[1]-s],[c[0]+s,c[1]+s]]
    mM = [min(data.min(axis=0)[2],f.optimumValue),max(data.max(axis=0)[2],f.optimumValue),]
    return r , mM

def loadData(funcname):
    try:
        t = funcname.split(',')[1]
        q=eval(t)(dimension=2)
    except NameError:
        print(f"Oops!  {funcname} not found.  Try again...")
        exit(1)

    filename = './npy/' + funcname +'-2.npy'
    if os.path.exists(filename) :
        data = np.load(filename)
        x = data[:,0]
        y = data[:,1]
    else:
        raise ValueError('File not found')
    return q,data,x,y

def contour(funcname, rect = None):
    f, data, x, y = loadData(funcname)

    if rect is None:
        rect, mM = plotRect(f, data)
    xlist = np.linspace(rect[0][0], rect[1][0],levels)
    ylist = np.linspace(rect[0][1], rect[1][1],levels)
    X,Y = np.meshgrid(xlist, ylist)
    
    start  = abs(mM[0])
    ln = mM[1] - mM[0]
    levs = np.geomspace(.1,start+ln,levels) + mM[0]

    def g(x,y):
        return f.f(np.array([x,y]))
    h = np.frompyfunc(g, 2, 1)
    Z = h(X,Y)

    fig, ax = plt.subplots()
    plt.figure(figsize=(8, 8))
    contour = plt.contour(X, Y, Z, levels, colors='c');
    # contour = plt.contourf(X, Y, Z, 20)
    # plt.colorbar(contour)
    plt.clabel(contour, inline=True, fontsize=8)

    plt.title(funcname)
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')

    plt.plot(x, y, linewidth=1.0, color = 'b',marker="x",markersize=3)
    plt.plot(f.optimumPoint[0],f.optimumPoint[1], marker="+", markersize=7, markeredgecolor="red", markerfacecolor="red")
    plt.plot(f.XStart[0],f.XStart[1], marker="o", markersize=7, markeredgecolor="green", markerfacecolor="green")
    plt.plot(x[-1],y[-1], marker="x", markersize=10, markeredgecolor="b", markerfacecolor="g")
    return plt

def showFunc3D(funcname):
    q, data, x, y = loadData(funcname)
    r, __ = plotRect(q,data)

    xgrid = np.mgrid[r[0][0]:r[1][0]:0.1, r[0][1]:r[1][1]:0.1]
    xvec = xgrid.reshape(2, -1).T

    def loss(x, sign=1.):
        return sign * (q.f(x))

    F = np.vstack([loss(xi) for xi in xvec]).reshape(xgrid.shape[1:])

    ax = plt.axes(projection='3d')
    # ax.hold(True)
    ax.plot_surface(xgrid[0], xgrid[1], F, rstride=1, cstride=1,cmap=plt.cm.jet, shade=True, alpha=0.7, linewidth=0)
    ax.plot3D(data[:,0], data[:,1], data[:,2], 'or', mec='w', label='Path')
    ax.plot3D(q.XStart[0], q.XStart[1], q.f(q.XStart), 'og', mec='w', label='Start')
    ax.plot3D(q.optimumPoint[0], q.optimumPoint[1], q.f(q.optimumPoint), 'Xy', mec='w', label='Optimum')
    
    ax.legend(fancybox=True, numpoints=1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    
    plt.title(funcname)
    return plt

def descent(filename,dimension):
    y = np.load(f'./npy/{filename}-{dimension}.npy')[:,-1]
    x = np.arange(np.size(y))
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),ylim=(0, 8), yticks=np.arange(1, 8))
    plt.title(filename)
    plt.xlabel('Iteractions')
    plt.ylabel('delta')
    return plt

def batch():
    pics = './pics'
    npy = './npy'
    import pathlib
    pathlib.Path(pics).mkdir(parents=True, exist_ok=True) 
    import os
    for root, dirs, files in os.walk(npy):
        for name in files:
            if name.find('-2.') > 0:
                nm = name.replace('-2.npy','')
                inp = npy + '/' + name
                out = pics + '/' + nm + '.png'
                itime = os.path.getmtime(inp)

                out = f'{pics}/descent {nm}.png'
                otime = os.path.getmtime(out) if os.path.isfile(out) else itime
                if itime >= otime:
                    plt=descent(nm,2)
                    plt.savefig(out)
                    plt.close()
                    print(out)

                out = f'{pics}/contour {nm}.png'
                otime = os.path.getmtime(out) if os.path.isfile(out) else itime
                if itime >= otime:
                    plt=descent(nm,2)
                    plt=contour(nm)
                    plt.savefig(out)
                    plt.close()
                    print(out)

def center2rect(center, radius):
    return [[center[0]-radius,center[1]-radius],[center[0]+radius,center[1]+radius]]

if __name__ == "__main__":

    # if len(sys.argv) != 2:
    #     print(f"Usage: {sys.argv[0]} funcname")
    #     exit()
    # f = sys.argv[1]

    batch()
    exit()
    f = 'desasc4.desasc,JB05'
    
    # p=descent(f,2)
    # p = contour(f,center2rect(center=[0,0],radius= 10))
    p = contour(f)
    # p=showFunc3D(f)
    p.show()

