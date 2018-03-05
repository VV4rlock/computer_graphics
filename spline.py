import bresenham
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

__color=np.array([200,100,254], dtype=np.uint8)
__height=500
__width=500




image=bresenham.teapop.create_2d_rgb_arr(__height,__width)
A=np.array([100,200])
B=np.array([400,400])

def spline(A, B, image):
    M = np.array([
                  [1, A[0], A[0]*A[0], A[0]*A[0]*A[0]],
                  [1, B[0], B[0]*B[0], B[0]*B[0]*B[0]],
                  [0, 1, 2*A[0], 3*A[0]*A[0]],
                  [0, 1, 2*B[0], 3*B[0]*B[0]]
                ])

    Y = np.array([A[1], B[1], rnd.randint(0,5), rnd.randint(0,5)])
    a = np.linalg.inv(M).dot(Y)


    k=int(a[0]+A[0]*a[1]+A[0]*A[0]*a[2]+A[0]*A[0]*A[0]*a[3])
    for i in range(A[0]+1, B[0]+1):
        y = int(a[0]+i*a[1]+i*i*a[2]+i*i*i*a[3])
        try:
            bresenham.teapop.bresenham_line(image,__color,i-1,k,i,y)
            k=y
            #image[-y, i, :] = __color
        except:
            pass

def draw_point(A,color):
    try:
        image[-A[1], A[0], :] = color
        image[-A[1]+1, A[0], :] = color
        image[-A[1]-1, A[0], :] = color
        image[-A[1], A[0]+1, :] = color
        image[-A[1], A[0]-1, :] = color
    except:
        pass


point = sorted([(rnd.randint(100, __width-100), rnd.randint(100, __height-100)) for _ in range(10)],
               key=lambda x: x[0])


for i in range(len(point)-1):
    spline(point[i], point[i+1], image)
    draw_point(point[i],np.array([100,255,0]))
draw_point(point[-1],np.array([100,255,0]))

plt.figure()
plt.imshow(image)
plt.show()