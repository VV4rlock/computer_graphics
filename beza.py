import bresenham
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random as rnd

__color=np.array([200,100,254], dtype=np.uint8)
__step=0.2
__height=500
__width=500
__filename="teapot.obj"
__name_of_save_image="img.png"


def beza(P0, P1, P2, P3, im):
    step=0.05
    length=np.arange(0,1,step)
    y=P0
    for t in length:
        p=(P0*(1-t)**3+P1*3*t*(1-t)**2+P2*3*t*t*(1-t)+P3*t**3).astype(int)
        im[-p[1], p[0], :] = __color
        bresenham.teapop.bresenham_line(im,__color,y[0],y[1],p[0],p[1])
        y=p

P0=np.array([100,100])
P1=np.array([150,450])
P2=np.array([300,400])
P3=np.array([400,80])


fig = plt.figure()
ims=[]

for t in range(500):
    print(t)
    image=bresenham.teapop.create_2d_rgb_arr(__height,__width)
    beza(P0,P1+np.array([t,0]),P2-np.array([t,0]),P3,image)
    im = plt.imshow(image, animated=True)
    ims.append([im])


print('Frames creation finshed.')
ani = animation.ArtistAnimation(fig, ims, interval=10, repeat=True, blit=True)

plt.show()