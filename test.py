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



b=bresenham.teapop(__filename)
x,y=b.get_x_y()

min_x=min(x)
min_y=min(y)
max=max(max(x),max(y))



fig = plt.figure()
ims = []

args_of_angle = np.arange(0, 2*np.pi, __step)

offset = np.pi*3/4

for ang in args_of_angle:
    print(ang)
    ax,ay=b.get_aphine_x_y(ang,1+abs(math.sin(ang/2)),1+abs(math.sin(ang/2)),0,0)
    ax = ((ax + abs(min_x)) / max) * 0.3+0.2
    ay = ((ay + abs(min_y)) / max) * 0.3+0.5
    ax,ay=b.reflect_by_min_h_w(ax,ay,__height,__width)
    fr=b.frame(__height,__width,b.peaks,ax,ay,[abs(math.sin(ang))*254,
                                               abs(math.sin(ang+offset))*254,abs(math.sin(ang-offset))*254])
    im=plt.imshow(fr, animated=True)
    ims.append([im])

print('Frames creation finshed.')
ani = animation.ArtistAnimation(fig, ims, interval=150, repeat=True, blit=True)
# ani.save('dynamic_images.mp4')
plt.show()
