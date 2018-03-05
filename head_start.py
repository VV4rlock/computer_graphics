import head
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

__step=1
__radius=5
source=np.array([500, 500, 500])
id= np.array([250,250,250]) #deffus
kd=1
i_s=10 #spaw
k_s=10
alpha=10

obj=head.Obj_render()
obj.set_light(source,id,kd,i_s,k_s,alpha)
#obj.show_image(obj.print2dhead(700,700,10,3,[-__radius*math.sin(0),0,-__radius*math.cos(0)]))

#exit(0)
fig = plt.figure()
ims = []

args_of_angle = np.arange(0, 2*np.pi, __step)
l=len(args_of_angle)
k=1
for ang in args_of_angle:
    print("rendering {}/{} frame".format(k,l))
    k+=1
    #obj.set_light(np.array([-__radius*math.sin(ang),0,-__radius*math.cos(ang)]),id,kd,i_s,k_s,alpha)
    fr = obj.print2dhead(700,700,10,3,[-__radius*math.sin(ang),0,-__radius*math.cos(ang)])
    im = plt.imshow(fr, animated=True)
    ims.append([im])

print('Frames creation finshed.')
ani = animation.ArtistAnimation(fig, ims, interval=150, repeat=True, blit=True)
#ani.save('dynamic_images.mp4')
plt.show()