import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as alg
import math
import matplotlib.animation as animation
import random as rnd

class Obj_render:
    def __init__(self):
        self.texture=plt.imread('african_head_diffuse.tga')
        x=[]
        y=[]
        z=[]
        self.peaks=[]
        self.number_of_texture_coord=[]
        self.third=[]
        self.normals=[]
        x_textures=[]
        y_textures=[]
        with open("african_head.obj") as f:
            for i in f.readlines():
                if i.startswith("v "):
                    b = list(map(float, i[2:].split()))
                    x.append(b[0])
                    y.append(b[1])
                    z.append(b[2])
                elif i.startswith("f "):
                    temp=i[2:].split() # верш/nomer текстурi/еще чтото
                    a=list(map(int,temp[0].split('/')))
                    b=list(map(int,temp[1].split('/')))
                    c=list(map(int,temp[2].split('/')))
                    self.peaks.append((a[0],b[0],c[0]))
                    self.number_of_texture_coord.append((a[1],b[1],c[1]))
                    self.third.append((a[2],b[2],c[2]))
                elif i.startswith("vt "):
                    temp=list(map(float,i[3:].split()))
                    x_textures.append(temp[0])
                    y_textures.append(temp[1])
                elif i.startswith("vn "):
                    self.normals.append(tuple(map(float, i[3:].split())))
        self.x=np.array(x)
        self.y=np.array(y)
        self.z=np.array(z)
        self.x_textures=np.array(x_textures)
        self.y_textures=np.array(y_textures)
        self.set_light(np.array([500, 500, 500]), np.array([250,250,250]), 1,10,10,10)

    def set_light(self,source,id,kd,i_s,k_s,alpha,am=None):
        self.light_source, self.id, self.kd = source, id,kd
        self.i_s=i_s
        self.k_s=k_s
        self.alpha=alpha
        if am is None:
            self.amb = 0.3*source
        else:
            self.amb=am

    def print2dhead(self,width,height,f,n,camera):
        im=Obj_render.create_2d_rgb_arr(height,width)
        camera = np.array(camera)
        self.camera=camera
        r=math.sqrt((camera[0]*camera[0]+camera[2]*camera[2]))
        x, y, z = self.aphine_rotation_y(-camera[2]/r, camera[0]/r, self.x, self.y, self.z)
        px = width/2
        py = height/2
        ox = width/2
        oy = height/2
        x, y, z = self.reflect_modification(n, camera,x,y,z)
        x=px*x+ox
        y=py*y+oy
        z=int((f-n)/2)*z+(f+n)/2
        z_buffer = np.ones((width + 1, height + 1), dtype=np.float16) * (-float("Inf"))
        for i in range(len(self.peaks)):
            A_norm = np.array(self.normals[self.peaks[i][0] - 1])
            B_norm = np.array(self.normals[self.peaks[i][1] - 1])
            C_norm = np.array(self.normals[self.peaks[i][2] - 1])
            norm = self.normolize(A_norm + B_norm + C_norm)
            if camera.dot(norm) >= 0:
                continue
            self.fill_triangle(im,z_buffer,self.number_of_texture_coord[i],
                               self.peaks[i],x,y,z,norm)
        return im

    def normolize(self,a):
        return a/math.sqrt(a.dot(a))

    def ambient(self):
        return self.amb

    def cos_between(self,a,b):
        return (a.dot(b)/math.sqrt(a.dot(a)*b.dot(b)))

    def spawlar(self,a,norm,light_source,i_s,k_s,alpha,camera):
        L=np.array([light_source[0]-a[0],light_source[1]-a[1],light_source[2]-a[2]])
        V=np.array([camera[0]-a[0],camera[1]-a[1],camera[2]-a[2]])
        n=norm.dot(norm)
        R=2*norm*(L.dot(norm)/n)-L
        t=self.cos_between(R, V)
        #if t<=0:
        #    return 0
        return k_s*i_s*(t**alpha)

    def diffuse(self,a,norm,light_source,id,kd):
        L=np.array([light_source[0]-a[0],light_source[1]-a[1],light_source[2]-a[2]])
        temp=self.cos_between(L,norm)
        #if temp<=0:
        #    return 0
        return kd*temp*id

    def reflect_modification(self,D,camera,x,y,z):
        w =abs(z- math.sqrt(camera[2]**2+camera[0]**2))
        x = x*D/w
        y = y*D/w
        z = z/w*D
        return (x,y,z)

    def aphine_rotation_y(self,cos,sin,x,y,z):
        x1= cos*x+sin*z
        z1 = -sin*x+cos*z
        y1=y
        return (x1,y1,z1)

    @staticmethod
    def print_pixel(im, color, x, y):
        if x < 0:
            return
        try:
            im[im.shape[0] - y - 1, x, :] = color
        except:
            print("не удалось напечатать пиксель", x, y)

    @staticmethod
    def create_2d_rgb_arr(height, width):
        return np.zeros((height, width, 3), dtype=np.uint8)

    @staticmethod
    def show_image(image):
        plt.figure()
        plt.imshow(image)
        plt.show()

    @staticmethod
    def save_image(filename, image):
        plt.imsave(filename,image)

    @staticmethod
    def bresenham_line(im, color, x1, y1, x2, y2):
        # assert min(x1,x2,y1,y2)>-1
        deltaX = abs(x2 - x1)
        deltaY = abs(y2 - y1)
        signX = 1 if x1 < x2 else -1
        signY = 1 if y1 < y2 else -1
        error = deltaX - deltaY
        Obj_render.print_pixel(im, color, x2, y2)
        while x1 != x2 or y1 != y2:
            Obj_render.print_pixel(im, color, x1, y1)
            error2 = error * 2
            if error2 > -deltaY:
                error -= deltaY
                x1 += signX
            if error2 < deltaX:
                error += deltaX
                y1 += signY

    @staticmethod
    def print_triangle(im,color,peaks,x,y):
        Obj_render.bresenham_line(im,color,x[peaks[0]-1],y[peaks[0]-1],
                              x[peaks[1]-1],y[peaks[1]-1])
        Obj_render.bresenham_line(im, color, x[peaks[2] - 1], y[peaks[2] - 1],
                              x[peaks[1] - 1], y[peaks[1] - 1])
        Obj_render.bresenham_line(im, color, x[peaks[2] - 1], y[peaks[2] - 1],
                              x[peaks[0] - 1], y[peaks[0] - 1])



    def fill_triangle(self, im, z_buffer, texture_number, peaks, x, y, z,norm):
        A=(x[peaks[0] - 1].astype(np.float128), y[peaks[0] - 1].astype(np.float128),z[peaks[0] - 1].astype(np.float128))
        B=(x[peaks[1] - 1].astype(np.float128), y[peaks[1] - 1].astype(np.float128),z[peaks[1] - 1].astype(np.float128))
        C=(x[peaks[2] - 1].astype(np.float128), y[peaks[2] - 1].astype(np.float128),z[peaks[2] - 1].astype(np.float128))
        def det(A, B, C):
            return A[0] * B[1] - A[1] * B[0] - A[0] * C[1] + A[1] * C[0] + B[0] * C[1] - C[0] * B[1]
        S=det(A,B,C)
        if S==0:
            return
        #print(S)
        x_min=(min(A[0],B[0],C[0])).astype(int)
        x_max=(max(A[0],B[0],C[0])).astype(int)
        y_min=(min(A[1],B[1],C[1])).astype(int)
        y_max=(max(A[1],B[1],C[1])).astype(int)
        #print(A,B,C)
        temp=((B[1]-A[1])*(C[2]-A[2])-(B[2]-A[2])*(C[1]-A[1]),
              (B[0]-A[0])*(C[2]-A[2])-(B[2]-A[2])*(C[0]-A[0]),
              (B[0]-A[0])*(C[1]-A[1])-(B[1]-A[1])*(C[0]-A[0]))

        for x in range(x_min,x_max+1):
            for y in range(y_min,y_max+1):
                a = det([x, y], B, C)/S
                if a < 0:
                    continue
                b = det([x, y], C, A)/S
                if b < 0:
                    continue
                c = 1-a-b
                if c < 0:
                    continue
                z_temp = (temp[1]*(y-A[1])-temp[0]*(x-A[0]))/temp[2]+A[2]
                if z_temp > z_buffer[x, y]:
                    z_buffer[x, y] = z_temp
                    t_x = int(
                        (a * self.x_textures[texture_number[0] - 1] + b * self.x_textures[texture_number[1] - 1] + c *
                         self.x_textures[texture_number[2] - 1]) * self.texture.shape[0])
                    t_y = int(
                        (a * self.y_textures[texture_number[0] - 1] + b * self.y_textures[texture_number[1] - 1] + c *
                         self.y_textures[texture_number[2] - 1]) * self.texture.shape[1])
                    I = self.ambient() + self.diffuse(np.array([x, y, z_temp]), norm, self.light_source, self.id, self.kd) +\
                        self.spawlar(np.array([x,y,z_temp]),norm,self.light_source,self.i_s,self.k_s,self.alpha, self.camera)
                    if I[0] > 255:
                        I[0] = 255
                    if I[1] > 255:
                        I[1] = 255
                    if I[2] > 255:
                        I[2] = 255
                    if I[0] <0:
                        I[0] =0
                    if I[1] <0:
                        I[1] =0
                    if I[2] <0:
                        I[2] = 0
                    color = (self.texture[-t_y, t_x].astype(np.float128) / 255 * I).astype(int)
                    Obj_render.print_pixel(im, color, x, y)
