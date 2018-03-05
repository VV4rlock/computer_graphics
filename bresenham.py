import matplotlib.pyplot as plt
import numpy as np
import math


class teapop:

    def __init__(self,filename): #return x,y (0..1)
        x=[]
        y=[]
        with open(filename) as file:
            for i in file.readlines():
                if (i[0]=='v'):
                    b = list(map(float, i[2:].split()))
                    x.append(b[0])
                    y.append(b[1])
        self.x,self.y=np.array(x), np.array(y)-max(y)/2
        self.peaks = []
        with open(filename) as file:
            for i in file.readlines():
                if (i[0] == 'f'):
                    self.peaks.append(tuple(map(int, i[2:].split())))

    def coord_normalization(self,x,y):
        x = np.array(x) + abs(min(x))
        y = np.array(y) + abs(min(y))
        all_max = max(max(x), max(y))
        return x / all_max, y / all_max

    def get_x_y(self):
        return self.x,self.y

    def get_peaks(self):
        return self.peaks

    @staticmethod
    def create_2d_rgb_arr(height,width):
        return np.zeros((height,width,3), dtype = np.uint8)

    def reflect_to_large_scale_by_the_bigest_side(self,x,y,height, width):
        mx=max(x)
        my=max(y)
        m=min(height/my,width) if mx>my else min(width/mx,height)
        return (x*(m-1)).astype(int), (y*(m-1)).astype(int)

    def show_image(self,image):
        plt.figure()
        plt.imshow(image)
        plt.show()

    def save_image(self,filename, image):
        plt.imsave(filename,image)

    @staticmethod
    def print_pixel(im,color,x,y):
        if x<0:
            return
        try:
            im[im.shape[0] - y-1, x, :] = color
        except:
            print("не удалось напечатать пиксель",x,y)

    @staticmethod
    def bresenham_line(im,color,x1,y1,x2,y2):
        #assert min(x1,x2,y1,y2)>-1
        deltaX = abs(x2 - x1)
        deltaY = abs(y2 - y1)
        signX = 1 if x1 < x2 else -1
        signY = 1 if y1 < y2 else -1
        error = deltaX - deltaY
        teapop.print_pixel(im, color, x2, y2)
        while x1 != x2 or y1 != y2:
            teapop.print_pixel(im,color,x1, y1)
            error2 = error * 2
            if error2 > -deltaY:
                error -= deltaY
                x1 += signX
            if error2 < deltaX:
                error += deltaX
                y1 += signY


    def print_triangle(self,im,color,peaks,x,y):
        self.bresenham_line(im,color,x[peaks[0]-1],y[peaks[0]-1],
                              x[peaks[1]-1],y[peaks[1]-1])
        self.bresenham_line(im, color, x[peaks[2] - 1], y[peaks[2] - 1],
                              x[peaks[1] - 1], y[peaks[1] - 1])
        self.bresenham_line(im, color, x[peaks[2] - 1], y[peaks[2] - 1],
                              x[peaks[0] - 1], y[peaks[0] - 1])

    def reflect_by_min_h_w(self,x, y, height, width):
        m = min(height, width)
        return (x * (m - 1)).astype(int), (y * (m - 1)).astype(int)

    def frame(self,height, width, peaks, x, y, color):
        im = self.create_2d_rgb_arr(height, width)
        for i in peaks:
            self.print_triangle(im, color, i, x, y)
        return im

    def aphine_point(self,x, y, angle, x_cof, y_cof, x_offset, y_offset):
        R = np.array([[x_cof * math.cos(angle), math.sin(angle), 0],
                      [-math.sin(angle), y_cof * math.cos(angle), 0],
                      [x_offset, y_offset, 1]])
        a = np.array([x, y, 1]).dot(R)
        return a[0], a[1]

    def get_aphine_x_y(self, angle, x_cof, y_cof, x_offset, y_offset):
        x=np.zeros(self.x.shape)
        y=np.zeros(self.y.shape)
        for i in range(len(self.x)):
            x[i],y[i]=self.aphine_point(self.x[i],self.y[i],angle,x_cof,y_cof,x_offset,y_offset)
        return x,y