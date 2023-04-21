import math
import cv2

class rotated_box:
    def __init__(self, xmin=None, ymin=None, xmax=None, ymax=None, w=None, h=None, angle=None, vertices=None, text=''):
        if vertices is None:
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            self.w, self.h = w, h # rectangle width and height
            self.d = math.sqrt(self.w**2 + self.h**2)/2.0 # distance from center to vertices    
            self.c = (int(cx+self.w/2.0),int(cy+self.h/2.0)) # center point coordinates
            self.angle = angle # rotation angle
            self.alpha = math.radians(self.angle) # rotation angle in radians
            self.beta = math.atan2(self.h, self.w) # angle between d and horizontal axis

            # Center Rotated vertices in image frame
            self.p0 = (int(self.c[0] - self.d * math.cos(self.beta - self.alpha)), int(self.c[1] - self.d * math.sin(self.beta-self.alpha))) 
            self.p1 = (int(self.c[0] - self.d * math.cos(self.beta + self.alpha)), int(self.c[1] + self.d * math.sin(self.beta+self.alpha))) 
            self.p2 = (int(self.c[0] + self.d * math.cos(self.beta - self.alpha)), int(self.c[1] + self.d * math.sin(self.beta-self.alpha))) 
            self.p3 = (int(self.c[0] + self.d * math.cos(self.beta + self.alpha)), int(self.c[1] - self.d * math.sin(self.beta+self.alpha))) 

            self.vertices = [self.p0, self.p1, self.p2, self.p3]

        else:
            self.vertices = [(int(vertices[0]), int(vertices[1])),
                            (int(vertices[2]), int(vertices[3])),
                            (int(vertices[4]), int(vertices[5])),
                            (int(vertices[6]), int(vertices[7]))]
        self.text = text

    def get_4_points(self):
        return self.vertices

    def draw(self, image):
        for i in range(-1, len(self.vertices)-1):
            cv2.line(image, (self.vertices[i][0], self.vertices[i][1]), (self.vertices[i+1][0],self.vertices[i+1][1]), (255,0,0), 2)

        for i in range(-1, len(self.vertices)-1):
            cv2.circle(image, (self.vertices[i][0], self.vertices[i][1]), 4, (0, 0, 255),-1)
            cv2.putText(image, f'{i+1 if i!=-1 else 4}', (self.vertices[i][0], self.vertices[i][1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),1)
        
        y = self.vertices[0][1] - 9
        dy = 36
        for i, t in enumerate(self.text.split('\n')):
            cv2.putText(image, t, (self.vertices[0][0]+5, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),1)
            y = y + dy

            # cv2.putText(image, text, (self.vertices[0][0]+5, self.vertices[0][1]-9),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1)
