import pygame as p
import math
import numpy as np
import time

p.init()

# Display Settings
display_width = 800
display_height = 600
display = p.display.set_mode((display_width, display_height))
CAPTION = "3d"
p.display.set_caption(CAPTION)
p.event.set_grab(True)
p.mouse.set_visible(False)

#Clock
clock = p.time.Clock()
FPS = 120

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initial Camera Settings
alpha = 0.  # Angle in the xy plane
beta = 0.  # Angle in the xz plane
gamma = 0.  # Angle in the yz plane
x0 = 0.  # x coordinate of camera
y0 = 0.  # y coordinate of camera
z0 = 0.  # z coordinate of camera

# Vector Calculations
def unitVector(vector):
    return vector / np.linalg.norm(vector)

def angleBetweenVectors(v1, v2):  # Returns angle between vectors measured in radians
    v1_u = unitVector(v1)
    v2_u = unitVector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def intersectionOfLineAndPlane(lineData, planeData):  # Returns the point (if any) of intersection between a line and a plane
    # lineData is in the form :: [p, v] where p is the point and v is a vector, given as np.arrays
    # planeData is in the form :: [p, n] as above, but n is the normal vector to the plane
    V0 = planeData[0]
    P0 = lineData[0]
    n = planeData[1]
    u = lineData[1]
    if np.dot(n, u) == 0:
        return np.array([0, 0, 0])
    w = P0 - V0
    s = -np.dot(n, w) / np.dot(n, u)  # Calculates the value of the parametric variable
    return P0 + s*u
    
# Camera
def getAxisMatrix(ALPHA, BETA, GAMMA):
    alpha_rotation_matrix = np.array([
        [math.cos(ALPHA), -math.sin(ALPHA), 0],
        [math.sin(ALPHA), math.cos(ALPHA), 0],
        [0, 0, 1]
        ])  # Matrix required to rotate a plane by ALPHA in the xy plane
    beta_rotation_matrix = np.array([
        [math.cos(BETA), 0, -math.sin(BETA)],
        [0, 1, 0],
        [math.sin(BETA), 0, math.cos(BETA)]
        ])  # Matrix required to rotate a plane by ALPHA in the xz plane
    gamma_rotation_matrix = np.array([
        [1, 0, 0],
        [0, math.cos(GAMMA), -math.sin(GAMMA)],
        [0, math.sin(GAMMA), math.cos(GAMMA)]
        ])  # Matrix required to rotate a plane by ALPHA in the yz plane
    resultant_matrix = np.dot(np.dot(alpha_rotation_matrix, beta_rotation_matrix), gamma_rotation_matrix)
    return resultant_matrix
        
def findCameraDirection(ALPHA, BETA, GAMMA):
    i = math.sin(ALPHA)*math.sin(GAMMA) - math.cos(ALPHA)*math.sin(BETA)*math.cos(GAMMA)
    j = -math.cos(ALPHA)*math.sin(GAMMA) - math.sin(ALPHA)*math.sin(BETA)*math.cos(GAMMA)
    k = math.cos(BETA)*math.cos(GAMMA)
    return np.array([i, j, k])

def getInverseAxisMatrix(ALPHA, BETA, GAMMA):
    resultant_matrix = np.linalg.inv(getAxisMatrix(ALPHA, BETA, GAMMA))
    return resultant_matrix

def getAnglesFromMousePosition():
    x, y = p.mouse.get_rel()
    ALPHA = 0
    BETA = x/display_width
    GAMMA = y/display_height
    return np.array([ALPHA, BETA, GAMMA])

def getRelativeAxes(key):
    axesDict = {}
    axesDict["forward"] = -camera_direction
    axesDict["forward"][1] = 0.
    axesDict["side"] = np.cross(camera_direction, np.array([0, 1, 0]))
    return axesDict[key]
    

axisMatrix = getAxisMatrix(alpha, beta, gamma)  # The orientation of the xyz axis relative to the camera
inverseAxisMatrix = np.linalg.inv(axisMatrix)  # The inverse of the above,  used in order to map back to the display
camera_position = np.array([x0, y0, z0])  # Initially at the origin, position given as a coordinate np.array
camera_direction = findCameraDirection(alpha, beta, gamma)  # Direction is given as a unit vector np.array and is the normal to the camera_screen
camera_screen = [camera_position + camera_direction, camera_direction]  # Represents the plane the display is in [point, normal unit vector]
shift = np.array([0.5, 0.5, 0])
scale = np.array([display_width, display_width, 0])

#Objects
class Shape:
    def __init__(self, vertices, color):
        self.vertices = vertices
        self.color = color

    @property
    def center(self):
        return np.sum(self.vertices, axis=0)/len(self.vertices)

    def map_point_to_screen(self, point):
        line_to_camera = [point, camera_position - point]
        intersection_point = intersectionOfLineAndPlane(line_to_camera, camera_screen)
        mapped_point = ((np.dot(inverseAxisMatrix, intersection_point - camera_position)+shift)*scale)[:-1]
        return mapped_point

    def map_vertices(self):
        self.mapped_vertices = []
        for vertex in self.vertices:
            self.mapped_vertices.append(self.map_point_to_screen(vertex))
        return self.mapped_vertices

    def draw(self):
        for mapped_vertex in self.map_vertices():
            if np.linalg.norm(mapped_vertex) > display_width**2:  # Coordinates can't be too extreme
                return
        self.draw_method()

    def draw_method(self):
        p.draw.polygon(display, shape.color, np.rint(self.mapped_vertices))

    def vector_from_camera(self):
        return self.center - camera_position

    def distance_from_camera(self):
        return np.linalg.norm(self.vector_from_camera())*self.facing()

    def facing(self):
        return -np.sign(np.dot(self.vector_from_camera(), camera_direction))

    def is_front(self):
        return True if self.facing() == 1 else False

class Triangle(Shape):
    def __init__(self, vertices, color=WHITE):  # Vertices of the triangle, (should be np.array of shape (3, 1))
        super().__init__(vertices, color)
        self.v1 = vertices[0]
        self.v2 = vertices[1]
        self.v3 = vertices[2]

    @property
    def normal(self):
        return np.cross(self.v2-self.v1, self.v3-self.v1)

class Line(Shape):
    def __init__(self, vertices, color=WHITE):
        super().__init__(vertices, color)
        self.v1 = vertices[0]
        self.v2 = vertices[1]

    def draw_method(self):
        p.draw.line(display, self.color, *np.rint(self.mapped_vertices))

#Testing
shapes = []
##for i in range(11):
##        line_x = Line(np.array([[0, 0, i], [10, 0, i]]), RED)
##        line_z = Line(np.array([[i, 0, 0], [i, 0, 10]]), BLUE)
##        shapes += [line_x, line_z]
shapes.append(Triangle(np.array([[1, 0, -10], [0, 0, -10], [0, 1, -10]]), RED))
shapes.append(Triangle(np.array([[1, 0, -10], [1, 1, -10], [0, 1, -10]])))
tri1 = Triangle(np.array([[1, 0, 5], [0, 0, 5], [0, 1, 5]]), BLUE)
tri2 = Triangle(np.array([[1, 0, 5], [1, 1, 5], [0, 1, 5]]), BLUE)
shapes.append(tri1)
shapes.append(tri2)
tri1 = Triangle(np.array([[1, 0, 6], [0, 0, 6], [0, 1, 6]]), GREEN)
tri2 = Triangle(np.array([[1, 0, 6], [1, 1, 6], [0, 1, 6]]), GREEN)
shapes.append(tri1)
shapes.append(tri2)
##tri1 = Triangle(np.array([[0, 0, 5], [0, 0, 6], [0, 1, 6]]), GRAY)
##tri2 = Triangle(np.array([[0, 1, 6], [0, 1, 5], [0, 0, 5]]), GRAY)
##shapes.append(tri1)
##shapes.append(tri2)
##tri1 = Triangle(np.array([[1, 0, 5], [1, 0, 6], [1, 1, 6]]), RED)
##tri2 = Triangle(np.array([[1, 1, 6], [1, 1, 5], [1, 0, 5]]), RED)
##shapes.append(tri1)
##shapes.append(tri2)

# Run
running = True
t0= time.time()
while running:
    # Rendering
    display.fill(BLACK)
    camera_screen = [camera_position + camera_direction, camera_direction]  # Represents the plane the display is in.
    axisMatrix = getAxisMatrix(alpha, beta, gamma)
    inverseAxisMatrix = np.linalg.inv(axisMatrix)
    shapes.sort(key=lambda x: x.distance_from_camera(), reverse=True)
    for shape in shapes:
        if shape.is_front():
            shape.draw()
        else:
            break
    display.blit(p.transform.flip(display, False, False), (0, 0))
    p.display.update()
    # Controls
    keys = p.key.get_pressed()
    forward_speed, side_speed = (0, 0)
    if keys[p.K_a]:
        side_speed = -3
    elif keys[p.K_d]:
        side_speed = 3
    if keys[p.K_w]:
        forward_speed = 3
    elif keys[p.K_s]:
        forward_speed = -3
    if keys[p.K_p]:
        print(camera_position)
        print(camera_direction)
    # Animate
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    delta_angles = getAnglesFromMousePosition()
    alpha += delta_angles[0]
    beta += delta_angles[1]
    gamma += delta_angles[2]
    #beta = -t
    #gamma = -0.2
    #camera_position = np.array([5*math.sin(t), 1, 5 + 5*math.cos(t)])
    camera_direction = findCameraDirection(alpha, beta, gamma)
    camera_position += forward_speed*dt*getRelativeAxes("forward") + side_speed*dt*getRelativeAxes("side")
    # Misc
    for event in p.event.get():
        if event.type == p.QUIT:
            running = False
    clock.tick(FPS)

