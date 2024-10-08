import asyncio
import pygame as p
import numpy as np
import time

p.init()

# Display Settings
CAPTION = "3d"
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

display = p.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
p.display.set_caption(CAPTION)
p.event.set_grab(True)
p.mouse.set_visible(False)

#Clock
clock = p.time.Clock()
FPS = 60

# Colors
BLACK = p.Color(0, 0, 0)
WHITE = p.Color(255, 255, 255)
GRAY = p.Color(128, 128, 128)
RED = p.Color(255, 0, 0)
GREEN = p.Color(0, 255, 0)
BLUE = p.Color(0, 0, 255)

MOVE_SPEED = 5

#Objects
class Line:
    def __init__(self, point: np.ndarray, vector: np.ndarray):
        self.point = point
        self.vector = vector

class Plane:
    def __init__(self, point: np.ndarray, normal: np.ndarray):
        self.point = point
        self.normal = normal

class Polygon:
    def __init__(self, vertices: np.ndarray, color: p.Color):
        self.vertices = vertices
        self.color = color

    @property
    def center(self):
        return np.sum(self.vertices, axis=0)/len(self.vertices)

    def map_point_to_screen(self, point, camera_position, camera_screen, inverseAxisMatrix, shift, scale):
        line_to_camera = Line(point, camera_position - point)
        intersection_point = get_line_plane_intersection(line_to_camera, camera_screen)
        return ((inverseAxisMatrix @ (intersection_point - camera_position)+shift)*scale)[:-1]

    def map_vertices(self, camera_position, camera_screen, inverseAxisMatrix, shift, scale):            
        self.mapped_vertices = [self.map_point_to_screen(vertex, camera_position, camera_screen, inverseAxisMatrix, shift, scale) for vertex in self.vertices]
        return self.mapped_vertices

    def draw(self, camera_position, camera_screen, inverseAxisMatrix, shift, scale):
        self.map_vertices(camera_position, camera_screen, inverseAxisMatrix, shift, scale)
        self.draw_method()

    def draw_method(self):
        p.draw.polygon(display, self.color, np.rint(self.mapped_vertices))

    def vector_from_camera(self, camera_position):
        return self.center - camera_position

    def distance_from_camera(self, camera_position, camera_direction):
        return np.linalg.norm(self.vector_from_camera(camera_position)) * self.facing(camera_position, camera_direction)

    def facing(self, camera_position, camera_direction):
        return -np.sign(np.dot(self.vector_from_camera(camera_position), camera_direction))

    def is_front(self, camera_position, camera_direction):
        return self.facing(camera_position, camera_direction) == 1

class Triangle(Polygon):
    def __init__(self, vertices: np.ndarray, color=WHITE):
        super().__init__(vertices, color)
        assert vertices.shape == (3, 3)
        self.v1 = vertices[0]
        self.v2 = vertices[1]
        self.v3 = vertices[2]

    @property
    def normal(self):
        return np.cross(self.v2-self.v1, self.v3-self.v1)

class Segment(Polygon):
    def __init__(self, vertices, color=WHITE):
        super().__init__(vertices, color)
        self.v1 = vertices[0]
        self.v2 = vertices[1]

    def draw_method(self):
        p.draw.line(display, self.color, *np.rint(self.mapped_vertices))

# Vector Calculations
def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

def get_angle_between_vectors(u: np.ndarray, v: np.ndarray):  # Returns angle between vectors measured in radians
    # THETA = arccos(u.v/||u||||v||) 
    return np.arccos(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))

def get_line_plane_intersection(line: Line, plane: Plane) -> np.ndarray | None:  # Returns the point (if any) of intersection between a line and a plane
    """
    Line is given by points p such that: p = p0 + tv, where p0 is a fixed point, v fixed direction vector, t scalar.
    Plane is given by points q such that: n.(q - p1) = 0, where n is normal to plane, p1 fixed point.
    
    Result is given by: p = p0 + (n.(p1-p0) / n.v) v
    """
    p0 = line.point
    v = line.vector
    p1 = plane.point
    n = plane.normal
    
    ndotv = np.dot(n, v)   
    if np.isclose(ndotv, 0):
        return None
    
    t = np.dot(n, p1 - p0) / ndotv
    return p0 + t*v
    
# Camera
def get_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    # Anticlockwise rotations
    c, s = np.cos(alpha), np.sin(alpha)
    alpha_rotation_matrix = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
        ])
    # Matrix required to rotate a plane by BETA in the xz plane
    c, s = np.cos(beta), np.sin(beta)
    beta_rotation_matrix = np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
        ])
    # Matrix required to rotate a plane by GAMMA in the yz plane
    c, s = np.cos(gamma), np.sin(gamma)
    gamma_rotation_matrix = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
        ])
    return alpha_rotation_matrix @ beta_rotation_matrix @ gamma_rotation_matrix
        
def get_camera_direction(alpha, beta, gamma):
    i = np.sin(alpha)*np.sin(gamma) - np.cos(alpha)*np.sin(beta)*np.cos(gamma)
    j = -np.cos(alpha)*np.sin(gamma) - np.sin(alpha)*np.sin(beta)*np.cos(gamma)
    k = np.cos(beta)*np.cos(gamma)
    return np.array([i, j, k])

def get_angles_from_mouse() -> tuple[float, float, float]:
    x, y = p.mouse.get_rel()
    alpha = 0
    beta = x / DISPLAY_WIDTH
    gamma = y / DISPLAY_HEIGHT
    return alpha, beta, gamma


#Testing
polygons: list[Polygon] = []
polygons.append(Triangle(np.array([[1, 0, -10], [0, 0, -10], [0, 1, -10]]), RED))
polygons.append(Triangle(np.array([[1, 0, -10], [1, 1, -10], [0, 1, -10]])))

polygons.append(Triangle(np.array([[1, 0, 5], [0, 0, 5], [0, 1, 5]]), BLUE))
polygons.append(Triangle(np.array([[1, 0, 5], [1, 1, 5], [0, 1, 5]]), BLUE))

polygons.append(Triangle(np.array([[1, 0, 6], [0, 0, 6], [0, 1, 6]]), GREEN))
polygons.append(Triangle(np.array([[1, 0, 6], [1, 1, 6], [0, 1, 6]]), GREEN))

polygons.append(Triangle(np.array([[0, 0, 5], [0, 0, 6], [0, 1, 6]]), GRAY))
polygons.append(Triangle(np.array([[0, 1, 6], [0, 1, 5], [0, 0, 5]]), GRAY))

# Run
async def main():
    # Initial Camera Settings
    alpha = 0.  # Angle in the xy plane
    beta = 0.  # Angle in the xz plane
    gamma = 0.  # Angle in the yz plane
    x0 = 0.  # x coordinate of camera
    y0 = 0.  # y coordinate of camera
    z0 = 0.  # z coordinate of camera

    camera_position = np.array([x0, y0, z0])  # Initially at the origin, position given as a coordinate np.array
    camera_direction = get_camera_direction(alpha, beta, gamma)  # Direction is given as a unit vector np.array and is the normal to the camera_screen
    SHIFT = np.array([0.5, 0.5, 0])
    SCALE = np.array([DISPLAY_WIDTH, DISPLAY_WIDTH, 0])

    t0 = time.time()
    
    running = True
    while running:
        camera_screen = Plane(camera_position + camera_direction, camera_direction)  # Represents the plane the display is in [point, normal unit vector
        axisMatrix = get_rotation_matrix(alpha, beta, gamma)  # The orientation of the xyz axis relative to the camera
        inverseAxisMatrix = np.linalg.inv(axisMatrix)  # The inverse of the above, used in order to map back to the display
        
        # Controls
        keys = p.key.get_pressed()
        forward_speed, side_speed, fly_speed = 0, 0, 0
        if keys[p.K_a]:
            side_speed -= MOVE_SPEED
        if keys[p.K_d]:
            side_speed += MOVE_SPEED
        if keys[p.K_w]:
            forward_speed += MOVE_SPEED
        if keys[p.K_s]:
            forward_speed -= MOVE_SPEED
        if keys[p.K_SPACE]:
            fly_speed += MOVE_SPEED
        if keys[p.K_LSHIFT]:
            fly_speed -= MOVE_SPEED
        if keys[p.K_ESCAPE]:
            p.event.post(p.event.Event(p.QUIT))
        
        # Rendering
        display.fill(BLACK)
        sorted_polygons = sorted(polygons, key=lambda x: -x.distance_from_camera(camera_position, camera_direction))
        for polygon in sorted_polygons:
            if polygon.is_front(camera_position, camera_direction):
                polygon.draw(camera_position, camera_screen, inverseAxisMatrix, SHIFT, SCALE)
        
        display.blit(p.transform.flip(display, False, False), (0, 0))

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        d0, d1, d2 = get_angles_from_mouse()
        alpha += d0
        beta += d1
        gamma += d2
        
        forward_v = -camera_direction
        forward_v[1] = 0.
        side_v = np.cross(camera_direction, np.array([0, 1, 0]))
        fly_v = np.array([0., 1., 0.])

        camera_direction = get_camera_direction(alpha, beta, gamma)
        camera_position += forward_speed*dt*forward_v + side_speed*dt*side_v + fly_speed*dt*fly_v
        
        for event in p.event.get():
            if event.type == p.QUIT:
                running = False
                
        p.display.update()
        clock.tick(FPS)
        await asyncio.sleep(0)

asyncio.run(main())