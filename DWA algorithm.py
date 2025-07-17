"""

Dynamic Window Approach with Deep Reinforcement Learning
by Carlos VJ

"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from AC import Actor_Critic

import openpyxl
from openpyxl import Workbook

from matplotlib import animation


show_animation = True



def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)  # Calculate the dynamic window

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob) # Calculate the optimal control input and predicted trajectory
    
    return u, trajectory # Return the control input and trajectory


class RobotType(Enum):
    """
    Enum representing the robot's shape (circle or rectangle).
    """
    circle = 0
    rectangle = 10


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # 1 [m/s] 
        self.min_speed = -0.3  # -0.5 [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.3  # 0.2 [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 0.5
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1  # 1 [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = np.random.uniform(-1, 20, size=(15, 2)) #coordenadas de los obstáculos

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()


def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt  # Update yaw angle
    x[0] += u[0] * math.cos(x[2]) * dt # Update x-coordinate
    x[1] += u[0] * math.sin(x[2]) * dt # Update y-coordinate
    x[3] = u[0]  # Update linear velocity
    x[4] = u[1]  # Update yaw rate

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate] # [vmin, vmax, yawrate_min, yawrate_max]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt, # Minimum linear velocity considering acceleration
          x[3] + config.max_accel * config.dt, # Maximum linear velocity considering acceleration
          x[4] - config.max_delta_yaw_rate * config.dt, # Minimum yaw rate considering acceleration
          x[4] + config.max_delta_yaw_rate * config.dt] # Maximum yaw rate considering acceleration

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width, fc='black', ec='black')
    plt.plot(x, y)

def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             config.robot_length / 2, -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k", label='Robot')
        plt.fill(outline[0, :], outline[1, :], color="skyblue", alpha=0.5)
    elif config.robot_type == RobotType.circle:
        # Dibuja el cuerpo del robot como un círculo semitransparente
        circle = plt.Circle((x, y), config.robot_radius, color="blue", alpha=0.5, label="Robot")
        plt.gcf().gca().add_artist(circle)
        
        # Dirección del robot
        out_x = x + config.robot_radius * np.cos(yaw)
        out_y = y + config.robot_radius * np.sin(yaw)
        plt.arrow(x, y, out_x - x, out_y - y, head_width=0.1, head_length=0.2, fc='black', ec='black')

    # Mejora general del gráfico
    plt.axis("equal")
    plt.grid(True)

def update_obstacles(ob, step):
    """
    Mueve los obstáculos en patrones simples.
    Por ejemplo: hacia la derecha y de regreso.
    """
    ob_updated = ob.copy()

    # Movimiento tipo onda senoidal en y
    for i in range(len(ob_updated)):
        ob_updated[i][1] += 0.05 * np.sin(0.1 * step + i)

        # Alternativamente, puedes moverlos también en x:
        ob_updated[i][0] += 0.03 * np.cos(0.05 * step + i)

    return ob_updated


def main(gx=20, gy=15, robot_type=RobotType.rectangle, alfa = 0, k = 0.5, H_umbral = 0.3):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), teta(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    r = 0
    step = 0
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob.copy()
    
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["alfa", "vr", "wr", "vp", "wp"]) # Encabezados

    while True:
        dist_to_goali = math.hypot(x[0] - goal[0], x[1] - goal[1])
        red = Actor_Critic(x[0], x[1], x[2], x[3], x[4], ob)
        u_ac = red.actor()
        u_ac = u_ac[0].tolist()
        red.critic()
        
        ob = update_obstacles(ob, step)
        if step == 0:
            ob_traj = [ob.copy()]
        else:
            ob_traj.append(ob.copy())

        
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        
        # Función de transición (entropía alfa)
        v = alfa * (u_ac[0]) + (1 - alfa) * (u[0])    #velocidad lineal híbrida
        w = alfa * (u_ac[1]) + (1 - alfa) * (u[1])    #velocidad angular híbrida
        sheet.append([alfa, u[0], u[1], u_ac[0], u_ac[1]])
        #v = u_ac[0]
        #w = u_ac[1]
        #u = [v,w]
        H = -(u_ac[0]*math.log(u_ac[0]) + u_ac[1]*math.log(u_ac[1]))  #Entropía, incertifdumbre de la red
        
        alfa = 1 / (1+ math.exp(-k * (H_umbral - H)))  #k es la velocidad de la transición 
        

        x = motion(x, u, config.dt)  # simulate robot 
        trajectory = np.vstack((trajectory, x))  # store state history
        x2 = motion(x, u, config.dt)  # simulate robot 

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1])
            plt.plot(x2[0], x2[1], "xr")
            plot_robot(x2[0], x2[1], x2[2], config)
            plot_arrow(x2[0], x2[1], x2[2])
                       
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goalf = math.hypot(x[0] - goal[0], x[1] - goal[1])
        reward = red.reward(dist_to_goalf, dist_to_goali)
        r = r + reward
        print("step: ", step)
        step += 1
        
        if step == 6000:
            break
        
        if dist_to_goalf <= config.robot_radius:
            print("Goal!!")
            break
    
    workbook.save("datos_robot.xlsx") # Guardar el archivo Excel
    print("Datos guardados en datos_robot.xlsx")
    print("Done")
    
    ob_traj = np.array(ob_traj)
    min_len = min(len(ob_traj), len(trajectory))
    trajectory = trajectory[:min_len]
    ob_traj = ob_traj[:min_len]
    
    if show_animation:
        # Animación estática final
        plt.close("all")  # Cierra cualquier figura anterior

        # Extraer solo las coordenadas x, y para la animación
        xy_traj = trajectory[:, :2]  # Solo las primeras dos columnas

        fig, ax = plt.subplots()
        ax.set_xlim(np.min(xy_traj[:, 0]) - 1, np.max(xy_traj[:, 0]) + 1)
        ax.set_ylim(np.min(xy_traj[:, 1]) - 1, np.max(xy_traj[:, 1]) + 1)
        ax.set_aspect('equal')
        ax.set_title("Trayectoria del robot")
        ax.grid(True)

        robot_dot, = ax.plot([], [], 'bo', label="Robot")
        trajectory_line, = ax.plot([], [], 'b--', label="Trayectoria")
        ob_dots, = ax.plot([], [], 'ok') 
        

        def init():
            robot_dot.set_data([], [])
            trajectory_line.set_data([], [])
            ob_dots.set_data([], [])
            return robot_dot, trajectory_line, ob_dots

        def animate(i):
            robot_dot.set_data(trajectory[i, 0], trajectory[i, 1])
            trajectory_line.set_data(trajectory[:i+1, 0], trajectory[:i+1, 1])

        # Obstáculos para este frame
            ob = ob_traj[i]
            ox = ob[:, 0]
            oy = ob[:, 1]
            ob_dots.set_data(ox, oy)

            return robot_dot, trajectory_line, ob_dots


        ani = animation.FuncAnimation(
            fig, animate,
            frames=len(trajectory),
            init_func=init,
            blit=True,
            interval=100
            )

        ani.save("robot_path.gif", writer="imagemagick", fps=10)  # Usar 'pillow' si imagemagick falla
        print("GIF guardado como robot_path.gif")

        plt.show()
    

if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)
    #main(robot_type=RobotType.circle)
    






                   



   
    
 


