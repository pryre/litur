#!/usr/bin/env python3

import typing
import math

import rclpy
from rclpy.node import Node
from dotstar_driver_interfaces.msg import DotColor
 
from litur.tools.dynamics import ForwardDynamics
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

import numpy as np
import rbdl

class ShowReachability(Node):
    def __init__(self, model:str=""):
        super().__init__('show_reachability')
        self.dynamics = ForwardDynamics()
        self.timer_period = 0.01
        self.new_data = False
        self.fig = None
    #     self.publisher_ = self.create_publisher(String, 'topic', 10)
    #     timer_period = 0.5  # seconds
        self.timer_render = self.create_timer(self.timer_period, self.cb_render)
        # self.timer_data = self.create_timer(0.1, self.cb_data)
    #     self.i = 0
        # self.xdata = [float(x) / 100 for x in range(-100,101)]
        # self.ydata = len(self.xdata)*[0]
        
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        # self.ax.set_xlim([-1, 1])
        # self.ax.set_ylim([-1, 1])
        self.ax.set_xlabel('r1 (rad)')
        self.ax.set_ylabel('r2 (rad)')
        self.ax.set_zlabel('R1 Counter Torque')
        # self.plot, = self.ax.plot(self.xdata, self.ydata, 'r-')
        self.fig.add_axes(self.ax)
        self.fig.canvas.mpl_connect('close_event', self.fig_closed)
        self.fig.show()
        self.figure_shown = True

        if model:
            self.model = self.load_model(model)
        else:
            self.model = None


    def __del__(self):
        self.get_logger().info("Shutdown")
        self.destroy_node()
        if self.fig is not None:
            if self.figure_shown:
                plt.close(self.fig)

            self.fig = None
    

    def fig_closed(self, event):
        self.figure_shown = False
        self.get_logger().info('Figure closed')


    # def cb_data(self):
    #     t = self.get_clock().now()
    #     dt = t.nanoseconds / 1000000
    #     self.ydata = [math.sin(x)*math.cos(dt) + math.cos(x)*math.sin(dt) for x in self.xdata]
    #     self.new_data = True


    def cb_render(self):
        if not self.figure_shown:
            return

        if(self.new_data):
            # self.get_logger().info("Updating Plot")
            self.plot.set_ydata(self.ydata)
            self.fig.canvas.draw()
            self.new_data = False

        # Always have a slight pause to let window updates happen
        self.fig.canvas.flush_events()
        #plt.pause(self.timer_period/10)

        # msg = String()
        # msg.data = 'Hello World: %d' % self.i
        # self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        # self.i += 1


    def load_model(self, model):
        # Create a new model
        model = rbdl.Model()
        model.gravity = np.array([0, 0, -9.80665])

        # Create a joint from joint type
        joint_rot_y = rbdl.Joint.fromJointType("JointTypeRevoluteY")

        # Create a body for given mass, center of mass, and inertia at
        # the CoM
        body = rbdl.Body.fromMassComInertia(
            1., 
            np.array([0., 0., 0.5]),
            np.eye(3) * 0.05)
        xtrans= rbdl.SpatialTransform()
        xtrans.r = np.array([0., 0., 1.])

        # You can print all types
        # self.get_logger().info(str(model))
        # self.get_logger().info(str(body))
        # self.get_logger().info(str(body.mInertia))
        # self.get_logger().info(str(xtrans))

        # Construct the model
        body_1 = model.AppendBody(rbdl.SpatialTransform(), joint_rot_y, body)
        body_2 = model.AppendBody(xtrans, joint_rot_y, body)
        # body_3 = model.AppendBody(xtrans, joint_rot_y, body)
        # self.get_logger().info(str(model))

        return model


    def update_data(self):
        if self.model is None:
            return

        # Create numpy arrays for the state
        q = np.zeros(self.model.q_size)
        qdot = np.zeros(self.model.qdot_size)
        qddot = np.zeros(self.model.qdot_size)
        tau = np.zeros(self.model.qdot_size)

        samples = 11
        r_min = -math.pi/2
        r_max = math.pi/2
        r1 = np.linspace(r_min, r_max, samples) #[float(x) / 100 for x in range(-100,101)]
        r2 = np.linspace(r_min, r_max, samples) #[float(x) / 100 for x in range(-100,101)]
        t = np.empty((samples,samples))


        for i in range(0, samples):
            for j in range(0, samples):
                q[0] = r1[i]
                q[1] = r2[j]
                rbdl.InverseDynamics(self.model, q, qdot, qddot, tau)
                t[i][j] = tau[0]

        X,Y = np.meshgrid(r1, r2)
        # self.plot, = self.ax.plot(xdata, ydata, 'r-')
        self.plot = self.ax.plot_surface(X, Y, t, cmap=cm.coolwarm, antialiased=False)


def main(args=None):
    rclpy.init(args=args)

    sr = ShowReachability('test_model.urdf')
    sr.update_data()

    try:
        rclpy.spin(sr)
    except KeyboardInterrupt:
        pass

    del sr
    rclpy.shutdown()

if __name__ == '__main__':
    main()