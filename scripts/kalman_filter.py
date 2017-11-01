#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry


"""
	X- = FX + BU
	P- = FPF^T + Q
	K = (P-)H(H(P-)H^T + R)^-1
	X+ = X- +  K(z - HX-)
	P+ = (I-KH)P-
"""

current_time = 0
last_time = 0

x0 = np.mat([[None],[None],[None],[None]]) #1x4
p0 = np.mat([[None,None,None,None],[None,None,None,None],[None,None,None,None],[None,None,None,None]])

z = np.mat([[None],[None],[None],[None]])
R = np.mat(np.eye(len(x0)))

pose_ok = False
vel_ok = False
odom_ok = False

fix = -1
first_time = True


def pose_callback(data):
	global x0, p0, odom_ok, first_time
	rlon = data.pose.pose.position.x
	rlat = data.pose.pose.position.y
	rxvel = data.twist.twist.linear.x
	ryvel = data.twist.twist.linear.y
	x0[0,0] = rlon
	x0[1,0] = rxvel
	x0[2,0] = rlat
	x0[3,0] = ryvel
	if first_time:
		p0[0,0] = rxvel ** 2
		p0[2,2] = rxvel ** 2
		p0[1,1] = ryvel ** 2
		p0[3,3] = ryvel ** 2
		first_time = False

	odom_ok = True

def gps_callback(data):
	global state, R, pose_ok, fix
	z[0,0] = data.longitude
	z[2,0] = data.latitude
	R[0,0] = data.position_covariance[0]
	R[2,2] = data.position_covariance[4] 

	fix = data.status.status

	pose_ok = True

def vel_callback(data):
	global state, R, vel_ok
	z[1,0] = -data.twist.linear.y
	z[3,0] = data.twist.linear.x
	R[1,1] = data.twist.linear.y ** 2
	R[3,3] = data.twist.linear.x ** 2

	vel_ok = True

def get_F(dt):
	global x0
	F = np.mat(np.eye(len(x0)))
	for i in range(0,len(F),2):
		F[i,i+1] = dt
	return F

# def predict_state(x, f):

# 	return f*x

def predict_cov(p, q, f):
	
	FT = f.getT()
	P = f*p*FT + q
	return P

def predict(p, q = 0.01):
	global x0, p0
	global current_time, last_time

	current_time = rospy.get_rostime()
	current_time = current_time.secs + current_time.nsecs / 1000000000.0
	dt = current_time - last_time

	F = get_F(dt)

	# if x[0,0] == None:
	# 	x_out = state
	# else:
	# 	x_out = predict_state(x,F)

	# if p[0,0] == None:
	# 	p_out = np.mat(np.eye(len(x0))) * 0.00001
	# else:
	# Q = np.mat(np.eye(len(x0))) * q
	Q = np.mat([[0.25 * dt ** 4, 0.5 * dt ** 3],[0.5 * dt ** 3, dt**2]]) * q**2
	p_out = predict_cov(p, Q, F)
	

	last_time = rospy.get_rostime()
	last_time = last_time.secs + last_time.nsecs / 1000000000.0

	# return x_out, p_out
	return p_out

def get_gain(p, h, r):
	HPHT = h*p*h.getT()
	HPHTR = HPHT + r
	HPHTR_inv = HPHTR.getI()
	K = p*h.getT()*HPHTR_inv
	return K

def update_with_z(x, k, z, h):
	Y = z - (h*x)
	x_out = x + k*Y
	return x_out

def update_cov(k, h, p):
	I = np.mat(np.eye(len(p)))
	p_out = (I-k*h)*p
	return p_out

def correct(x, p, r, z):
	H = np.mat(np.eye(len(x)))
	K = get_gain(p, H, r)
	x_new = update_with_z(x, K, z, H)
	p_new = update_cov(K, H, p)
	return x_new, p_new

if __name__ == "__main__":
	rospy.init_node("kalman_filter")

	rospy.Subscriber("fix", NavSatFix, gps_callback)
	rospy.Subscriber("vel", TwistStamped, vel_callback)
	rospy.Subscriber("pose", Odometry, pose_callback)
	kf_pub = rospy.Publisher("kf", NavSatFix, queue_size=1)

	current_time = rospy.get_rostime()
	current_time = current_time.secs + current_time.nsecs / 1000000000.0
	last_time = rospy.get_rostime()
	last_time = last_time.secs + last_time.nsecs / 1000000000.0

	f = NavSatFix()
	r = rospy.Rate(20)

	try:
		while not rospy.is_shutdown():
			if fix >= 0:
				if pose_ok and vel_ok and odom_ok:
					pp = predict(p0)
					x2, p0 = correct(x0, pp, R, z)

					f.longitude = x2[0,0]
					f.latitude = x2[2,0]
					
			pose_ok = False
			vel_ok = False
			odom_ok = False
			f.status.status = fix
			kf_pub.publish(f)
			# r.sleep()
	except rospy.ROSInterruptException:
		pass