#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistStamped


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
p0 = np.mat([[None, None],[None, None]]) #2x2

state = np.mat([[None],[None],[None],[None]])
R = np.mat(np.eye(len(x0)))

pose_ok = False
vel_ok = False

fix = -1

def pose_callback(data):
	global state, R, pose_ok, fix
	state[0,0] = data.longitude
	state[2,0] = data.latitude
	R[0,0] = data.position_covariance[0]
	R[2,2] = data.position_covariance[4] 

	fix = data.status.status

	pose_ok = True

def vel_callback(data):
	global state, R, vel_ok
	state[1,0] = -data.twist.linear.y
	state[3,0] = data.twist.linear.x
	R[1,1] = data.twist.linear.y ** 2
	R[3,3] = data.twist.linear.x ** 2

	vel_ok = True

def predict_state(x, f):

	return f*x

def predict_cov(p, q, f):
	
	FT = f.getT()
	P = f*p*FT + q
	return P

def predict(x, p, q=0.00001):
	global x0
	global current_time, last_time

	current_time = rospy.get_rostime()
	current_time = current_time.secs + current_time.nsecs / 1000000000.0
	dt = current_time - last_time

	F = np.mat(np.eye(len(x0)))
	for i in range(0,len(F),2):
		F[i,i+1] = dt

	if x[0,0] == None:
		x_out = state
	else:
		x_out = predict_state(x,F)

	if p[0,0] == None:
		p_out = np.mat(np.eye(len(x0))) * 0.00001
	else:
		Q = np.mat(np.eye(len(x0))) * q
		p_out = predict_cov(p, Q, F)
	last_time = rospy.get_rostime()
	last_time = last_time.secs + last_time.nsecs / 1000000000.0
	return x_out, p_out

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

	rospy.Subscriber("fix", NavSatFix, pose_callback)
	rospy.Subscriber("vel", TwistStamped, vel_callback)
	kf_pub = rospy.Publisher("kf", NavSatFix, queue_size=1)

	current_time = rospy.get_rostime()
	current_time = current_time.secs + current_time.nsecs / 1000000000.0
	last_time = rospy.get_rostime()
	last_time = last_time.secs + last_time.nsecs / 1000000000.0

	f = NavSatFix()


	try:
		while not rospy.is_shutdown():
			if fix >= 0:
				if pose_ok and vel_ok:
					xx, pp = predict(x0, p0)
					x0, p0 = correct(xx, pp, R, state)

					f.longitude = x0[0,0]
					f.latitude = x0[2,0]
					
					pose_ok = False
					vel_ok = False
			f.status.status = fix
			kf_pub.publish(f)

	except rospy.ROSInterruptException:
		pass