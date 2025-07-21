import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import SMS_lightcurve_classes as SMSclasses
import matplotlib.ticker as ticker
from scipy import integrate
import matplotlib.colors as colors

############--------------------------------------------------------------------
############ plot parameters: --------------------------------------------------
myfontsize = 14


############ global variables: -------------------------------------------------
M_sun_gram = 1.988e33
R_sun_cm = 6.957e10
day = 60*60*24 # seconds in a day
year = 365.25*day
opacity = 0.35 # cm^2/g # opacity of primordial gas 
Tion = 6000.
m_Hydrogen = 1.6738e-24 # mean molecular wheight of atomic hydrogen in gram
c0 = 2.998e10 # speed of light in cm/s
h = 6.626176e-27 # in erg*s
kB = 1.38065e-16 # in erg/K
G = 6.674e-8 # gravitational constant [cm^3/g/s^2]
pc_in_cm = 3.0856775814914e18 # cm

############ run names: --------------------------------------------------------
# define run cases. use models from the paper for accreting SMSs:
#		[SMS Radius, Ekin_eje, Meje]
FujH1 = [1e4*R_sun_cm, 9.5e54, 23613*M_sun_gram]
FujH2 = [1e4*R_sun_cm, 5.1e55, 37255*M_sun_gram]
FujH4 = [1e4*R_sun_cm, 1.9e56, 82467*M_sun_gram]
FujDif1 = [1e4*R_sun_cm, 2.8e56, 110322*M_sun_gram]
FujHe1 = [1e4*R_sun_cm, 1.7e54, 5611*M_sun_gram]
FujHe2 = [1e4*R_sun_cm, 8.6e54, 8199*M_sun_gram]
FujHe4 = [1e4*R_sun_cm, 4.2e55, 19378*M_sun_gram]

NagCol1 = [1e4*R_sun_cm, 8.49e54, 29737*M_sun_gram]
NagCol2 = [1e4*R_sun_cm, 1.22e55, 96731*M_sun_gram]
NagPul1 = [1e4*R_sun_cm, 1.31e53, 3051*M_sun_gram]
NagPul2 = [1e4*R_sun_cm, 1.73e53, 4312*M_sun_gram]
NagExp = [1e4*R_sun_cm, 1.03e55, 298600*M_sun_gram]

############ helper functions: --------------------------------------------------
def hex_to_rgb(value):
	value = value.lstrip('#')
	lv = len(value)
	return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
def rgb_to_hex(rgb):
	return '%02x%02x%02x' % rgb
def mean_color(color1, color2):
	rgb1 = hex_to_rgb(color1)
	rgb2 = hex_to_rgb(color2)
	avg = lambda x, y: round((x+y) / 2)
	new_rgb = ()
	for i in range(len(rgb1)):
		new_rgb += (avg(rgb1[i], rgb2[i]),)
	return rgb_to_hex(new_rgb)

# return indices of time ticks:
def find_year_time_ticks(time_data, interval):
	t0 = time_data[0] #
	# iterate through the array ot find all points where 1 time interval has passed:
	time_data = np.array(time_data) - t0 #remove bias
	#
	num_ticks_max= 50
	tick_array = []
	for k in range(1,num_ticks_max,1):
		tmp_iteration_arr = np.array(time_data) - interval*k
		j=0
		for j in range(len(tmp_iteration_arr)):
			# iterate until we cross the interval value (i.e. until the array becomes positive):
			if tmp_iteration_arr[j] > 0.:
				tick_array.append(j)
				break
		if j > len(time_data)-2:
			break # this means we found all ticks
	return tick_array
		
def compute_xy_time_ticks(time_data, interval, xdada_in, ydata_in):
	time_tick_array =  find_year_time_ticks(time_data, interval)
	x_arr=np.zeros(len(time_tick_array))
	y_arr=np.zeros(len(time_tick_array))
	for index in range(len(time_tick_array)):
		x_arr[index] = xdada_in[time_tick_array[index]]
		y_arr[index] = ydata_in[time_tick_array[index]]
	return x_arr, y_arr


############--------------------------------------------------------------------
# Figure 1 of the SMS explosion summary
def SMS_explosion_summary_graphic():
	#
	def two_zone_circle(centre_pos=(1,1), r_inner = 0.25, r_outer = 1.0, cmap_inner_range=[0.18,0.35], cmap_outer_range=[0.62,0.82]):
		zone_points=100
		zone_inner=np.ndarray((zone_points,),dtype=plt.Circle)
		zone_outer=np.ndarray((zone_points,),dtype=plt.Circle)
		colors_core = mpl.pyplot.cm.jet(np.linspace(cmap_inner_range[0],cmap_inner_range[1],zone_points))
		colors_antmosphere = mpl.pyplot.cm.jet(np.linspace(cmap_outer_range[0],cmap_outer_range[1],zone_points))
		#
		#full_zones = np.array([plt.Circle(centre_pos, r_inner, fc="None", ec="black",lw=2),plt.Circle(centre_pos, r_outer, fc="None", ec="black",lw=2)])
		full_zones = np.array([plt.Circle(centre_pos, r_inner, fc="None", ec="None",lw=2)])
		#inner zone:
		for i in range(zone_points):
			zone_inner[i] = plt.Circle(centre_pos, i/(zone_points)*r_inner, color = colors_core[i])
		#outer zone:
		for i in range(zone_points):
			zone_outer[i] = plt.Circle(centre_pos, i/(zone_points)*(r_outer-r_inner)+r_inner, color = colors_antmosphere[i])
		# append both zones:
		full_zones = np.append(full_zones,zone_inner)
		full_zones = np.append(full_zones,zone_outer)
		# add black outlines:
		#full_zones.insert(0,[plt.Circle(centre_pos, r_inner, fc="None", ec="black",lw="3")])
		#full_zones.append([plt.Circle(centre_pos, r_outer, fc="None", ec="black",lw="3")])
		return full_zones
	#
	def polar_line_scramble(cente_xy=[1,1],rad_range=[0,np.pi], radius=0.5, amplitude=0.1, num_bumps=15.0):
		phi_arr = np.linspace(rad_range[0],rad_range[1],200)
		r_of_phi = np.zeros(len(phi_arr))
		# create value in polar:
		r_of_phi = radius*(1. + amplitude*np.exp(np.sin(num_bumps*phi_arr)) )
		# convert to cartesian:
		#x_arr = np.zeros(len(phi_arr))
		#y_arr = np.zeros(len(phi_arr))
		x_arr = r_of_phi*np.cos(phi_arr) + cente_xy[0]
		y_arr = r_of_phi*np.sin(phi_arr) + cente_xy[1]
		return x_arr, y_arr
	#
	def radial_accent_lines(cente_xy=[1,1],rad_range=[0,2*np.pi], radius_range=[0.1,1.0],num_lines=6):
		lines_arr = [None]*num_lines
		phi_arr = np.linspace(rad_range[0],rad_range[1],num_lines)
		for i in range(num_lines):
			# create line sone by one
			r_range = radius_range
			phi_value = [phi_arr[i],phi_arr[i]]
			# convert to cartesian
			x_arr = r_range*np.cos(phi_value) + cente_xy[0]
			y_arr = r_range*np.sin(phi_value) + cente_xy[1]
			lines_arr[i] = [x_arr, y_arr]
		return np.array(lines_arr)

	# color map:
	jet_cmap = mpl.pyplot.cm.jet(np.linspace(0,1,100))
	
	fig, ax = plt.subplots()
	fig.figsize=(6,3)
	ax.set_aspect('equal', 'box')

	###### PLOT PANEL 1:
	centre_pos_1 = (2.25,3.0)
	r_inner1=0.25; r_outer1=1
	circle_arr1 = two_zone_circle(centre_pos_1,r_inner1, r_outer1)
	for j in range(len(circle_arr1)-1,-1,-1):
		ax.add_patch(circle_arr1[j])
	# add black contours:
	ax.add_patch(plt.Circle(centre_pos_1, r_inner1+0.01, fc="None", ec="black",lw=0.5))
	ax.add_patch(plt.Circle(centre_pos_1, r_outer1, fc="None", ec="black",lw=0.8))

	###### PLOT PANEL 2:
	centre_pos_2 = (4.5,3.0)
	r_inner2=0.3; r_outer2=1
	circle_arr2 = two_zone_circle(centre_pos_2,r_inner2, r_outer2, [0.62,0.65], [0.65,0.82])
	for j in range(len(circle_arr2)-1,-1,-1):
		ax.add_patch(circle_arr2[j])
	# add black hole and outer line:
	ax.add_patch(plt.Circle(centre_pos_2, 0.06, fc="black", ec="black",lw=1))
	ax.add_patch(plt.Circle(centre_pos_2, r_outer2, fc="None", ec="black",lw=0.8))
	# add accent lines:
	accent_lines2 = radial_accent_lines([centre_pos_2[0],centre_pos_2[1]],[0,2*np.pi],[0.09,0.32],25)
	for i in range(len(accent_lines2)):
		plt.plot(accent_lines2[i][0], accent_lines2[i][1], c=jet_cmap[78], lw=0.4)
	# add explosion waves:
	scramble_x, scramble_y = polar_line_scramble([centre_pos_2[0],centre_pos_2[1]], [0,2.*np.pi], 0.305, amplitude=0.15)
	plt.plot(scramble_x,scramble_y,c="black", ls="--", lw=0.5)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_2[0],centre_pos_2[1]], [0,2.*np.pi], 0.3, amplitude=0.05)
	plt.plot(scramble_x,scramble_y,c="black", ls="--", lw=0.5)
	


	###### PLOT PANEL 3:
	centre_pos_3 = (6.75,3.)
	r_inner3=0.65; r_outer3=1
	circle_arr3 = two_zone_circle(centre_pos_3,r_inner3, r_outer3, [0.62,0.68], [0.68,0.84])
	for j in range(len(circle_arr3)-1,-1,-1):
		ax.add_patch(circle_arr3[j])
	# add black hole and outer line:
	ax.add_patch(plt.Circle(centre_pos_3, 0.06, fc="black", ec="black",lw=1))
	ax.add_patch(plt.Circle(centre_pos_3, r_outer3, fc="None", ec="black",lw=0.8))
	# add accent lines:
	accent_lines3 = radial_accent_lines([centre_pos_3[0],centre_pos_3[1]],[0,2*np.pi],[0.09,0.89],25)
	for i in range(len(accent_lines3)):
		plt.plot(accent_lines3[i][0], accent_lines3[i][1], c=jet_cmap[78], lw=0.4)
	# add explosion waves:
	scramble_x, scramble_y = polar_line_scramble([centre_pos_3[0],centre_pos_3[1]], [0,2.*np.pi], 0.85, amplitude=0.06,num_bumps=15)
	plt.plot(scramble_x,scramble_y,c="black", ls="--", lw=0.6)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_3[0],centre_pos_3[1]], [0,2.*np.pi], 0.83, amplitude=0.04,num_bumps=15)
	plt.plot(scramble_x,scramble_y,c="black", ls="--", lw=0.6)
	

	###### PLOT PANEL 4:
	centre_pos_4 = (9.0,3.)
	r_inner4=0.45; r_outer4=0.55
	circle_arr4_1 = two_zone_circle(centre_pos_4,0.5, 0.67, [0.62,0.68], [0.8,0.9])
	for j in range(len(circle_arr4_1)-1,-1,-1):
		ax.add_patch(circle_arr4_1[j])
	circle_arr4 = two_zone_circle(centre_pos_4,r_inner4, r_outer4, [0.62,0.68], [0.68,0.84])
	for j in range(len(circle_arr4)-1,-1,-1):
		ax.add_patch(circle_arr4[j])
	# add black hole and outer line:
	ax.add_patch(plt.Circle(centre_pos_4, 0.025, fc="black", ec="black",lw=1))
	ax.add_patch(plt.Circle(centre_pos_4, 0.68, fc="None", ec="black",lw=0.4))
	# add accent lines:
	accent_lines4 = radial_accent_lines([centre_pos_4[0],centre_pos_4[1]],[0,2*np.pi],[0.4975,0.67],41)
	for i in range(len(accent_lines4)):
		plt.plot(accent_lines4[i][0], accent_lines4[i][1], c=jet_cmap[73], lw=0.4)
	# add explosion waves:
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [0,2.*np.pi], 0.605, amplitude=0.045,num_bumps=15)
	plt.plot(scramble_x,scramble_y,c="black", ls="--", lw=0.5)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [0,2.*np.pi], 0.635, amplitude=0.025,num_bumps=15)
	plt.plot(scramble_x,scramble_y,c="black", ls="--", lw=0.5)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [0,2.*np.pi], 0.5, amplitude=0.018,num_bumps=15)
	plt.plot(scramble_x,scramble_y,c="black", ls="--", lw=0.5)
	# add CSM:
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [0.1*np.pi,0.7*np.pi], 0.75, amplitude=0.015,num_bumps=15)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.4)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [0.55*np.pi,0.9*np.pi], 0.71, amplitude=0.015,num_bumps=8)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.4)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [1.1*np.pi,1.5*np.pi], 0.79, amplitude=0.013,num_bumps=12)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.4)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [1.7*np.pi,1.95*np.pi], 0.705, amplitude=0.015,num_bumps=15)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.4)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [0.85*np.pi,1.3*np.pi], 0.75, amplitude=0.015,num_bumps=8)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.4)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [1.2*np.pi,1.65*np.pi], 0.715, amplitude=0.013,num_bumps=12)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.4)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [1.5*np.pi,2.05*np.pi], 0.75, amplitude=0.013,num_bumps=15)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.4)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [1.98*np.pi,2.4*np.pi], 0.71, amplitude=0.013,num_bumps=18)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.4)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [0.35*np.pi,1.4*np.pi], 0.83, amplitude=0.013,num_bumps=18)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.4)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [1.65*np.pi,2.28*np.pi], 0.83, amplitude=0.013,num_bumps=14)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.4)
	# CSM outer layer:
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [1.25*np.pi,1.78*np.pi], 0.89, amplitude=0.013,num_bumps=22)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.3)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [0.15*np.pi,1.18*np.pi], 0.895, amplitude=0.013,num_bumps=20)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.3)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [0.3*np.pi,0.85*np.pi], 0.95, amplitude=0.013,num_bumps=16)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.3)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [1.14*np.pi,1.7*np.pi], 0.95, amplitude=0.013,num_bumps=16)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.3)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [1.75*np.pi,2.2*np.pi], 0.95, amplitude=0.013,num_bumps=10)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.3)
	scramble_x, scramble_y = polar_line_scramble([centre_pos_4[0],centre_pos_4[1]], [0.1*np.pi,0.27*np.pi], 1.0, amplitude=0.013,num_bumps=7)
	plt.plot(scramble_x,scramble_y,c="green", ls="--", lw=0.3)

	# connecting lines PANEL 3 to 4:
	plt.plot([centre_pos_3[0]+0.15,centre_pos_4[0]],[centre_pos_3[1]+0.99,centre_pos_4[1]+0.52], ls=":", lw = 0.6, c="black")
	plt.plot([centre_pos_3[0]+0.15,centre_pos_4[0]],[centre_pos_3[1]-0.99,centre_pos_4[1]-0.52], ls=":", lw = 0.6, c="black")

	# add description texts:
	# upper texts:
	plt.text(centre_pos_1[0]-0.75,centre_pos_1[1]+1.85, "Phase 1: SMS\nbefore onset of\nthe GRI", fontsize=6, color="black", va="top",ha="left")
	plt.text(centre_pos_2[0]-0.75,centre_pos_2[1]+1.85, "Phase 2: After\ncore collapse and\nexplosion", fontsize=6, color="black", va="top",ha="left")
	plt.text(centre_pos_3[0]-0.75,centre_pos_3[1]+1.85, "Phase 3: Ejecta\nbreaks out of\ninitial SMS radius", fontsize=6, color="black", va="top",ha="left")
	plt.text(centre_pos_4[0]-0.75,centre_pos_4[1]+1.85, "Phase 4: After\nejecta break out,\nlight curve is visible", fontsize=6, color="black", va="top",ha="left")
	#lower texts:
	plt.text(centre_pos_1[0]-0.75,centre_pos_1[1]-1.15, "Bloated atmosphere\nof SMS has 10-80%\nof mass but 99% of\nradius", fontsize=6, color="black", va="top",ha="left")
	plt.text(centre_pos_2[0]-0.75,centre_pos_2[1]-1.15, "Energetic ejecta from\ncore collapse sweep\nthrough the SMS\natmosphere", fontsize=6, color="black", va="top",ha="left")
	plt.text(centre_pos_3[0]-0.75,centre_pos_3[1]-1.15, "SMS atmosphere is\nthermalised, becomes\nunbound and a part of\nfreely expanding ejecta", fontsize=6, color="black", va="top",ha="left")
	plt.text(centre_pos_4[0]-0.75,centre_pos_4[1]-1.15, "Total ejecta expands\nand interacts with the\nsurrounding CSM via\na shock", fontsize=6, color="black", va="top",ha="left")
	#inner texts:
	plt.text(centre_pos_1[0]-0.2,centre_pos_1[1]-0.35, "Core has 20-90%\nof SMS mass", fontsize=4, color="black", va="top",ha="left")
	plt.annotate(text="", xytext=(centre_pos_1[0]-0.18,centre_pos_1[1]-0.37), xy=(centre_pos_1[0]-0.05,centre_pos_1[1]-0.04), arrowprops=dict(arrowstyle='->'), color="black", fontsize=5)
	plt.annotate(text="", xytext=(centre_pos_1[0]-0.73,centre_pos_1[1]-1.17), xy=(centre_pos_1[0]-0.6,centre_pos_1[1]-0.5), arrowprops=dict(arrowstyle='->'), color="black", fontsize=5)
	plt.annotate(text="", xytext=(centre_pos_2[0]-0.73,centre_pos_2[1]-1.17), xy=(centre_pos_2[0]-0.17,centre_pos_2[1]-0.22), arrowprops=dict(arrowstyle='->'), color="black", fontsize=5)
	plt.text(centre_pos_2[0]+0.01,centre_pos_2[1]-0.08, "BH", fontsize=4, color="black", va="top",ha="left")
	plt.annotate(text="", xytext=(centre_pos_3[0]-0.73,centre_pos_3[1]-1.17), xy=(centre_pos_3[0]-0.40,centre_pos_3[1]-0.37), arrowprops=dict(arrowstyle='->'), color="black", fontsize=5)
	plt.text(centre_pos_3[0]+0.01,centre_pos_3[1]-0.08, "BH", fontsize=4, color="black", va="top",ha="left")
	plt.annotate(text="", xytext=(centre_pos_4[0]-0.73,centre_pos_4[1]-1.16), xy=(centre_pos_4[0]-0.18,centre_pos_4[1]-0.6), arrowprops=dict(arrowstyle='->'), color="black", fontsize=5)
	plt.text(centre_pos_4[0]-0.15,centre_pos_4[1]-0.15, "Shock", fontsize=4, color="black", va="top",ha="left")
	plt.annotate(text="", xytext=(centre_pos_4[0]-0.13,centre_pos_4[1]-0.20), xy=(centre_pos_4[0]-0.57,centre_pos_4[1]-0.29), arrowprops=dict(arrowstyle='->'), color="black", fontsize=4)
	#plt.annotate(text="", xytext=(centre_pos_4[0]+0.13,centre_pos_4[1]-0.18), xy=(centre_pos_4[0]+0.45,centre_pos_4[1]+0.45), arrowprops=dict(arrowstyle='->'), color="black", fontsize=4)

	plt.text(centre_pos_4[0]+0.55,centre_pos_4[1]+0.8, "Dense\nCSM", fontsize=4, color="green", va="bottom",ha="left")

	x_lim=[1,10.25]
	y_lim=[1,5]
	plt.plot([x_lim[0],x_lim[1],x_lim[1],x_lim[0],x_lim[0]], [y_lim[0],y_lim[0],y_lim[1],y_lim[1],y_lim[0]], c="black", lw=3) # make box surrounding the plot
	plt.xlim(x_lim[0],x_lim[1])
	plt.ylim(y_lim[0],y_lim[1])
	plt.axis('off')

	picturename = 'Fig1_SMS_explosion-sketch.pdf'
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.01) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)
	return 0


def density_distribution_summary_graphic():
	
	def density_line():
		Rrs = 0.4; iRrs = 0
		Rs = 0.47; iRs = 0
		Rfs = 0.54; iRfs = 0
		radius = np.linspace(0.01, 0.99, 700)
		density = np.zeros(len(radius))
		for k in range(len(radius)):
			r = radius[k]
			density[k] = 0.4 * (r+0.05)**(-0.15)
			if r > Rfs:
				density[k] = 0.25 * (0.5/r)**2
			elif r > Rs:
				density[k] = 0.7; iRfs = k
			elif r > Rrs:
				density[k] = 0.8; iRs = k
			else: iRrs = k
			
			
		return radius, density, iRrs, iRs, iRfs

	#
	fig, ax = plt.subplots()

	radius, density, iRrs, iRs, iRfs = density_line()
	plt.plot(radius, density, c= "black", lw= 2)
	# put some nice colour gradients:
	npts = len(radius)
	# ejecta:
	colourmap = mpl.pyplot.cm.jet(np.linspace(0.63,0.7,iRrs-1))
	##colourmap = mpl.cm.get_cmap('jet')
	#normalize = mpl.colors.Normalize(vmin=0, vmax=1)
	for i in range(0,iRrs-1):
		k=i
		plt.fill_between([radius[i], radius[i+1]],[density[i], density[i+1]], color=colourmap[k],alpha=0.15)
	# shock region:
	colourmap = mpl.pyplot.cm.jet(np.linspace(0.8,0.95,iRfs-iRrs-1))
	#normalize = mpl.colors.Normalize(vmin=0.1, vmax=0.2)
	for i in range(iRrs,iRfs-1):
		k = i-iRrs
		plt.fill_between([radius[i], radius[i+1]],[density[i], density[i+1]], color=colourmap[k],alpha=0.15)
	# CSM region:
	colourmap = mpl.pyplot.cm.jet(np.linspace(0.12,0.42,npts-iRfs-1))
	#normalize = mpl.colors.Normalize(vmin=0.0, vmax=1)
	for i in range(iRfs,npts-1):
		k = i-iRfs
		plt.fill_between([radius[i], radius[i+1]],[density[i], density[i+1]], color=colourmap[k],alpha=0.15)



	# ejecta density distribution:
	plt.text(0.07,0.4, r"$\rho_\mathrm{eje} \propto r^{- n_\mathrm{eje}}$", fontsize=12, color="black", va="bottom",ha="left")
	plt.text(0.07,0.3, r"$\beta_\mathrm{eje} \propto r$", fontsize=12, color="black", va="bottom",ha="left")
	plt.text(0.2,0.6, r"Ejecta", fontsize=12, fontweight="bold", color="black", va="bottom",ha="center")
	# reverse shock:
	plt.text(0.28,0.81, r"$\rho_\mathrm{rs}$", fontsize=12, color="black", va="bottom",ha="center")
	plt.plot([0.28, 0.4], [0.8, 0.8], ls="--", lw=1, c="black")
	# shock:
	plt.text(0.47,0.9, r"Shock shell", fontsize=12, fontweight="bold", color="black", va="bottom",ha="center")
	# forward shock:
	plt.text(0.67, 0.71, r"$\rho_\mathrm{fs}$", fontsize=12, color="black", va="bottom",ha="center")
	plt.plot([0.5, 0.67], [0.7, 0.7], ls="--", lw=1, c="black")
	# CSM :
	plt.text(0.78,0.12, r"$\rho_\mathrm{CSM} \propto r^{-2}$", fontsize=12, color="black", va="bottom",ha="left")
	plt.text(0.71,0.02, r"$\beta_\mathrm{CSM} \approx 0$", fontsize=12, color="black", va="bottom",ha="left")
	plt.text(0.8,0.3, r"CSM", fontsize=12, fontweight="bold", color="black", va="bottom",ha="center")
	#
	# radii:
	plt.text(0.4,0.3, r"$R_\mathrm{rs}$", fontsize=12, color="black", va="bottom",ha="center")
	plt.plot([0.4, 0.4], [0.35, 0.5], ls="--", lw=1, c="black")
	plt.plot([0.4, 0.4], [0.0, 0.298], ls="--", lw=1, c="black")
	plt.text(0.47,0.55, r"$R_\mathrm{s}$", fontsize=12, color="black", va="bottom",ha="center")
	plt.plot([0.47, 0.47], [0.6, 0.75], ls="--", lw=1, c="black")
	plt.plot([0.47, 0.47], [0.0, 0.55], ls="--", lw=1, c="black")
	plt.text(0.54,0.05, r"$R_\mathrm{fs}$", fontsize=12, color="black", va="bottom",ha="center")
	plt.plot([0.54, 0.54], [0.1, 0.3], ls="--", lw=1, c="black")
	plt.plot([0.54, 0.54], [0.0, 0.05], ls="--", lw=1, c="black")



	# plot the axes and axis labels:
	x_lim=[0,1]
	y_lim=[0,1]
	#
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2)
	ax.spines["left"].set_position(("data", 0))
	ax.spines["bottom"].set_position(("data", 0))
	# Hide the top and right spines.
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
	ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
	plt.xticks([]) # turn off axis ticks
	plt.yticks([])


	plt.xlim(x_lim[0],x_lim[1])
	plt.ylim(y_lim[0],y_lim[1])
	plt.xlabel("Radius", fontsize=18)
	plt.ylabel("Density", fontsize=18)

	picturename = 'Fig2_density_distribution_summary_sketch.pdf'
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.01) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)
	return 0




# single star color evolution in JWST filters:
def single_star_Luminosity_evolution(model_in, model_name_in):
	
	# choose which model to run:
	model_run = model_in
	model_name = model_name_in

	#-------------------------------------------
	R_0 = model_run[0]; E_kin_eje = model_run[1]; M_eje = model_run[2]
	#R_0 = 1e4*R_sun_cm; E_kin_eje = 1e53; M_eje = 2.6e5*M_sun_gram
	model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
	model.max_integration_time = 200.*year
	model.stop_after_opt_thick_phase = False
	model.integrate_model()
	#
	print(model.shock_opt_thick_Lbol_peak_luminosity, model.shock_opt_thick_Lbol_peak_time/year, model.shock_transparent_Lbol_initial_luminosity)
	#
	colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	#plt.plot(model.time_arr/year, model.Lbol_arr, linewidth=1.5, label= "Case H2-S: $E_{kin}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0) + "$cm$", ls="-", c="b")
	# make curve in transpacent phase dashed, select these values:
	time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
	time_arr_shock_transparent = model.time_arr[model.t_shock_transparent_index:-1]
	Lbol_arr_shock_opt_thick = model.Lbol_arr[0:model.t_shock_transparent_index]
	Lbol_arr_shock_transparent = model.Lbol_arr[model.t_shock_transparent_index:-1]

	#plt.plot(model.time_arr/year, model.Lbol_arr, linewidth=1.5, ls="-", c=colour_arr[5], label= "Case "+model_name+": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")

	plt.plot(time_arr_shock_opt_thick/year, Lbol_arr_shock_opt_thick, linewidth=1.5, ls="-", c="red", label= "Case "+model_name+": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
	plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls="-", c="orange", label= "Case "+model_name+": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")

	print(model.shock_opt_thick_Lbol_peak_variability_timescale/year)
	plt.hlines(y = model.shock_opt_thick_Lbol_peak_luminosity, color="blue", linestyle='-',xmin=(model.shock_opt_thick_Lbol_peak_time-model.shock_opt_thick_Lbol_peak_variability_timescale)/year,xmax=(model.shock_opt_thick_Lbol_peak_time) / year)
	#plt.hlines(y=1e45, xmin=4, xmax=10, label="test")
	#plt.plot([1,10], [1e45,1e45],label="test2")
	#plt.scatter([10],[1e45])

	plt.semilogy()
	#plt.loglog()
	plt.title(r"Rest-frame bolometric luminosity", fontsize=14) # add title to the whole plots
	plt.xlabel(r"$t_{source}$ [years]", fontsize=14)
	plt.ylabel(r"$L_{bol}$ [erg s$^{-1}$]", fontsize=14)
	plt.xlim(0,200) # time in yr
	#plt.ylim(5.65133e9,5.65135e9) # y-axis
	plt.ylim(5e41,5e46)

	plt.legend(loc="upper right", ncol=1, fontsize = 8)
	'''
	time_tick_array =  find_year_time_ticks(time_arr_shock_opt_thick, year)
	print(time_tick_array)
	x_arr=np.zeros(len(time_tick_array))
	y_arr=np.zeros(len(time_tick_array))
	for index in range(len(time_tick_array)):
		x_arr[index] = time_arr_shock_opt_thick[time_tick_array[index]]
		y_arr[index] = 1e45
	print(x_arr/year)
	print("t0=", time_arr_shock_opt_thick[0]/year)
	'''
	x_arr, y_arr = compute_xy_time_ticks(time_arr_shock_opt_thick, year, time_arr_shock_opt_thick, Lbol_arr_shock_opt_thick)
	plt.scatter(x_arr/year,y_arr, c="green")

	min_index = model.shock_opt_thick_Lbol_local_min_after_t0_index
	print(min_index)
	plt.scatter([model.time_arr[min_index]/year], [model.Lbol_arr[min_index]], marker="*")

	picturename = 'paper_plots/test/test_FujH1_Lbol.pdf'
	plt.show()
	#plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	#plt.close()
	print("Picture saved as: " + picturename)


def multiple_star_Luminosity_evolution(model_in_arr, model_name_in_arr, picname_in, legend_location="upper right", legend_ncol=1):
	
	# choose which model to run:
	model_run_arr = model_in_arr
	model_name_arr = model_name_in_arr
	#colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	colour_arr = ["#bf7fc8","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"] # choose better-distinguishable colours

	# add observational constraints
	xarr = [0.0, 1000.]
	yarr1 = [2*1.08e45, 2*1.08e45]
	yarr2 = [0.5*1.08e45, 0.5*1.08e45]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="sandybrown")
	plt.text(1.4e2, 0.56*1.08e45, r"GNz–11", fontsize=8,color="sandybrown")
	#
	xarr = [0.0, 1000.]
	yarr1 = [(1.0+0.5)*1e46, (1.0+0.5)*1e46]
	yarr2 = [(1.0-0.4)*1e46, (1.0-0.4)*1e46]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="darkorange")
	plt.text(1.4e2, (1.0-0.4+0.06)*1.0e46, r"GNZ9", fontsize=8,color="darkorange")

	#-------------------------------------------
	for i in range(len(model_run_arr)):
		R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
		model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
		model.max_integration_time = 410.*year
		model.stop_after_opt_thick_phase = False
		model.integrate_model()
		# make curve in transpacent phase dashed, select these values:
		time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
		time_arr_shock_transparent = model.time_arr[model.t_shock_transparent_index:-1]
		Lbol_arr_shock_opt_thick = model.Lbol_arr[0:model.t_shock_transparent_index]
		Lbol_arr_shock_transparent = model.Lbol_arr[model.t_shock_transparent_index:-1]
		plt.plot(time_arr_shock_opt_thick/year, Lbol_arr_shock_opt_thick, linewidth=1.5, ls="-", c=colour_arr[i], label= "Case "+model_name_arr[i])#+": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls=":", c=colour_arr[i])

	#plt.semilogy()
	plt.loglog()
	plt.title(r"Rest-frame bolometric luminosity", fontsize=14) # add title to the whole plots
	plt.xlabel(r"$t_{source}$ [years]", fontsize=14)
	plt.ylabel(r"$L_{bol}$ [erg s$^{-1}$]", fontsize=14)
	plt.xlim(8e-3,400) # time in yr
	#plt.ylim(5.65133e9,5.65135e9) # y-axis
	plt.ylim(7e41,7e47)

	plt.legend(loc=legend_location, ncol=legend_ncol, fontsize = 8)

	


	picturename = picname_in
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)


def multiple_star_accreted_vs_ionized_CSM(model_in_arr, model_name_in_arr, picname_in):
	# choose which model to run:
	model_run_arr = model_in_arr
	model_name_arr = model_name_in_arr
	#colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	colour_arr = ["#bf7fc8","#7f7fdd","#7faded","#7fd2da","#7fda7f","#ffad7f"] # choose better-distinguishable colours

	def radiance_integrand(nu_in, T_eff_in): # == radiance / photon energy
		B_nu_over_h =  2.*pow(nu_in,2) / pow(c0,2) / (np.exp(h*nu_in/ (kB*T_eff_in)) -1. ) # spectral radiance / (nu*h)
		return ( B_nu_over_h )
	
	def radiance_integrand2(nu_in, T_eff_in): # == radiance / photon energy
		B_nu =  2.*pow(nu_in,3)*h / pow(c0,2) / (np.exp(h*nu_in/ (kB*T_eff_in)) -1. ) # spectral radiance
		return ( B_nu )
	
	def radiance_excess_energy_integrand(nu_in, T_eff_in): # == radiance / photon energy
		B_nu =  2.*pow(nu_in,3)*h / pow(c0,2) / (np.exp(h*nu_in/ (kB*T_eff_in)) -1. ) # spectral radiance
		B_nu = B_nu - 2.16e-11 #13.6eV
		if B_nu <0: return 1e-20

		return ( B_nu)
	
	nu_min = 3.288e15 # frequency of 13.6 eV ionizong photon
	nu_min = 5.803e14 # frequency of 3.4 eV Balmer line photon
	nu_max = 3e18 # few orders of magnitude more (almost infinity)
	#-------------------------------------------
	for i in range(len(model_run_arr)):
		R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
		model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
		model.max_integration_time = 400.*year
		model.stop_after_opt_thick_phase = True
		model.integrate_model()
		#
		#print(model.Temp_eff_arr)
		#print(model.Temp_BB_surface)
		#print(model.Temp_eff_arr - model.Temp_BB_surface)
		#exit()
		#
		# compute number of UV photons:
		num_UV_photons_per_second = np.zeros(len(model.time_arr))
		energy_excess_UV_photons_per_second = np.zeros(len(model.time_arr))
		Luminosity_photons_over_UV = np.zeros(len(model.time_arr))
		for k in range(len(model.time_arr)):
			T_color = model.Temp_eff_arr[k]
			if T_color > 1e13: T_color=1e13
			radiance_integral_tmp, err1 = integrate.quad(radiance_integrand, nu_min, nu_max, args=(T_color), epsabs=1e-8, epsrel=1e-8)
			radiance_integral_tmp2, err1 = integrate.quad(radiance_integrand2, nu_min, nu_max, args=(T_color), epsabs=1e-8, epsrel=1e-8)
			radiance_integral_tmp = radiance_integral_tmp* min(1., (model.Temp_BB_surface[k]/T_color)**4) # correct for the shifted BB
			radiance_integral_tmp2 = radiance_integral_tmp2* min(1., (model.Temp_BB_surface[k]/T_color)**4) # correct for the shifted BB

			num_UV_photons_per_second[k] = radiance_integral_tmp * 4.*np.pi*(model.Rphotosphere_arr[k]**2)
			# totla energy of UV photons:
			ene_UV_photons = radiance_integral_tmp2 * 4.*np.pi*(model.Rphotosphere_arr[k]**2)
			Luminosity_photons_over_UV[k] = ene_UV_photons
			#energy_excess_UV_photons_per_second[k] = ene_UV_photons - 2.16e-11* num_UV_photons_per_second[k] # each photon loses 13.6eV to ionise an atom
			energy_excess_UV_photons_per_second[k] = ene_UV_photons - 5.45e-12* num_UV_photons_per_second[k] # each photon loses 3.4eV to ionise an atom
			if energy_excess_UV_photons_per_second[k] < 0.: energy_excess_UV_photons_per_second[k] = 0.
		# make curve in transpacent phase dashed, select these values:
		time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
		#time_arr_shock_transparent = model.time_arr[model.t_shock_transparent_index:-1]
		Num_ph_per_sec = num_UV_photons_per_second[0:model.t_shock_transparent_index]
		Ionized_CSM_mass_per_sec = Num_ph_per_sec * m_Hydrogen
		#
		Ene_excess_UV_photons_per_sec = energy_excess_UV_photons_per_second[0:model.t_shock_transparent_index]
		#
		#Lbol_arr_shock_transparent = model.Lbol_arr[model.t_shock_transparent_index:-1]
		# compute amount of accreted CSM mass:
		T_CSM = 8000. # Kelvin
		R_gas = 4.733e7 # specificgas constant for primordial gas in erg/g/K
		eta0 = 8.86
		accreted_mass_CSM = eta0*R_gas*T_CSM/G *( model.Rforwardshock_arr[0:model.t_shock_transparent_index] - R_0*np.ones(len(time_arr_shock_opt_thick)) )
		accreted_mass_CSM_per_sec = np.zeros(len(time_arr_shock_opt_thick))
		Ionized_CSM_mass_cumulative = np.zeros(len(time_arr_shock_opt_thick))
		Ene_excess_UV_photons_cumulative = np.zeros(len(time_arr_shock_opt_thick))
		for j in range(len(accreted_mass_CSM_per_sec)-1):
			accreted_mass_CSM_per_sec[j] = (accreted_mass_CSM[j+1] - accreted_mass_CSM[j]) / (time_arr_shock_opt_thick[j+1] - time_arr_shock_opt_thick[j] )
			Ionized_CSM_mass_cumulative[j+1] = Ionized_CSM_mass_cumulative[j] + Ionized_CSM_mass_per_sec[j] * (time_arr_shock_opt_thick[j+1] - time_arr_shock_opt_thick[j])
			#
			Ene_excess_UV_photons_cumulative[j+1] = Ene_excess_UV_photons_cumulative[j] + Ene_excess_UV_photons_per_sec[j] * (time_arr_shock_opt_thick[j+1] - time_arr_shock_opt_thick[j])


		#plt.plot(time_arr_shock_opt_thick/year, Ionized_CSM_mass_per_sec, linewidth=1.5, ls="-", c=colour_arr[i], label= "Case "+model_name_arr[i])#+": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		#plt.plot(time_arr_shock_opt_thick/year, accreted_mass_CSM_per_sec, linewidth=1.5, ls="--", c=colour_arr[i], label= "Case "+model_name_arr[i])#+": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		
		plt.plot(time_arr_shock_opt_thick/year, Ionized_CSM_mass_cumulative/M_sun_gram, linewidth=1.5, ls="-", c=colour_arr[i], label= "Case "+model_name_arr[i])#+": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		plt.plot(time_arr_shock_opt_thick/year, accreted_mass_CSM/M_sun_gram, linewidth=1.5, ls="--", c=colour_arr[i])#+": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		'''
		plt.plot(time_arr_shock_opt_thick/year, Ionized_CSM_mass_cumulative/m_Hydrogen, linewidth=1.5, ls=":", c=colour_arr[i], label="N_photons,tot")
		# estimate energy deposited into the ionising photons into hydrogen by photon excess energy:
		'''
		Hydrogen_kinetic_energyUV = (2.*Ene_excess_UV_photons_cumulative / Ionized_CSM_mass_cumulative )**(0.5) # velocity in cm/s
		##plt.plot(time_arr_shock_opt_thick/year, Ene_excess_UV_photons_cumulative, linewidth=1.5, label="Ekin,H", ls=":", c=colour_arr[i])
		##plt.plot(time_arr_shock_opt_thick/year, Hydrogen_kinetic_energyUV, linewidth=1.5, label="v,H", ls="-.", c=colour_arr[i])
		#plt.plot(time_arr_shock_opt_thick/year, model.Rforwardshock_arr[0:model.t_shock_transparent_index]/3e18, linewidth=1.5, label="Rs", ls="-.", c=colour_arr[i])
		# estimate velocity from momentum:
		vel_momentum = time_arr_shock_opt_thick*Luminosity_photons_over_UV[0:model.t_shock_transparent_index] / c0 /Ionized_CSM_mass_cumulative
		##plt.plot(time_arr_shock_opt_thick/year, vel_momentum, linewidth=1.5, label="v_ph", ls="-.", c=colour_arr[i])
		

		# model fully ionized CSM and estimate tau off of this:
		print("for model: ", model_name_arr[i])
		Rph = opacity * eta0*R_gas*T_CSM/(4.*np.pi*G) * np.ones(len(time_arr_shock_opt_thick))
		Rs = model.Rforwardshock_arr[0:model.t_shock_transparent_index] # Rs at the end
		tau_ph_s = Rph/Rs - 1.*np.ones(len(time_arr_shock_opt_thick))
		t_diff = (Rph-Rs) * 3.* tau_ph_s / c0
		#t_diff_2 = opacity * (Ionized_CSM_mass_cumulative - accreted_mass_CSM) / (4.*np.pi*c0* Rs)
		#print("tau_ph,s = ", tau_ph_s)
		#print("t_diff = ", t_diff/year)
		#plt.plot(time_arr_shock_opt_thick/year, tau_ph_s, ls="-.", c=colour_arr[i] , label= "tau "+model_name_arr[i])
		#plt.plot(time_arr_shock_opt_thick/year, t_diff/year, ls=":", c=colour_arr[i] , label= "td1 "+model_name_arr[i])
		#plt.plot(time_arr_shock_opt_thick/year, t_diff_2/time_arr_shock_opt_thick, ls="--", c=colour_arr[i] , label= "td2 "+model_name_arr[i])


		#plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls=":", c=colour_arr[i])
	# add solid/dashed liene indicator:
	plt.plot([-10,-10],[-10,-10],linewidth=1.5, ls="-",c="black",label=r"$M_\mathrm{CSM,ion}$")
	plt.plot([-10,-10],[-10,-10],linewidth=1.5, ls="--",c="black",label=r"$M_\mathrm{CSM,acc}$")

	#plt.semilogy()
	plt.loglog()
	plt.title(r"Ionised vs. accreted CSM matter", fontsize=14) # add title to the whole plots
	plt.xlabel(r"$t_{source}$ [years]", fontsize=14)
	plt.ylabel(r"$M_\mathrm{CSM,ion}\:,\: M_\mathrm{CSM,acc}$ [$M_\odot$]", fontsize=14)
	plt.xlim(1e-2,2e2) # time in yr
	#plt.ylim(5.65133e9,5.65135e9) # y-axis
	plt.ylim(1,1e9)

	plt.legend(loc="lower right", ncol=1, fontsize = 8)

	
	picturename = picname_in
	plt.show()
	#plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	#plt.close()
	print("Picture saved as: " + picturename)


def multiple_star_StroemgenSphere_CSM(model_in_arr, model_name_in_arr, picname_in):
	# choose which model to run:
	model_run_arr = model_in_arr
	model_name_arr = model_name_in_arr
	#colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	colour_arr = ["#bf7fc8","#7f7fdd","#7faded","#7fd2da","#7fda7f","#ffad7f"] # choose better-distinguishable colours

	def radiance_integrand(nu_in, T_eff_in): # == radiance / photon energy
		B_nu_over_h =  2.*pow(nu_in,2) / pow(c0,2) / (np.exp(h*nu_in/ (kB*T_eff_in)) -1. ) # spectral radiance / (nu*h)
		return ( B_nu_over_h )
	
	#-------------------------------------------
	# LYMAN CONTINUUM PHOTONS:
	
	#nu_min = 3.288e15 # frequency of 13.6 eV ionizong photon  (should be 10.2 eV for Lyman-alpha ionisation!!!)
	nu_min = 5.803e14 # frequency of 3.4 eV Balmer line photon
	nu_max = 3e18 # few orders of magnitude more (essentially infinity)
	
	
	for i in range(len(model_run_arr)):
		R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
		model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
		model.max_integration_time = 400.*year
		model.max_timestep=5.*day
		model.stop_after_opt_thick_phase = True
		model.integrate_model()
		#
		#
		# compute number rate of UV photons:
		num_UV_photons_per_second = np.zeros(len(model.time_arr))
		energy_excess_UV_photons_per_second = np.zeros(len(model.time_arr))
		Luminosity_photons_over_UV = np.zeros(len(model.time_arr))
		for k in range(len(model.time_arr)):
			T_color = model.Temp_eff_arr[k]
			if T_color > 1e13: T_color=1e13
			radiance_integral_tmp, err1 = integrate.quad(radiance_integrand, nu_min, nu_max, args=(T_color), epsabs=1e-8, epsrel=1e-8)
			radiance_integral_tmp = radiance_integral_tmp* min(1., (model.Temp_BB_surface[k]/T_color)**4) # correct for the shifted BB

			num_UV_photons_per_second[k] = radiance_integral_tmp * 4.*np.pi*(model.Rphotosphere_arr[k]**2) # Photons/s
		
		# make curve in transpacent phase dashed, select these values:
		time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
		Q_ion_photon = num_UV_photons_per_second[0:model.t_shock_transparent_index]
		R_stroemgen = np.ones(model.t_shock_transparent_index)
		R_photosphere = model.Rphotosphere_arr[0:model.t_shock_transparent_index]
		tau_recomb = np.ones(model.t_shock_transparent_index)
		R_recomb_max = np.ones(model.t_shock_transparent_index) # maximum radius until recombination happens
		M_ion_kyohei = np.ones(model.t_shock_transparent_index) # estimate for ionised mass, based on ionisation timescale
		M_ion_kyohei_mod = np.ones(model.t_shock_transparent_index)
		M_Rstr = np.ones(model.t_shock_transparent_index) # estimate for ionised mass, mass inside Strömgen radius
		# Values for kyoheis balance equation for Balmer ionisation:
		f_ph = np.ones(model.t_shock_transparent_index)

		#
		T_CSM = 8000. # Kelvin
		R_gas = 4.733e7 # specificgas constant for primordial gas in erg/g/K
		eta0 = 8.86
		alpha_recomb = 2.425e-13  #3.355e-13 #4.54e-13 # recombination factor for n=2 hydrogen at 8000K, cm^3/s
		Larson_prefactor = 4.*np.pi*alpha_recomb*(eta0*R_gas*T_CSM/(4.*np.pi*G) / m_Hydrogen)**2 # for number density_profile n(r) = n0 * (r0/r)^2
		Num_n_prefactor = eta0*R_gas*T_CSM/(4.*np.pi*G) / m_Hydrogen
		print("Larson_prefactor =", Larson_prefactor)
		print("Num_n_prefactor =", Num_n_prefactor*m_Hydrogen)
		f_gamma = 2.93* Q_ion_photon / (Num_n_prefactor*4.*np.pi*c0)
		frac_ion = - 0.5*f_gamma + np.sqrt( (0.5*f_gamma)**2 + f_gamma)
		Q_treshold = (frac_ion**2) *Larson_prefactor/R_photosphere
		for j in range(1,len(R_stroemgen)):
			if Q_ion_photon[j] > Q_treshold[j]: 
				R_stroemgen[j] = 1e10*pc_in_cm # essentially infinity
				tau_recomb[j] = 1e5*year # seconds, essentially infinity
				R_recomb_max[j] = 1e-8*year
			else: 
				R_stroemgen[j] = R_photosphere[j] / (1. - Q_ion_photon[j]/Q_treshold[j])
				tau_recomb[j] = (  alpha_recomb * (Num_n_prefactor / (R_photosphere[j])**2)  * (1. - Q_ion_photon[j]/Q_treshold[j])  )**(-1)
				R_recomb_max[j] = np.sqrt( alpha_recomb* time_arr_shock_opt_thick[j] * Num_n_prefactor * (1. - Q_ion_photon[j]/Q_treshold[j]) )
			#
			#M_ion_kyohei_mod[j] = M_ion_kyohei_mod[j-1] + m_Hydrogen* Q_ion_photon[j] * (min(tau_recomb[j], time_arr_shock_opt_thick[j])- min(tau_recomb[j-1], time_arr_shock_opt_thick[j-1]))
			M_ion_kyohei_mod[j] = M_ion_kyohei_mod[j-1] + m_Hydrogen* Q_ion_photon[j] * (time_arr_shock_opt_thick[j] - time_arr_shock_opt_thick[j-1])
			M_ion_kyohei[j] =   m_Hydrogen* Q_ion_photon[j] * min(tau_recomb[j], time_arr_shock_opt_thick[j])
			M_Rstr[j] = 4.*np.pi*Num_n_prefactor * (R_stroemgen[j] - R_photosphere[j]) * m_Hydrogen
			#
			f_ph[j] = Q_ion_photon[j] / (Num_n_prefactor * 4.*np.pi* c0)



		plt.plot(time_arr_shock_opt_thick[5:]/year, R_stroemgen[5:]/pc_in_cm, linewidth=1.5, ls="-", c=colour_arr[i], label= model_name_arr[i])#+": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		plt.plot(time_arr_shock_opt_thick[5:]/year, R_photosphere[5:]/pc_in_cm, linewidth=1.5, ls="--", c=colour_arr[i])#+": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		#
		#plt.plot(time_arr_shock_opt_thick/year, tau_recomb/year, linewidth=1.5, ls=":", c=colour_arr[i])
		#
		#plt.plot(time_arr_shock_opt_thick/year, R_recomb_max/pc_in_cm, linewidth=1.5, ls="-.", c=colour_arr[i])
		#
		#plt.plot(time_arr_shock_opt_thick/year, M_ion_kyohei/M_sun_gram, linewidth=2., ls="-", c=colour_arr[i])
		#plt.plot(time_arr_shock_opt_thick/year, M_ion_kyohei_mod/M_sun_gram, linewidth=2., ls=":", c="black")
		#plt.plot(time_arr_shock_opt_thick/year, M_Rstr/M_sun_gram, linewidth=3, ls="--", c=colour_arr[i])
		# Koyheis balance equation:
		#plt.plot(time_arr_shock_opt_thick/year, f_ph, linewidth=1.5, ls="--", c="black")


		#plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls=":", c=colour_arr[i])
	# add solid/dashed line indicator:
	plt.plot([-10,-10],[-10,-10],linewidth=1.5, ls="-",c="black",label=r"$R_\mathrm{str}$")
	plt.plot([-10,-10],[-10,-10],linewidth=1.5, ls="--",c="black",label=r"$R_\mathrm{fs}$")
	#plt.plot([-10,-10],[-10,-10],linewidth=1.5, ls=":",c="black",label=r"$\tau_\mathrm{rec}$")
	#plt.plot([-10,-10],[-10,-10],linewidth=1.5, ls="-.",c="black",label=r"$R_\mathrm{rec}$")

	#plt.semilogy()
	plt.loglog()
	plt.title(r"Strömgen sphere and shock radius", fontsize=14) # add title to the whole plots
	plt.xlabel(r"$t_{source}$ [years]", fontsize=14)
	plt.ylabel(r"$R_\mathrm{str}$, $R_\mathrm{fs}$ [pc]", fontsize=14)
	plt.xlim(1e-2,2e2) # time in yr
	plt.ylim(1e-4,1e3) # radius in pc

	plt.legend(loc="upper left", ncol=1, fontsize = 10)

	
	picturename = picname_in
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.1) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)


def multiple_star_Luminosity_evolution_systematics_vary_Ekin(picname_in, xrange=[1e-3,3e2],yrange=[1e42,1e48]):
	
	# choose which model to run:
	num_models = 8
	model_run_arr = [None]*num_models
	model_name_arr = [None]*num_models
	colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	#colour_arr = ["#bf7fc8","#7f7fdd","#7faded","#7fd2da","#7fda7f","#ffad7f"] # choose better-distinguishable colours
	# init models:
	for o in range(num_models):
		model_run_arr[o] = [1e4*R_sun_cm, 10**(53+0.5*o), 1e4*M_sun_gram]
		model_name_arr[o] = "Model"
	#model_name_arr[0] = r"$E_{kin,eje} = 10^{53.0}\,$erg, $M_{eje} = 10^4\,M_\odot, R_{0} = 10^4\,R_\odot$"
	model_name_arr[0] = r"$E_\mathrm{kin,eje} = 10^{53.0}\,$erg"
	model_name_arr[1] = r"$E_\mathrm{kin,eje} = 10^{53.5}\,$erg"
	model_name_arr[2] = r"$E_\mathrm{kin,eje} = 10^{54.0}\,$erg"
	model_name_arr[3] = r"$E_\mathrm{kin,eje} = 10^{54.5}\,$erg"
	model_name_arr[4] = r"$E_\mathrm{kin,eje} = 10^{55.0}\,$erg"
	model_name_arr[5] = r"$E_\mathrm{kin,eje} = 10^{55.5}\,$erg"
	model_name_arr[6] = r"$E_\mathrm{kin,eje} = 10^{56.0}\,$erg"
	model_name_arr[7] = r"$E_\mathrm{kin,eje} = 10^{56.5}\,$erg"

	# add observational constraints
	'''
	xarr = [0.0, 1000.]
	yarr1 = [2*1.08e45, 2*1.08e45]
	yarr2 = [0.5*1.08e45, 0.5*1.08e45]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="sandybrown")
	plt.text(1.4e2, 0.56*1.08e45, r"GNz–11", fontsize=8,color="sandybrown")
	#
	xarr = [0.0, 1000.]
	yarr1 = [(1.0+0.5)*1e46, (1.0+0.5)*1e46]
	yarr2 = [(1.0-0.4)*1e46, (1.0-0.4)*1e46]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="darkorange")
	plt.text(1.4e2, (1.0-0.4+0.06)*1.0e46, r"GNZ9", fontsize=8,color="darkorange")
	'''
	#-------------------------------------------
	for i in range(len(model_run_arr)):
		R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
		model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
		model.max_integration_time = 310.*year
		model.stop_after_opt_thick_phase = False
		model.force_turn_off_non_thermal_effects = True
		model.integrate_model()
		# make curve in transpacent phase dashed, select these values:
		time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
		time_arr_shock_transparent = model.time_arr[model.t_shock_transparent_index:-1]
		Lbol_arr_shock_opt_thick = model.Lbol_arr[0:model.t_shock_transparent_index]
		Lbol_arr_shock_transparent = model.Lbol_arr[model.t_shock_transparent_index:-1]
		plt.plot(time_arr_shock_opt_thick/year, Lbol_arr_shock_opt_thick, linewidth=1.5, ls="-", c=colour_arr[i], label= "Case: "+ model_name_arr[i])#": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls=":", c=colour_arr[i])

	plt.plot([-100,-100],[-100,-100], c="white", label="$M_\mathrm{eje} = 10^4\,M_\odot, R_{0} = 10^4\,R_\odot$")
	#plt.semilogy()
	plt.loglog()
	plt.title(r"Rest-frame bolometric luminosity, varying $E_\mathrm{kin,eje}$", fontsize=14) # add title to the whole plots
	plt.xlabel(r"$t_{source}$ [years]", fontsize=14)
	plt.ylabel(r"$L_{bol}$ [erg s$^{-1}$]", fontsize=14)
	plt.xlim(xrange[0],xrange[1]) # time in yr
	#plt.ylim(5.65133e9,5.65135e9) # y-axis
	plt.ylim(yrange[0],yrange[1])

	plt.legend(loc="upper right", ncol=1, fontsize = 6)

	picturename = picname_in
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)


def multiple_star_Luminosity_evolution_systematics_vary_Meje(picname_in, xrange=[1e-3,3e2],yrange=[1e42,1e48]):
	
	# choose which model to run:
	num_models = 8
	model_run_arr = [None]*num_models
	model_name_arr = [None]*num_models
	colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	#colour_arr = ["#bf7fc8","#7f7fdd","#7faded","#7fd2da","#7fda7f","#ffad7f"] # choose better-distinguishable colours
	# init models:
	for o in range(num_models):
		model_run_arr[o] = [1e4*R_sun_cm, 10**(55), (10**(3+0.4*o))*M_sun_gram]
		model_name_arr[o] = "Model"
	#model_name_arr[0] = r"$E_{kin,eje} = 10^{53.0}\,$erg, $M_{eje} = 10^4\,M_\odot, R_{0} = 10^4\,R_\odot$"
	model_name_arr[0] = r"$M_\mathrm{eje} = 10^{3.0}\,M_\odot$"
	model_name_arr[1] = r"$M_\mathrm{eje} = 10^{3.4}\,M_\odot$"
	model_name_arr[2] = r"$M_\mathrm{eje} = 10^{3.8}\,M_\odot$"
	model_name_arr[3] = r"$M_\mathrm{eje} = 10^{4.2}\,M_\odot$"
	model_name_arr[4] = r"$M_\mathrm{eje} = 10^{4.6}\,M_\odot$"
	model_name_arr[5] = r"$M_\mathrm{eje} = 10^{5.0}\,M_\odot$"
	model_name_arr[6] = r"$M_\mathrm{eje} = 10^{5.4}\,M_\odot$"
	model_name_arr[7] = r"$M_\mathrm{eje} = 10^{5.8}\,M_\odot$"

	# add observational constraints
	'''
	xarr = [0.0, 1000.]
	yarr1 = [2*1.08e45, 2*1.08e45]
	yarr2 = [0.5*1.08e45, 0.5*1.08e45]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="sandybrown")
	plt.text(1.4e2, 0.56*1.08e45, r"GNz–11", fontsize=8,color="sandybrown")
	#
	xarr = [0.0, 1000.]
	yarr1 = [(1.0+0.5)*1e46, (1.0+0.5)*1e46]
	yarr2 = [(1.0-0.4)*1e46, (1.0-0.4)*1e46]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="darkorange")
	plt.text(1.4e2, (1.0-0.4+0.06)*1.0e46, r"GNZ9", fontsize=8,color="darkorange")
	'''
	#-------------------------------------------
	for i in range(len(model_run_arr)):
		R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
		model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
		model.max_integration_time = 710.*year
		model.stop_after_opt_thick_phase = False
		model.force_turn_off_non_thermal_effects = True
		model.integrate_model()
		# make curve in transpacent phase dashed, select these values:
		time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
		time_arr_shock_transparent = model.time_arr[model.t_shock_transparent_index:-1]
		Lbol_arr_shock_opt_thick = model.Lbol_arr[0:model.t_shock_transparent_index]
		Lbol_arr_shock_transparent = model.Lbol_arr[model.t_shock_transparent_index:-1]
		plt.plot(time_arr_shock_opt_thick/year, Lbol_arr_shock_opt_thick, linewidth=1.5, ls="-", c=colour_arr[i], label= "Case: "+ model_name_arr[i])#": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls=":", c=colour_arr[i])

	plt.plot([-100,-100],[-100,-100], c="white", label="$E_\mathrm{kin,eje} = 10^{55}\,$erg$, R_{0} = 10^4\,R_\odot$")
	#plt.semilogy()
	plt.loglog()
	plt.title(r"Rest-frame bolometric luminosity, varying $M_\mathrm{eje}$", fontsize=14) # add title to the whole plots
	plt.xlabel(r"$t_{source}$ [years]", fontsize=14)
	plt.ylabel(r"$L_{bol}$ [erg s$^{-1}$]", fontsize=14)
	plt.xlim(xrange[0],xrange[1]) # time in yr
	#plt.ylim(5.65133e9,5.65135e9) # y-axis
	plt.ylim(yrange[0],yrange[1])

	plt.legend(loc="upper right", ncol=1, fontsize = 6)

	picturename = picname_in
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)


def multiple_star_Luminosity_evolution_systematics_vary_R0(picname_in, xrange=[1e-3,3e2],yrange=[1e42,1e48]):
	
	# choose which model to run:
	num_models = 8
	model_run_arr = [None]*num_models
	model_name_arr = [None]*num_models
	colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	#colour_arr = ["#bf7fc8","#7f7fdd","#7faded","#7fd2da","#7fda7f","#ffad7f"] # choose better-distinguishable colours
	# init models:
	for o in range(num_models):
		model_run_arr[o] = [(10**(2+0.4*o))*R_sun_cm, 10**(55), 1e4*M_sun_gram]
		model_name_arr[o] = "Model"
	#model_name_arr[0] = r"$E_{kin,eje} = 10^{53.0}\,$erg, $M_{eje} = 10^4\,M_\odot, R_{0} = 10^4\,R_\odot$"
	model_name_arr[0] = r"$R_{0} = 10^{2.0}\,R_\odot$"
	model_name_arr[1] = r"$R_{0} = 10^{2.4}\,R_\odot$"
	model_name_arr[2] = r"$R_{0} = 10^{2.8}\,R_\odot$"
	model_name_arr[3] = r"$R_{0} = 10^{3.2}\,R_\odot$"
	model_name_arr[4] = r"$R_{0} = 10^{3.6}\,R_\odot$"
	model_name_arr[5] = r"$R_{0} = 10^{4.0}\,R_\odot$"
	model_name_arr[6] = r"$R_{0} = 10^{4.4}\,R_\odot$"
	model_name_arr[7] = r"$R_{0} = 10^{4.8}\,R_\odot$"

	# add observational constraints
	'''
	xarr = [0.0, 1000.]
	yarr1 = [2*1.08e45, 2*1.08e45]
	yarr2 = [0.5*1.08e45, 0.5*1.08e45]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="sandybrown")
	plt.text(1.4e2, 0.56*1.08e45, r"GNz–11", fontsize=8,color="sandybrown")
	#
	xarr = [0.0, 1000.]
	yarr1 = [(1.0+0.5)*1e46, (1.0+0.5)*1e46]
	yarr2 = [(1.0-0.4)*1e46, (1.0-0.4)*1e46]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="darkorange")
	plt.text(1.4e2, (1.0-0.4+0.06)*1.0e46, r"GNZ9", fontsize=8,color="darkorange")
	'''
	#-------------------------------------------
	for i in range(len(model_run_arr)):
		R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
		model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
		model.max_integration_time = 210.*year
		model.stop_after_opt_thick_phase = False
		model.force_turn_off_non_thermal_effects = True
		model.integrate_model()
		# make curve in transpacent phase dashed, select these values:
		time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
		time_arr_shock_transparent = model.time_arr[model.t_shock_transparent_index:-1]
		Lbol_arr_shock_opt_thick = model.Lbol_arr[0:model.t_shock_transparent_index]
		Lbol_arr_shock_transparent = model.Lbol_arr[model.t_shock_transparent_index:-1]
		plt.plot(time_arr_shock_opt_thick/year, Lbol_arr_shock_opt_thick, linewidth=1.5, ls="-", c=colour_arr[i], label= "Case: "+ model_name_arr[i])#": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls=":", c=colour_arr[i])

	plt.plot([-100,-100],[-100,-100], c="white", label="$E_\mathrm{kin,eje} = 10^{55}\,$erg$,$\n$ M_\mathrm{eje} = 10^4\,M_\odot$")
	#plt.semilogy()
	plt.loglog()
	plt.title(r"Rest-frame bolometric luminosity, varying $R_0$", fontsize=14) # add title to the whole plots
	plt.xlabel(r"$t_{source}$ [years]", fontsize=14)
	plt.ylabel(r"$L_{bol}$ [erg s$^{-1}$]", fontsize=14)
	plt.xlim(xrange[0],xrange[1]) # time in yr
	#plt.ylim(5.65133e9,5.65135e9) # y-axis
	plt.ylim(yrange[0],yrange[1])

	plt.legend(loc="upper right", ncol=1, fontsize = 6)

	picturename = picname_in
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)

# vary density distribution of the ejecta
def multiple_star_Luminosity_evolution_systematics_vary_n_density(picname_in, xrange=[1e-3,3e2],yrange=[1e42,1e48]):
	
	# choose which model to run:
	num_models = 6
	model_run_arr = [None]*num_models
	model_name_arr = [None]*num_models
	model_n_density_arr = [None]*num_models
	#colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	colour_arr = ["#bf7fc8","#7f7fdd","#7faded","#7fd2da","#7fda7f","#ffad7f"] # choose better-distinguishable colours
	# init models:
	for o in range(num_models):
		model_run_arr[o] = [1e4*R_sun_cm, 1e54, 1e3*M_sun_gram]
		model_name_arr[o] = "Model"
		model_n_density_arr[o] = o
	#model_name_arr[0] = r"$n_\mathrm{eje} = 0, E_{kin,eje} = 10^{55}\,$erg, $M_{eje} = 10^4\,M_\odot, R_{0} = 10^4\,R_\odot$"
	model_name_arr[0] = r"$n_\mathrm{eje} = 0$"
	model_name_arr[1] = r"$n_\mathrm{eje} = 1$"
	model_name_arr[2] = r"$n_\mathrm{eje} = 2$"
	#-------------------------------------------
	for i in range(3):
		R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
		model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity, n_density_in=model_n_density_arr[i])
		model.max_integration_time = 210.*year
		model.stop_after_opt_thick_phase = False
		model.force_turn_off_non_thermal_effects = True
		model.integrate_model()
		# make curve in transpacent phase dashed, select these values:
		time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
		time_arr_shock_transparent = model.time_arr[model.t_shock_transparent_index:-1]
		Lbol_arr_shock_opt_thick = model.Lbol_arr[0:model.t_shock_transparent_index]
		Lbol_arr_shock_transparent = model.Lbol_arr[model.t_shock_transparent_index:-1]
		plt.plot(time_arr_shock_opt_thick/year, Lbol_arr_shock_opt_thick, linewidth=1.5, ls="-", c=colour_arr[i], label= "Case: "+ model_name_arr[i])#": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls=":", c=colour_arr[i])
	plt.plot([-100,-100],[-100,-100], c="white", label="$R_{0} = 10^4\,R_\odot,$\n$E_\mathrm{kin,eje} = 10^{54}\,$erg$,$\n$ M_\mathrm{eje} = 10^3\,M_\odot$")
	plt.plot([-100,-100],[-100,-100], c="white", label=r"$––––––$") # separator between the two cases


	for o in range(3,num_models):
		model_run_arr[o] = [1e4*R_sun_cm, 1e56, 1e5*M_sun_gram]
		model_name_arr[o] = "Model"
		model_n_density_arr[o] = o-3
	model_name_arr[3] = r"$n_\mathrm{eje} = 0$"
	model_name_arr[4] = r"$n_\mathrm{eje} = 1$"
	model_name_arr[5] = r"$n_\mathrm{eje} = 2$"
	#-------------------------------------------
	for i in range(3,len(model_run_arr)):
		R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
		model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity, n_density_in=model_n_density_arr[i])
		model.max_integration_time = 210.*year
		model.stop_after_opt_thick_phase = False
		model.force_turn_off_non_thermal_effects = True
		model.integrate_model()
		# make curve in transpacent phase dashed, select these values:
		time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
		time_arr_shock_transparent = model.time_arr[model.t_shock_transparent_index:-1]
		Lbol_arr_shock_opt_thick = model.Lbol_arr[0:model.t_shock_transparent_index]
		Lbol_arr_shock_transparent = model.Lbol_arr[model.t_shock_transparent_index:-1]
		plt.plot(time_arr_shock_opt_thick/year, Lbol_arr_shock_opt_thick, linewidth=1.5, ls="-", c=colour_arr[i], label= "Case: "+ model_name_arr[i])#": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls=":", c=colour_arr[i])
	plt.plot([-100,-100],[-100,-100], c="white", label="$R_{0} = 10^4\,R_\odot,$\n$E_\mathrm{kin,eje} = 10^{56}\,$erg$,$\n$ M_\mathrm{eje} = 10^5\,M_\odot$")



	#plt.semilogy()
	plt.loglog()
	plt.title(r"Rest-frame bolometric luminosity, varying $n_\mathrm{eje}$", fontsize=14) # add title to the whole plots
	plt.xlabel(r"$t_{source}$ [years]", fontsize=14)
	plt.ylabel(r"$L_{bol}$ [erg s$^{-1}$]", fontsize=14)
	plt.xlim(xrange[0],xrange[1]) # time in yr
	#plt.ylim(5.65133e9,5.65135e9) # y-axis
	plt.ylim(yrange[0],yrange[1])

	plt.legend(loc="upper right", ncol=1, fontsize = 6)

	picturename = picname_in
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)


# vary density distribution of the CSM:
def multiple_star_Luminosity_evolution_systematics_vary_CSM(picname_in, xrange=[1e-3,3e2],yrange=[1e42,1e48]):
	
	# choose which model to run:
	num_models = 6
	model_run_arr = [None]*num_models
	model_name_arr = [None]*num_models
	model_n_CSM_arr = [None]*num_models
	model_density_CSM_arr = [None]*num_models
	#colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	colour_arr = ["#bf7fc8","#7f7fdd","#7faded","#7fd2da","#7fda7f","#ffad7f"] # choose better-distinguishable colours
	# init models:
	for o in range(num_models):
		model_run_arr[o] = [1e4*R_sun_cm, 1e54, 1e3*M_sun_gram]
		model_name_arr[o] = "Model"
		model_n_CSM_arr[o] = 2
		model_density_CSM_arr[o] = 10**(1-o)
	#model_name_arr[0] = r"$n_\mathrm{eje} = 0, E_{kin,eje} = 10^{55}\,$erg, $M_{eje} = 10^4\,M_\odot, R_{0} = 10^4\,R_\odot$"
	model_name_arr[0] = r"$10 \times \rho_\mathrm{CSM,L}$"
	model_name_arr[1] = r"$1 \times \rho_\mathrm{CSM,L}$"
	model_name_arr[2] = r"$0.1 \times \rho_\mathrm{CSM,L}$"
	#-------------------------------------------
	for i in range(3):
		R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
		model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
		model.max_integration_time = 210.*year
		model.stop_after_opt_thick_phase = False
		model.use_custom_CSM_powerlaw = True
		model.force_turn_off_non_thermal_effects = True
		model.CSM_density_prefactor = model_density_CSM_arr[i]
		model.CSM_power_law_exponent = model_n_CSM_arr[i]
		model.integrate_model()
		# make curve in transpacent phase dashed, select these values:
		time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
		time_arr_shock_transparent = model.time_arr[model.t_shock_transparent_index:-1]
		Lbol_arr_shock_opt_thick = model.Lbol_arr[0:model.t_shock_transparent_index]
		Lbol_arr_shock_transparent = model.Lbol_arr[model.t_shock_transparent_index:-1]
		plt.plot(time_arr_shock_opt_thick/year, Lbol_arr_shock_opt_thick, linewidth=1.5, ls="-", c=colour_arr[i], label= "Case: "+ model_name_arr[i])#": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls=":", c=colour_arr[i])
	plt.plot([-100,-100],[-100,-100], c="white", label="$R_{0} = 10^4\,R_\odot,$\n$E_\mathrm{kin,eje} = 10^{54}\,$erg$,$\n$ M_\mathrm{eje} = 10^3\,M_\odot$")
	plt.plot([-100,-100],[-100,-100], c="white", label=r"$––––––$") # separator between the two cases


	for o in range(3,num_models):
		model_run_arr[o] = [1e4*R_sun_cm, 1e56, 1e5*M_sun_gram]
		model_name_arr[o] = "Model"
		model_n_CSM_arr[o] = 2-0.5*(o-3)
		model_density_CSM_arr[o] = 1
	model_name_arr[3] = r"$n_\mathrm{CSM} = 2$"
	model_name_arr[4] = r"$n_\mathrm{CSM} = 1.5$"
	model_name_arr[5] = r"$n_\mathrm{CSM} = 1$"
	#-------------------------------------------
	for i in range(3,len(model_run_arr)):
		R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
		model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
		model.max_integration_time = 210.*year
		model.stop_after_opt_thick_phase = False
		model.use_custom_CSM_powerlaw = True
		model.force_turn_off_non_thermal_effects = True
		model.CSM_density_prefactor = model_density_CSM_arr[i]
		model.CSM_power_law_exponent = model_n_CSM_arr[i]
		model.integrate_model()
		# make curve in transpacent phase dashed, select these values:
		time_arr_shock_opt_thick = model.time_arr[0:model.t_shock_transparent_index]
		time_arr_shock_transparent = model.time_arr[model.t_shock_transparent_index:-1]
		Lbol_arr_shock_opt_thick = model.Lbol_arr[0:model.t_shock_transparent_index]
		Lbol_arr_shock_transparent = model.Lbol_arr[model.t_shock_transparent_index:-1]
		plt.plot(time_arr_shock_opt_thick/year, Lbol_arr_shock_opt_thick, linewidth=1.5, ls="-", c=colour_arr[i], label= "Case: "+ model_name_arr[i])#": $E_{kin,eje}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0/R_sun_cm) + "$R_\odot$")
		plt.plot(time_arr_shock_transparent/year, Lbol_arr_shock_transparent, linewidth=1.5, ls=":", c=colour_arr[i])
	plt.plot([-100,-100],[-100,-100], c="white", label="$R_{0} = 10^4\,R_\odot,$\n$E_\mathrm{kin,eje} = 10^{56}\,$erg$,$\n$ M_\mathrm{eje} = 10^5\,M_\odot$")



	#plt.semilogy()
	plt.loglog()
	plt.title(r"Rest-frame bolometric luminosity, varying CSM", fontsize=14) # add title to the whole plots
	plt.xlabel(r"$t_{source}$ [years]", fontsize=14)
	plt.ylabel(r"$L_{bol}$ [erg s$^{-1}$]", fontsize=14)
	plt.xlim(xrange[0],xrange[1]) # time in yr
	#plt.ylim(5.65133e9,5.65135e9) # y-axis
	plt.ylim(yrange[0],yrange[1])

	plt.legend(loc="upper right", ncol=1, fontsize = 6)

	picturename = picname_in
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)




# single star color evolution in JWST filters:
def single_star_colour_evolution_JWST(model_in, model_name_in, redshift_in, xrange=[0,150],yrange=[32,20],num_filters_in=5, use_insert=False):
	
	# choose which model to run:
	model_run = model_in
	model_name = model_name_in
	num_filters = num_filters_in

	# def placeholde container for the insert:
	if use_insert:
		container_time = [None]*num_filters
		container_mag = [None]*num_filters

	#-------------------------------------------
	R_0 = model_run[0]; E_kin_eje = model_run[1]; M_eje = model_run[2]
	model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
	model.max_integration_time = 300.*year
	model.integrate_model()
	#
	#plt.plot(model.time_arr/year, model.Lbol_arr, linewidth=1.5, label= "Case H2-S: $E_{kin}$ = " + '{:.2e}'.format(E_kin_eje) + "$erg, M_{eje}$ = " + '{:.2e}'.format(M_eje/M_sun_gram) + "$M_\odot, R_{0}$ = " + '{:.2e}'.format(R_0) + "$cm$", ls="-", c="b")
	
	#plt.figure(figsize=(6.4,4.8))
	fig, ax1 = plt.subplots(figsize=(6.4,4.8))

	# init filters:
	filter_JWST_NIRCam_F070W = SMSclasses.Telescope_filter("JWST_NIRCam_F070W")
	filter_JWST_NIRCam_F090W = SMSclasses.Telescope_filter("JWST_NIRCam_F090W")
	filter_JWST_NIRCam_F115W = SMSclasses.Telescope_filter("JWST_NIRCam_F115W")
	filter_JWST_NIRCam_F150W = SMSclasses.Telescope_filter("JWST_NIRCam_F150W")
	filter_JWST_NIRCam_F200W = SMSclasses.Telescope_filter("JWST_NIRCam_F200W")
	filter_JWST_NIRCam_F277W = SMSclasses.Telescope_filter("JWST_NIRCam_F277W")
	filter_JWST_NIRCam_F356W = SMSclasses.Telescope_filter("JWST_NIRCam_F356W")
	filter_JWST_NIRCam_F444W = SMSclasses.Telescope_filter("JWST_NIRCam_F444W")
	#filter_arr = [filter_JWST_NIRCam_F070W,filter_JWST_NIRCam_F090W,filter_JWST_NIRCam_F115W,filter_JWST_NIRCam_F150W,filter_JWST_NIRCam_F200W,filter_JWST_NIRCam_F277W,filter_JWST_NIRCam_F356W,filter_JWST_NIRCam_F444W]
	#colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	filter_arr = [filter_JWST_NIRCam_F444W,filter_JWST_NIRCam_F356W,filter_JWST_NIRCam_F277W,filter_JWST_NIRCam_F200W,filter_JWST_NIRCam_F150W,filter_JWST_NIRCam_F115W,filter_JWST_NIRCam_F090W,filter_JWST_NIRCam_F070W]
	model_name_arr = ["JWST F444W","JWST F356W","JWST F277W","JWST F200W","JWST F150W","JWST F115W","JWST F090W","JWST F070W"]
	colour_arr = ["#ffad7f","#dffd7f","#7fda7f","#7fd2da","#7faded","#7f7fdd","#ab7fce","#bf7fc8"]

	redshift = redshift_in
	ax1.scatter([-10], [-10], marker="o", c="white", label=r"z = " + str(redshift))
	for i in range(0,num_filters):
		ABmag_lightcurve = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_arr[i])
		ABmag_lightcurve.debug = False
		ABmag_lightcurve.compute_AB_magnitude()
		# split the light curve into two parts
		time_arr_shock_opt_thick = ABmag_lightcurve.time_arr[0:model.t_shock_transparent_index]
		time_arr_shock_transparent = ABmag_lightcurve.time_arr[model.t_shock_transparent_index:-1]
		mAB_arr_shock_opt_thick = ABmag_lightcurve.ABmag_arr[0:model.t_shock_transparent_index]
		mAB_arr_shock_transparent = ABmag_lightcurve.ABmag_arr[model.t_shock_transparent_index:-1]
		#plt.plot((1+redshift)*ABmag_lightcurve.time_arr/year, ABmag_lightcurve.ABmag_arr, linewidth=2.5, color=colour_arr[i], linestyle="-")
		ax1.plot((1+redshift)*time_arr_shock_opt_thick/year, mAB_arr_shock_opt_thick, linewidth=2.5, color=colour_arr[i], linestyle="-", label=model_name_arr[i])
		ax1.plot((1+redshift)*time_arr_shock_transparent/year, mAB_arr_shock_transparent, linewidth=2.5, color=colour_arr[i], linestyle=":")
		if use_insert:
			container_time[i] = (1+redshift)*time_arr_shock_opt_thick/year
			container_mag[i] = mAB_arr_shock_opt_thick
			
	
	#plt.plot((1+redshift)*ABmag_lightcurve.time_arr/year, ABmag_lightcurve.ABmag_arr, linewidth=2., color=colour_arr[i], linestyle="-", label= "Case H2-S at z = " + str(redshift))
	
	ax1.legend(loc="upper right", ncol=1, fontsize = 8, markerscale=0)

	plt.title("Apparent magnitude for model "+model_name, fontsize=14) # add title to the whole plots
	plt.xlabel("$t_{obs} = (1+z)t_{source}$ [years]", fontsize=14)
	plt.ylabel("$m_{AB} (1000s$ exposure) [mag]", fontsize=14)
	max_xlim=xrange[1] # year
	plt.xlim(xrange[0],xrange[1]) # time in year
	plt.ylim(yrange[0],yrange[1]) # AB magnitude

	if use_insert:
		left, bottom, width, height = [0.35, 0.6, 0.35, 0.26] # modify to move the inset curve and change its size
		ax2 = fig.add_axes([left, bottom, width, height])
		#ax2.plot([27, 26, 20],[5, 10, 40])
		for j in range(num_filters):
			ax2.plot(container_time[j], container_mag[j], linewidth=2.5, color=colour_arr[j], linestyle="-")
		ax2.set_xlim([0,100])
		ax2.set_ylim([26,18])
		ax2.set_xlabel("$t_{obs}$ [years]")
		ax2.set_ylabel("$m_{AB}$ [mag]")

	
	# filter magnitude bounds:
	for j in range(num_filters):
		print(filter_arr[j].filter_magnitude_bound)
		# add nice-looking filter constraints
		#j = num_filters-1-j
		xarr = [j/(num_filters)*max_xlim, (j+1)/(num_filters)*max_xlim]
		yarr1 = [filter_arr[j].filter_magnitude_bound, filter_arr[j].filter_magnitude_bound]
		yarr2 = [100, 100]
		ax1.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color=colour_arr[j])
		ax1.text((j+0.5)/(num_filters)*max_xlim, filter_arr[j].filter_magnitude_bound+0.1, model_name_arr[j], fontsize=8, color="#"+mean_color(colour_arr[j],"#2f4f4f"), va="top",ha="center")
		#plt.axhline(y = filter_arr[j].filter_magnitude_bound, color=colour_arr[j], linestyle='-',xmin=xpos[j]-0.05,xmax=xpos[j]+0.05)
		#plt.hlines(y = filter_arr[j].filter_magnitude_bound, color=colour_arr[j], linestyle='-',xmin=xpos[j]-10,xmax=xpos[j]+10)
		#plt.text(xpos[j], filter_arr[j].filter_magnitude_bound+0.05, filter_arr[j].filter_name, fontsize=10, fontweight="semibold",va="top",ha="center",c=colour_arr[j],fontfamily="cursive")
	
	#picturename = 'paper_plots/mAB_model_'+model_name+'_z_'+str(redshift)+'.pdf'
	picturename = 'Fig7_eta_branch_mAB_model_'+model_name+'_z_'+str(redshift)+'.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.1) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)


# colour evolution tracks in JWST filters: (F277W-F444W) - F444W diagram
def tracks_F277W_F444W_vs_F444_diagram_at_different_redshifts(figname, cut_t0_peak=True):

	filter_JWST_NIRCam_F277W = SMSclasses.Telescope_filter("JWST_NIRCam_F277W")
	#filter_JWST_NIRCam_F356W = SMSclasses.Telescope_filter("JWST_NIRCam_F356W")
	filter_JWST_NIRCam_F444W = SMSclasses.Telescope_filter("JWST_NIRCam_F444W")

	redshift_array = [7,10,15,20]
	linestyls_array = ["-", "--", "-.", ":"]
	#model_run_arr = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2,FujHe1,FujHe2,FujHe4,NagExp,NagPul1,NagPul2]
	#model_name_arr= ["FujH1","FujH2","FujH4","FujDif1","NagCol1","NagCol2","FujHe1","FujHe2","FujHe4,""NagExp","NagPul1","NagPul2"]
	model_run_arr = [FujH1,FujH4,FujDif1,NagCol2,FujHe1,FujHe4,NagExp,NagPul2]
	model_name_arr = ["FujH1","FujH4","FujDif1","NagCol2","FujHe1","FujHe4","NagExp","NagPul2"]
	colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 200.*year
			model.stop_after_opt_thick_phase = True
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_F277W = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_JWST_NIRCam_F277W)
			ABmag_lightcurve_F277W.debug = False
			ABmag_lightcurve_F277W.compute_AB_magnitude()
			mag_F277W = ABmag_lightcurve_F277W.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_F444W = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_JWST_NIRCam_F444W)
			ABmag_lightcurve_F444W.debug = False
			ABmag_lightcurve_F444W.compute_AB_magnitude()
			mag_F444W = ABmag_lightcurve_F444W.ABmag_arr[start_index:model.t_shock_transparent_index]
			if j==0:
				plt.plot(mag_F444W, (mag_F277W-mag_F444W), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			else:
				plt.plot(mag_F444W, (mag_F277W-mag_F444W), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	order = [0,1,8,2,3,9,4,5,10,6,7,11]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=4, fontsize = 8)
	
	xarr = [20., filter_JWST_NIRCam_F444W.filter_magnitude_bound, filter_JWST_NIRCam_F444W.filter_magnitude_bound, 100]
	yarr1 = [8.7, 0.4, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -1.25, "detection bound\n(1000s exposure)", fontsize=8, color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("JWST NIRCam $mag_{F444W}$", fontsize=14)
	plt.ylabel("JWST NIRCam $mag_{F277W}-mag_{F444W}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-3,6) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = 'paper_plots/test/test_filter_tracks_F277W_F444W_vs_F444_at_redshift.pdf'
	if (not cut_t0_peak): picturename = 'paper_plots/test/test_filter_tracks_F277W_F444W_vs_F444_at_redshift_t0peak.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)
# colour evolution tracks in JWST filters: (F277W-F444W) - F444W diagram
def tracks_F277W_F444W_vs_F444_diagram_at_different_redshifts_with_time_ticks_eta0(figname, cut_t0_peak=True):

	filter_JWST_NIRCam_F277W = SMSclasses.Telescope_filter("JWST_NIRCam_F277W")
	#filter_JWST_NIRCam_F356W = SMSclasses.Telescope_filter("JWST_NIRCam_F356W")
	filter_JWST_NIRCam_F444W = SMSclasses.Telescope_filter("JWST_NIRCam_F444W")

	redshift_array = [7,10,15,20]
	linestyls_array = ["-", "--", "-.", ":"]
	#model_run_arr = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2,FujHe1,FujHe2,FujHe4,NagExp,NagPul1,NagPul2]
	#model_name_arr= ["FujH1","FujH2","FujH4","FujDif1","NagCol1","NagCol2","FujHe1","FujHe2","FujHe4,""NagExp","NagPul1","NagPul2"]
	#model_run_arr = [FujH1,FujH4,FujDif1,NagCol2,FujHe1,FujHe4,NagExp,NagPul2]
	#model_name_arr = ["FujH1","FujH4","FujDif1","NagCol2","FujHe1","FujHe4","NagExp","NagPul2"]
	#colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	#model_run_arr = [FujH4, FujHe1, FujHe4, NagPul2]
	#model_name_arr = ["FujH4","FujHe1","FujHe4","NagPul2"]
	#
	#model_run_arr = [FujH1, FujHe1, FujHe4, NagPul2]
	#model_name_arr = ["FujH1","FujHe1","FujHe4","NagPul2"]
	#colour_arr = ["#ab7fce","#7fd2da","#7fda7f","#ffad7f"]

	#model_run_arr = [FujH1,FujHe1,NagCol1,NagCol2,NagPul1,NagPul2,NagPul2,NagExp]
	#model_name_arr = ["FujH1","FujHe1","NagCol1","NagCol2","NagPul1","NagPul2","NagPul2","NagExp"]
	#colour_arr = ["#bf7fc8","#7fda7f","#7fd2da","#7f7fdd","#ffad7f","#7faded","#dffd7f","#ab7fce"]

	                    #[x, o, o, o, o, o, x, o] nein zu index 0 und 6
	model_run_arr = [NagPul2,FujH1,FujHe1,NagCol1,NagCol2,NagPul1,NagPul2,NagPul2]
	model_name_arr = [" ", "FujH1","FujHe1","NagCol1","NagCol2","NagPul1"," ","NagPul2"]
	colour_arr = ["#dffd7f","#bf7fc8","#7fda7f","#7fd2da","#7f7fdd","#ffad7f","#dffd7f","#7faded"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			#if i==0 and j==0:
			#	plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
			#	continue
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 400.*year
			model.stop_after_opt_thick_phase = True
			model.force_turn_off_non_thermal_effects = False
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_F277W = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_JWST_NIRCam_F277W)
			ABmag_lightcurve_F277W.debug = False
			ABmag_lightcurve_F277W.compute_AB_magnitude()
			mag_F277W = ABmag_lightcurve_F277W.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_F444W = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_JWST_NIRCam_F444W)
			ABmag_lightcurve_F444W.debug = False
			ABmag_lightcurve_F444W.compute_AB_magnitude()
			mag_F444W = ABmag_lightcurve_F444W.ABmag_arr[start_index:model.t_shock_transparent_index]
			# prepare scatter plot for time ticks:
			time_array = model.time_arr[start_index:model.t_shock_transparent_index]
			x_arr, y_arr = compute_xy_time_ticks(time_array, year, mag_F444W, (mag_F277W-mag_F444W))
			if i !=0 and i!=6:
				plt.scatter(x_arr, y_arr, c="#"+mean_color(colour_arr[i],"#2f4f4f"), marker="o", s=5.0,zorder=10)
			#
			
			if j==0:
				if i==0 or i==6:
					plt.plot([-100,-99], [-100,-99], linewidth=1.5, alpha=0, label=" ")
				else:
					plt.plot(mag_F444W, (mag_F277W-mag_F444W), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			else:
				if i==0 or i==6:
					plt.plot([-100,-99], [-100,-99], linewidth=1.5, alpha=0)
				else:
					plt.plot(mag_F444W, (mag_F277W-mag_F444W), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	#order = [0,4,1,5,2,6,3,7]
	order = [0,1,8,2,3,9,4,5,10,6,7,11]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=4, fontsize = 8)
	
	xarr = [20., filter_JWST_NIRCam_F444W.filter_magnitude_bound, filter_JWST_NIRCam_F444W.filter_magnitude_bound, 100]
	yarr1 = [8.7, 0.4, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -1.25, "detection bound\n(1000s exposure)", fontsize=8, color="#404040")

	# add explanatory marking text:
	plt.text(18.5+1.7, -1.25+0.7, "1 year time interval\nin source-frame", fontsize=8, color="#404040")
	plt.annotate(text="", xytext=(19.4+1.7,-0.7+0.7), xy=(20+1.65,0+0.75), arrowprops=dict(arrowstyle='->'), color="#404040")
	plt.annotate(text="", xytext=(19.3+1.7,-0.7+0.7), xy=(19.1+2.13,0.2+0.74), arrowprops=dict(arrowstyle='->'), color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("JWST NIRCam $mag_{F444W}$", fontsize=14)
	plt.ylabel("JWST NIRCam $mag_{F277W}-mag_{F444W}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-3,6) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = figname #'paper_plots/test/Balmerbreak_eta_model_filter_tracks_F277W_F444W_vs_F444_with_timeticks_at_redshift_eta_small.pdf'
	#picturename = 'paper_plots/test/eta_model_filter_tracks_F277W_F444W_vs_F444_with_timeticks_at_redshift.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)

def tracks_F277W_F444W_vs_F444_diagram_at_different_redshifts_with_time_ticks_eta1(figname, cut_t0_peak=True):

	filter_JWST_NIRCam_F277W = SMSclasses.Telescope_filter("JWST_NIRCam_F277W")
	#filter_JWST_NIRCam_F356W = SMSclasses.Telescope_filter("JWST_NIRCam_F356W")
	filter_JWST_NIRCam_F444W = SMSclasses.Telescope_filter("JWST_NIRCam_F444W")

	#redshift_array = [7,10,15,20]
	
	linestyls_array = ["-", "--", "-.", ":"]
	#model_run_arr = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2,FujHe1,FujHe2,FujHe4,NagExp,NagPul1,NagPul2]
	#model_name_arr= ["FujH1","FujH2","FujH4","FujDif1","NagCol1","NagCol2","FujHe1","FujHe2","FujHe4,""NagExp","NagPul1","NagPul2"]
	#model_run_arr = [FujH1,FujH4,FujDif1,NagCol2,FujHe1,FujHe4,NagExp,NagPul2]
	#model_name_arr = ["FujH1","FujH4","FujDif1","NagCol2","FujHe1","FujHe4","NagExp","NagPul2"]
	#colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	#model_run_arr = [FujH4, FujHe1, FujHe4, NagPul2]
	#model_name_arr = ["FujH4","FujHe1","FujHe4","NagPul2"]
	#
	#model_run_arr = [FujH1, FujHe1, FujHe4, NagPul2]
	#model_name_arr = ["FujH1","FujHe1","FujHe4","NagPul2"]
	#colour_arr = ["#ab7fce","#7fd2da","#7fda7f","#ffad7f"]

	#model_run_arr = [FujHe2, FujHe4, FujH2, FujDif1]
	#model_name_arr = ["FujHe2","FujHe4","FujH4","FujDif1"]
	#colour_arr = ["#ab7fce","#7fd2da","#7fda7f","#ffad7f"]

	redshift_array = [7,20]
	model_run_arr = [FujHe4, FujDif1]
	model_name_arr = ["FujHe4","FujDif1"]
	colour_arr = ["#7fd2da","#ffad7f"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			#if i==0 and j==0:
			#	plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
			#	continue
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 400.*year
			model.stop_after_opt_thick_phase = True
			model.force_turn_off_non_thermal_effects = False
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_F277W = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_JWST_NIRCam_F277W)
			ABmag_lightcurve_F277W.debug = False
			ABmag_lightcurve_F277W.compute_AB_magnitude()
			mag_F277W = ABmag_lightcurve_F277W.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_F444W = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_JWST_NIRCam_F444W)
			ABmag_lightcurve_F444W.debug = False
			ABmag_lightcurve_F444W.compute_AB_magnitude()
			mag_F444W = ABmag_lightcurve_F444W.ABmag_arr[start_index:model.t_shock_transparent_index]
			# prepare scatter plot for time ticks:
			time_array = model.time_arr[start_index:model.t_shock_transparent_index]
			x_arr, y_arr = compute_xy_time_ticks(time_array, 5*year, mag_F444W, (mag_F277W-mag_F444W))

			plt.scatter(x_arr, y_arr, c="#"+mean_color(colour_arr[i],"#2f4f4f"), marker="o", s=5.0,zorder=10)
			#
			
			if j==0:
				plt.plot(mag_F444W, (mag_F277W-mag_F444W), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			
			plt.plot(mag_F444W, (mag_F277W-mag_F444W), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	order = [0,2,1,3]
	#order = [0,4,1,5,2,6,3,7]
	#order = [0,1,8,2,3,9,4,5,10,6,7,11]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=2, fontsize = 8)
	
	xarr = [20., filter_JWST_NIRCam_F444W.filter_magnitude_bound, filter_JWST_NIRCam_F444W.filter_magnitude_bound, 100]
	yarr1 = [8.7, 0.4, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -1.25, "detection bound\n(1000s exposure)", fontsize=8, color="#404040")

	# add explanatory marking text:
	plt.text(18.5+1.2, -1.25-0.6, "5 year time interval\nin source-frame", fontsize=8, color="#404040")
	plt.annotate(text="", xytext=(19.4+1.2,-0.7-0.6), xy=(21.27,-0.84), arrowprops=dict(arrowstyle='->'), color="#404040")
	plt.annotate(text="", xytext=(19.3+1.2,-0.7-0.6), xy=(20.3,-0.7), arrowprops=dict(arrowstyle='->'), color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("JWST NIRCam $mag_{F444W}$", fontsize=14)
	plt.ylabel("JWST NIRCam $mag_{F277W}-mag_{F444W}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-3,6) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = figname #'paper_plots/test/eta_model_filter_tracks_F277W_F444W_vs_F444_with_timeticks_at_redshift_eta_large.pdf'
	#picturename = 'paper_plots/test/eta_model_filter_tracks_F277W_F444W_vs_F444_with_timeticks_at_redshift.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)


# colour-colour evolution tracks in JWST filters: (F277W-F444W) - (F277W-F356W)
def tracks_F277W_F444W_vs_F277W_F356W_diagram_at_different_redshifts(figname, cut_t0_peak=True):

	filter_JWST_NIRCam_F277W = SMSclasses.Telescope_filter("JWST_NIRCam_F277W")
	filter_JWST_NIRCam_F356W = SMSclasses.Telescope_filter("JWST_NIRCam_F356W")
	filter_JWST_NIRCam_F444W = SMSclasses.Telescope_filter("JWST_NIRCam_F444W")

	redshift_array = [7,10,15,20]
	linestyls_array = ["-", "--", "-.", ":"]
	#model_run_arr = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2,FujHe1,FujHe2,FujHe4,NagExp,NagPul1,NagPul2]
	#model_name_arr= ["FujH1","FujH2","FujH4","FujDif1","NagCol1","NagCol2","FujHe1","FujHe2","FujHe4,""NagExp","NagPul1","NagPul2"]
	model_run_arr = [FujH1,FujH4,FujDif1,NagCol2,FujHe1,FujHe4,NagExp,NagPul2]
	model_name_arr = ["FujH1","FujH4","FujDif1","NagCol2","FujHe1","FujHe4","NagExp","NagPul2"]
	colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 200.*year
			model.stop_after_opt_thick_phase = True
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_F277W = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_JWST_NIRCam_F277W)
			ABmag_lightcurve_F277W.debug = False
			ABmag_lightcurve_F277W.compute_AB_magnitude()
			mag_F277W = ABmag_lightcurve_F277W.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_F356W = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_JWST_NIRCam_F356W)
			ABmag_lightcurve_F356W.debug = False
			ABmag_lightcurve_F356W.compute_AB_magnitude()
			mag_F356W = ABmag_lightcurve_F356W.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_F444W = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_JWST_NIRCam_F444W)
			ABmag_lightcurve_F444W.debug = False
			ABmag_lightcurve_F444W.compute_AB_magnitude()
			mag_F444W = ABmag_lightcurve_F444W.ABmag_arr[start_index:model.t_shock_transparent_index]
			if j==0:
				plt.plot((mag_F277W-mag_F356W), (mag_F277W-mag_F444W), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			else:
				plt.plot((mag_F277W-mag_F356W), (mag_F277W-mag_F444W), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	order = [0,1,8,2,3,9,4,5,10,6,7,11]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=4, fontsize = 8)
	
	#xarr = [20., filter_JWST_NIRCam_F444W.filter_magnitude_bound, filter_JWST_NIRCam_F444W.filter_magnitude_bound, 100]
	#yarr1 = [8.7, 0.4, -10, -10]
	#yarr2 = [10, 10, 10, 10]
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	#plt.text(29.5, -1.25, "detection bound\n(1000s exposure)", fontsize=8, color="#404040")

	model_name="FujH1"
	plt.title("Colour colour diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("JWST NIRCam $mag_{F277W}-mag_{F356W}$", fontsize=14)
	plt.ylabel("JWST NIRCam $mag_{F277W}-mag_{F444W}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(-1,4) # AB magnitude
	plt.ylim(-3,6) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = 'paper_plots/test/test_filter_tracks_F277W_F444W_vs_F277W_F356W_at_redshift.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)

# colour evolution tracks in EUCLID filters: (Jband-Hband) - Hband diagram
def tracks_EUCLID_Jband_Hband_vs_Hband_diagram_at_different_redshifts(figname, cut_t0_peak=True):

	filter_EUCLID_NISP_Jband = SMSclasses.Telescope_filter("EUCLID_NISP_Jband")
	filter_EUCLID_NISP_Hband = SMSclasses.Telescope_filter("EUCLID_NISP_Hband")

	redshift_array = [2,4,6,10]
	linestyls_array = ["-", "--", "-.", ":"]
	#model_run_arr = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2,FujHe1,FujHe2,FujHe4,NagExp,NagPul1,NagPul2]
	#model_name_arr= ["FujH1","FujH2","FujH4","FujDif1","NagCol1","NagCol2","FujHe1","FujHe2","FujHe4,""NagExp","NagPul1","NagPul2"]
	model_run_arr = [FujH1,FujH4,FujDif1,NagCol2,FujHe1,FujHe4,NagExp,NagPul2]
	model_name_arr = ["FujH1","FujH4","FujDif1","NagCol2","FujHe1","FujHe4","NagExp","NagPul2"]
	colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]
	#
	# for VMS:
	'''
	VMS_880 = [1e3*R_sun_cm, 2.9e53, 88*M_sun_gram]
	VMS_6600 = [1e3*R_sun_cm, 2.2e54, 660*M_sun_gram]
	model_run_arr = [VMS_880,VMS_6600,VMS_880,VMS_6600]
	model_name_arr = ["VMS_880","VMS_6600","VMS_880","VMS_6600"]
	'''

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 200.*year
			model.stop_after_opt_thick_phase = True
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_Jband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Jband)
			ABmag_lightcurve_Jband.debug = False
			ABmag_lightcurve_Jband.compute_AB_magnitude()
			mag_Jband = ABmag_lightcurve_Jband.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_Hband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Hband)
			ABmag_lightcurve_Hband.debug = False
			ABmag_lightcurve_Hband.compute_AB_magnitude()
			mag_Hband = ABmag_lightcurve_Hband.ABmag_arr[start_index:model.t_shock_transparent_index]
			# prepare scatter plot for time ticks:
			#time_array = model.time_arr[start_index:model.t_shock_transparent_index]
			#x_arr, y_arr = compute_xy_time_ticks(time_array, year, mag_Hband, (mag_Jband-mag_Hband))
			#plt.scatter(x_arr, y_arr, c="#"+mean_color(colour_arr[i],"#2f4f4f"), marker="o", s=5.0,zorder=10)
			if j==0:
				plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			else:
				plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	order = [0,1,8,2,3,9,4,5,10,6,7,11]
	###order = [0,1,2,3]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=4, fontsize = 8)
	
	xarr = [20., filter_EUCLID_NISP_Hband.filter_magnitude_bound, filter_EUCLID_NISP_Hband.filter_magnitude_bound, 100]
	yarr1 = [filter_EUCLID_NISP_Hband.filter_magnitude_bound-20., 0.0, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -0.5, "detection bound\n(EDF survey)", fontsize=8, color="#404040")
	# detection bound for euclid wide field survey:
	xarr_2 = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound-2, filter_EUCLID_NISP_Hband.filter_magnitude_bound-2, 100]
	yarr1_2 = [filter_EUCLID_NISP_Hband.filter_magnitude_bound-2-10., 0.0, -10, -10]
	#yarr2 = [10, 10, 10, 10]
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.plot(xarr_2, yarr1_2, alpha = 0.3, color="grey", lw=2, ls="--")
	plt.text(18.5, 4.5, "detection bound\n(EWF survey)", fontsize=8, color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("EUCLID NISP $mag_\mathrm{H–band}$", fontsize=14)
	plt.ylabel("EUCLID NISP $mag_\mathrm{J–band}-mag_\mathrm{H–band}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-2,5) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = 'paper_plots/test/test_filter_tracks_EUCLID_Jband_Hband_vs_Hband_at_redshift.pdf'
	###picturename = 'paper_plots/test/test_VMS_filter_tracks_EUCLID_Jband_Hband_vs_Hband_at_redshift.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)
# colour evolution tracks in EUCLID filters: (Jband-Hband) - Hband diagram
def tracks_EUCLID_Jband_Hband_vs_Hband_diagram_at_different_redshifts_with_time_ticks(figname, cut_t0_peak=True):

	filter_EUCLID_NISP_Jband = SMSclasses.Telescope_filter("EUCLID_NISP_Jband")
	filter_EUCLID_NISP_Hband = SMSclasses.Telescope_filter("EUCLID_NISP_Hband")

	redshift_array = [4,6,8,10]
	linestyls_array = ["-", "--", "-.", ":"]
	#model_run_arr = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2,FujHe1,FujHe2,FujHe4,NagExp,NagPul1,NagPul2]
	#model_name_arr= ["FujH1","FujH2","FujH4","FujDif1","NagCol1","NagCol2","FujHe1","FujHe2","FujHe4,""NagExp","NagPul1","NagPul2"]
	model_run_arr = [FujH4, FujHe1, FujHe4, NagPul2]
	model_name_arr = ["FujH4","FujHe1","FujHe4","NagPul2"]
	colour_arr = ["#ab7fce","#7fd2da","#7fda7f","#ffad7f"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 200.*year
			model.stop_after_opt_thick_phase = True
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_Jband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Jband)
			ABmag_lightcurve_Jband.debug = False
			ABmag_lightcurve_Jband.compute_AB_magnitude()
			mag_Jband = ABmag_lightcurve_Jband.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_Hband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Hband)
			ABmag_lightcurve_Hband.debug = False
			ABmag_lightcurve_Hband.compute_AB_magnitude()
			mag_Hband = ABmag_lightcurve_Hband.ABmag_arr[start_index:model.t_shock_transparent_index]
			# prepare scatter plot for time ticks:
			time_array = model.time_arr[start_index:model.t_shock_transparent_index]
			x_arr, y_arr = compute_xy_time_ticks(time_array, year, mag_Hband, (mag_Jband-mag_Hband))
			plt.scatter(x_arr, y_arr, c="#"+mean_color(colour_arr[i],"#2f4f4f"), marker="o", s=5.0,zorder=10)
			if j==0:
				plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			else:
				plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	order = [0,4,1,5,2,6,3,7]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=4, fontsize = 8)
	
	xarr = [20., filter_EUCLID_NISP_Hband.filter_magnitude_bound, filter_EUCLID_NISP_Hband.filter_magnitude_bound, 100]
	yarr1 = [filter_EUCLID_NISP_Hband.filter_magnitude_bound-20., 0.0, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -0.5, "detection bound\n(EDF survey)", fontsize=8, color="#404040")
	# detection bound for euclid wide field survey:
	xarr_2 = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound-2, filter_EUCLID_NISP_Hband.filter_magnitude_bound-2, 100]
	yarr1_2 = [filter_EUCLID_NISP_Hband.filter_magnitude_bound-2-10., 0.0, -10, -10]
	#yarr2 = [10, 10, 10, 10]
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.plot(xarr_2, yarr1_2, alpha = 0.3, color="grey", lw=2, ls="--")
	plt.text(18.5, 4.5, "detection bound\n(EWF survey)", fontsize=8, color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("EUCLID NISP $mag_\mathrm{H–band}$", fontsize=14)
	plt.ylabel("EUCLID NISP $mag_\mathrm{J–band}-mag_\mathrm{H–band}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-2,5) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = 'paper_plots/test/test_filter_tracks_EUCLID_Jband_Hband_vs_Hband_at_redshift_with_time_ticks_eta_test.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)

def tracks_EUCLID_Jband_Hband_vs_Hband_diagram_at_different_redshifts_with_time_ticks_eta0(figname, cut_t0_peak=True):

	filter_EUCLID_NISP_Jband = SMSclasses.Telescope_filter("EUCLID_NISP_Jband")
	filter_EUCLID_NISP_Hband = SMSclasses.Telescope_filter("EUCLID_NISP_Hband")

	redshift_array = [4,6,8,10]
	linestyls_array = ["-", "--", "-.", ":"]
	model_run_arr = [NagPul2,FujH1,FujHe1,NagCol1,NagCol2,NagPul1,NagPul2,NagPul2]
	model_name_arr = [" ", "FujH1","FujHe1","NagCol1","NagCol2","NagPul1"," ","NagPul2"]
	colour_arr = ["#dffd7f","#bf7fc8","#7fda7f","#7fd2da","#7f7fdd","#ffad7f","#dffd7f","#7faded"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 400.*year
			model.stop_after_opt_thick_phase = True
			model.force_turn_off_non_thermal_effects = False
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_Jband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Jband)
			ABmag_lightcurve_Jband.debug = False
			ABmag_lightcurve_Jband.compute_AB_magnitude()
			mag_Jband = ABmag_lightcurve_Jband.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_Hband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Hband)
			ABmag_lightcurve_Hband.debug = False
			ABmag_lightcurve_Hband.compute_AB_magnitude()
			mag_Hband = ABmag_lightcurve_Hband.ABmag_arr[start_index:model.t_shock_transparent_index]
			# prepare scatter plot for time ticks:
			time_array = model.time_arr[start_index:model.t_shock_transparent_index]
			x_arr, y_arr = compute_xy_time_ticks(time_array, year, mag_Hband, (mag_Jband-mag_Hband))
			if i !=0 and i!=6:
				plt.scatter(x_arr, y_arr, c="#"+mean_color(colour_arr[i],"#2f4f4f"), marker="o", s=5.0,zorder=10)
			
			if j==0:
				if i==0 or i==6:
					plt.plot([-100,-99], [-100,-99], linewidth=1.5, alpha=0, label=" ")
				else:
					plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			else:
				if i==0 or i==6:
					plt.plot([-100,-99], [-100,-99], linewidth=1.5, alpha=0)
				else:
					plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	#order = [0,4,1,5,2,6,3,7]
	order = [0,1,8,2,3,9,4,5,10,6,7,11]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=4, fontsize = 8)
	
	xarr = [20., filter_EUCLID_NISP_Hband.filter_magnitude_bound, filter_EUCLID_NISP_Hband.filter_magnitude_bound, 100]
	yarr1 = [filter_EUCLID_NISP_Hband.filter_magnitude_bound-20., 0.0, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -0.5, "detection bound\n(EDF survey)", fontsize=8, color="#404040")
	# detection bound for euclid wide field survey:
	xarr_2 = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound-2, filter_EUCLID_NISP_Hband.filter_magnitude_bound-2, 100]
	yarr1_2 = [filter_EUCLID_NISP_Hband.filter_magnitude_bound-2-10., 0.0, -10, -10]
	#yarr2 = [10, 10, 10, 10]
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.plot(xarr_2, yarr1_2, alpha = 0.3, color="grey", lw=2, ls="--")
	plt.text(18.5, 4.5, "detection bound\n(EWF survey)", fontsize=8, color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("EUCLID NISP $mag_\mathrm{H–band}$", fontsize=14)
	plt.ylabel("EUCLID NISP $mag_\mathrm{J–band}-mag_\mathrm{H–band}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-2,5) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = figname #'paper_plots/test/Balmerbreak_eta_model_filter_tracks_EUCLID_Jband_Hband_vs_Hband_at_redshift_with_time_ticks_eta_small.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)

def tracks_EUCLID_Jband_Hband_vs_Hband_diagram_at_different_redshifts_with_time_ticks_eta1(figname, cut_t0_peak=True):

	filter_EUCLID_NISP_Jband = SMSclasses.Telescope_filter("EUCLID_NISP_Jband")
	filter_EUCLID_NISP_Hband = SMSclasses.Telescope_filter("EUCLID_NISP_Hband")

	#redshift_array = [4,6,8,10]
	redshift_array = [4,11.5]
	linestyls_array = ["-", "--", "-.", ":"]
	model_run_arr = [FujHe4, FujDif1]
	model_name_arr = ["FujHe4","FujDif1"]
	colour_arr = ["#7fd2da","#ffad7f"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 400.*year
			model.stop_after_opt_thick_phase = True
			model.force_turn_off_non_thermal_effects = False
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_Jband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Jband)
			ABmag_lightcurve_Jband.debug = False
			ABmag_lightcurve_Jband.compute_AB_magnitude()
			mag_Jband = ABmag_lightcurve_Jband.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_Hband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Hband)
			ABmag_lightcurve_Hband.debug = False
			ABmag_lightcurve_Hband.compute_AB_magnitude()
			mag_Hband = ABmag_lightcurve_Hband.ABmag_arr[start_index:model.t_shock_transparent_index]
			# prepare scatter plot for time ticks:
			time_array = model.time_arr[start_index:model.t_shock_transparent_index]
			x_arr, y_arr = compute_xy_time_ticks(time_array, 5.*year, mag_Hband, (mag_Jband-mag_Hband))
			
			plt.scatter(x_arr, y_arr, c="#"+mean_color(colour_arr[i],"#2f4f4f"), marker="o", s=5.0,zorder=10)
			
			if j==0:
				plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])

			plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	#order = [0,4,1,5,2,6,3,7]
	order = [0,2,1,3]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=2, fontsize = 8)
	
	xarr = [20., filter_EUCLID_NISP_Hband.filter_magnitude_bound, filter_EUCLID_NISP_Hband.filter_magnitude_bound, 100]
	yarr1 = [filter_EUCLID_NISP_Hband.filter_magnitude_bound-20., 0.0, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -0.5, "detection bound\n(EDF survey)", fontsize=8, color="#404040")
	# detection bound for euclid wide field survey:
	xarr_2 = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound-2, filter_EUCLID_NISP_Hband.filter_magnitude_bound-2, 100]
	yarr1_2 = [filter_EUCLID_NISP_Hband.filter_magnitude_bound-2-10., 0.0, -10, -10]
	#yarr2 = [10, 10, 10, 10]
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.plot(xarr_2, yarr1_2, alpha = 0.3, color="grey", lw=2, ls="--")
	plt.text(18.5, 4.5, "detection bound\n(EWF survey)", fontsize=8, color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("EUCLID NISP $mag_\mathrm{H–band}$", fontsize=14)
	plt.ylabel("EUCLID NISP $mag_\mathrm{J–band}-mag_\mathrm{H–band}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-2,5) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = figname #'paper_plots/test/eta_model_filter_tracks_EUCLID_Jband_Hband_vs_Hband_at_redshift_with_time_ticks_eta_large.pdf'
	##eta_model_filter_tracks_F277W_F444W_vs_F444_with_timeticks_at_redshift_eta_small

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)



# colour evolution tracks in Roman ST filters: (Jband-Hband) - Hband diagram
def tracks_RST_Jband_Hband_vs_Hband_diagram_at_different_redshifts(figname, cut_t0_peak=True):

	filter_EUCLID_NISP_Jband = SMSclasses.Telescope_filter("ROMAN_WFI_Jband")
	filter_EUCLID_NISP_Hband = SMSclasses.Telescope_filter("ROMAN_WFI_Hband")

	redshift_array = [5,7,9,10.5]
	linestyls_array = ["-", "--", "-.", ":"]
	#model_run_arr = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2,FujHe1,FujHe2,FujHe4,NagExp,NagPul1,NagPul2]
	#model_name_arr= ["FujH1","FujH2","FujH4","FujDif1","NagCol1","NagCol2","FujHe1","FujHe2","FujHe4,""NagExp","NagPul1","NagPul2"]
	model_run_arr = [FujH1,FujH4,FujDif1,NagCol2,FujHe1,FujHe4,NagExp,NagPul2]
	model_name_arr = ["FujH1","FujH4","FujDif1","NagCol2","FujHe1","FujHe4","NagExp","NagPul2"]
	colour_arr = ["#bf7fc8","#ab7fce","#7f7fdd","#7faded","#7fd2da","#7fda7f","#dffd7f","#ffad7f"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 200.*year
			model.stop_after_opt_thick_phase = True
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_Jband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Jband)
			ABmag_lightcurve_Jband.debug = False
			ABmag_lightcurve_Jband.compute_AB_magnitude()
			mag_Jband = ABmag_lightcurve_Jband.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_Hband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Hband)
			ABmag_lightcurve_Hband.debug = False
			ABmag_lightcurve_Hband.compute_AB_magnitude()
			mag_Hband = ABmag_lightcurve_Hband.ABmag_arr[start_index:model.t_shock_transparent_index]
			# prepare scatter plot for time ticks:
			#time_array = model.time_arr[start_index:model.t_shock_transparent_index]
			#x_arr, y_arr = compute_xy_time_ticks(time_array, year, mag_Hband, (mag_Jband-mag_Hband))
			#plt.scatter(x_arr, y_arr, c="#"+mean_color(colour_arr[i],"#2f4f4f"), marker="o", s=5.0,zorder=10)
			if j==0:
				plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			else:
				plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	order = [0,1,8,2,3,9,4,5,10,6,7,11]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=4, fontsize = 8)
	
	xarr = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound, filter_EUCLID_NISP_Hband.filter_magnitude_bound, 100]
	yarr1 = [filter_EUCLID_NISP_Jband.filter_magnitude_bound-10., filter_EUCLID_NISP_Jband.filter_magnitude_bound-filter_EUCLID_NISP_Hband.filter_magnitude_bound, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -0.5, "detection bound\n(SN deep survey)", fontsize=8, color="#404040")
	# detection bound for roman SN medium field survey:
	xarr_2 = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3, filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3, 100]
	yarr1_2 = [filter_EUCLID_NISP_Jband.filter_magnitude_bound-1.7-10., filter_EUCLID_NISP_Jband.filter_magnitude_bound-1.7-(filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3), -10, -10]
	#yarr2 = [10, 10, 10, 10]
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.plot(xarr_2, yarr1_2, alpha = 0.3, color="grey", lw=2, ls="--")
	plt.text(20.5, 4.5, "detection bound\n(SN medium survey)", fontsize=8, color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("RST WFI $mag_\mathrm{H–band}$", fontsize=14)
	plt.ylabel("RST WFI $mag_\mathrm{J–band}-mag_\mathrm{H–band}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-2,5) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = 'paper_plots/test/test_filter_tracks_RST_Jband_Hband_vs_Hband_at_redshift.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)

# colour evolution tracks in Roman ST filters: (Jband-Hband) - Hband diagram
def tracks_RST_Jband_Hband_vs_Hband_diagram_at_different_redshifts_with_time_ticks(figname, cut_t0_peak=True):

	filter_EUCLID_NISP_Jband = SMSclasses.Telescope_filter("ROMAN_WFI_Jband")
	filter_EUCLID_NISP_Hband = SMSclasses.Telescope_filter("ROMAN_WFI_Hband")

	redshift_array = [5,7,9,10.5]
	linestyls_array = ["-", "--", "-.", ":"]
	#model_run_arr = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2,FujHe1,FujHe2,FujHe4,NagExp,NagPul1,NagPul2]
	#model_name_arr= ["FujH1","FujH2","FujH4","FujDif1","NagCol1","NagCol2","FujHe1","FujHe2","FujHe4,""NagExp","NagPul1","NagPul2"]
	model_run_arr = [FujH4, FujHe1, FujHe4, NagPul2]
	model_name_arr = ["FujH4","FujHe1","FujHe4","NagPul2"]
	colour_arr = ["#ab7fce","#7fd2da","#7fda7f","#ffad7f"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 200.*year
			model.stop_after_opt_thick_phase = True
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_Jband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Jband)
			ABmag_lightcurve_Jband.debug = False
			ABmag_lightcurve_Jband.compute_AB_magnitude()
			mag_Jband = ABmag_lightcurve_Jband.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_Hband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Hband)
			ABmag_lightcurve_Hband.debug = False
			ABmag_lightcurve_Hband.compute_AB_magnitude()
			mag_Hband = ABmag_lightcurve_Hband.ABmag_arr[start_index:model.t_shock_transparent_index]
			# prepare scatter plot for time ticks:
			time_array = model.time_arr[start_index:model.t_shock_transparent_index]
			x_arr, y_arr = compute_xy_time_ticks(time_array, year, mag_Hband, (mag_Jband-mag_Hband))
			plt.scatter(x_arr, y_arr, c="#"+mean_color(colour_arr[i],"#2f4f4f"), marker="o", s=5.0,zorder=10)
			if j==0:
				plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			else:
				plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	order = [0,4,1,5,2,6,3,7]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=4, fontsize = 8)
	
	xarr = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound, filter_EUCLID_NISP_Hband.filter_magnitude_bound, 100]
	yarr1 = [filter_EUCLID_NISP_Jband.filter_magnitude_bound-10., filter_EUCLID_NISP_Jband.filter_magnitude_bound-filter_EUCLID_NISP_Hband.filter_magnitude_bound, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -0.5, "detection bound\n(SN deep survey)", fontsize=8, color="#404040")
	# detection bound for roman SN medium field survey:
	xarr_2 = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3, filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3, 100]
	yarr1_2 = [filter_EUCLID_NISP_Jband.filter_magnitude_bound-1.7-10., filter_EUCLID_NISP_Jband.filter_magnitude_bound-1.7-(filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3), -10, -10]
	#yarr2 = [10, 10, 10, 10]
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.plot(xarr_2, yarr1_2, alpha = 0.3, color="grey", lw=2, ls="--")
	plt.text(20.5, 4.5, "detection bound\n(SN medium survey)", fontsize=8, color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("RST WFI $mag_\mathrm{H–band}$", fontsize=14)
	plt.ylabel("RST WFI $mag_\mathrm{J–band}-mag_\mathrm{H–band}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-2,5) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = 'paper_plots/test/test_filter_tracks_RST_Jband_Hband_vs_Hband_at_redshift_with_time_ticks.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)

def tracks_RST_Jband_Hband_vs_Hband_diagram_at_different_redshifts_with_time_ticks_eta0(figname, cut_t0_peak=True):

	filter_EUCLID_NISP_Jband = SMSclasses.Telescope_filter("ROMAN_WFI_Jband")
	filter_EUCLID_NISP_Hband = SMSclasses.Telescope_filter("ROMAN_WFI_Hband")

	redshift_array = [5,7,9,10.5]
	linestyls_array = ["-", "--", "-.", ":"]
	model_run_arr = [NagPul2,FujH1,FujHe1,NagCol1,NagCol2,NagPul1,NagPul2,NagPul2]
	model_name_arr = [" ", "FujH1","FujHe1","NagCol1","NagCol2","NagPul1"," ","NagPul2"]
	colour_arr = ["#dffd7f","#bf7fc8","#7fda7f","#7fd2da","#7f7fdd","#ffad7f","#dffd7f","#7faded"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 200.*year
			model.stop_after_opt_thick_phase = True
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_Jband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Jband)
			ABmag_lightcurve_Jband.debug = False
			ABmag_lightcurve_Jband.compute_AB_magnitude()
			mag_Jband = ABmag_lightcurve_Jband.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_Hband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Hband)
			ABmag_lightcurve_Hband.debug = False
			ABmag_lightcurve_Hband.compute_AB_magnitude()
			mag_Hband = ABmag_lightcurve_Hband.ABmag_arr[start_index:model.t_shock_transparent_index]
			# prepare scatter plot for time ticks:
			time_array = model.time_arr[start_index:model.t_shock_transparent_index]
			x_arr, y_arr = compute_xy_time_ticks(time_array, year, mag_Hband, (mag_Jband-mag_Hband))
			if i !=0 and i!=6:
				plt.scatter(x_arr, y_arr, c="#"+mean_color(colour_arr[i],"#2f4f4f"), marker="o", s=5.0,zorder=10)
			
			if j==0:
				if i==0 or i==6:
					plt.plot([-100,-99], [-100,-99], linewidth=1.5, alpha=0, label=" ")
				else:
					plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			else:
				if i==0 or i==6:
					plt.plot([-100,-99], [-100,-99], linewidth=1.5, alpha=0)
				else:
					plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	#order = [0,4,1,5,2,6,3,7]
	order = [0,1,8,2,3,9,4,5,10,6,7,11]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=4, fontsize = 8)
	
	xarr = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound, filter_EUCLID_NISP_Hband.filter_magnitude_bound, 100]
	yarr1 = [filter_EUCLID_NISP_Jband.filter_magnitude_bound-10., filter_EUCLID_NISP_Jband.filter_magnitude_bound-filter_EUCLID_NISP_Hband.filter_magnitude_bound, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -0.5, "detection bound\n(SN deep survey)", fontsize=8, color="#404040")
	# detection bound for roman SN medium field survey:
	xarr_2 = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3, filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3, 100]
	yarr1_2 = [filter_EUCLID_NISP_Jband.filter_magnitude_bound-1.7-10., filter_EUCLID_NISP_Jband.filter_magnitude_bound-1.7-(filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3), -10, -10]
	#yarr2 = [10, 10, 10, 10]
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.plot(xarr_2, yarr1_2, alpha = 0.3, color="grey", lw=2, ls="--")
	plt.text(20.5, 4.5, "detection bound\n(SN medium survey)", fontsize=8, color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("RST WFI $mag_\mathrm{H–band}$", fontsize=14)
	plt.ylabel("RST WFI $mag_\mathrm{J–band}-mag_\mathrm{H–band}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-2,5) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = figname #'paper_plots/test/Balmerbreak_eta_model_filter_tracks_RST_Jband_Hband_vs_Hband_at_redshift_with_time_ticks_eta_small.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)


def tracks_RST_Jband_Hband_vs_Hband_diagram_at_different_redshifts_with_time_ticks_eta1(figname, cut_t0_peak=True):

	filter_EUCLID_NISP_Jband = SMSclasses.Telescope_filter("ROMAN_WFI_Jband")
	filter_EUCLID_NISP_Hband = SMSclasses.Telescope_filter("ROMAN_WFI_Hband")

	redshift_array = [5,10.5]
	linestyls_array = ["-", "--", "-.", ":"]
	#model_run_arr = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2,FujHe1,FujHe2,FujHe4,NagExp,NagPul1,NagPul2]
	#model_name_arr= ["FujH1","FujH2","FujH4","FujDif1","NagCol1","NagCol2","FujHe1","FujHe2","FujHe4,""NagExp","NagPul1","NagPul2"]
	model_run_arr = [FujHe4, FujDif1]
	model_name_arr = ["FujHe4","FujDif1"]
	colour_arr = ["#7fd2da","#ffad7f"]

	for j in range(len(redshift_array)):
		for i in range(len(model_run_arr)):
			R_0 = model_run_arr[i][0]; E_kin_eje = model_run_arr[i][1]; M_eje = model_run_arr[i][2]
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 200.*year
			model.stop_after_opt_thick_phase = True
			model.integrate_model()
			#
			start_index = 0
			if cut_t0_peak:
				start_index = model.shock_opt_thick_Lbol_local_min_after_t0_index

			redshift = redshift_array[j]
			ABmag_lightcurve_Jband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Jband)
			ABmag_lightcurve_Jband.debug = False
			ABmag_lightcurve_Jband.compute_AB_magnitude()
			mag_Jband = ABmag_lightcurve_Jband.ABmag_arr[start_index:model.t_shock_transparent_index]
			ABmag_lightcurve_Hband = SMSclasses.ABmagnitude_lightcurve(model, redshift, filter_EUCLID_NISP_Hband)
			ABmag_lightcurve_Hband.debug = False
			ABmag_lightcurve_Hband.compute_AB_magnitude()
			mag_Hband = ABmag_lightcurve_Hband.ABmag_arr[start_index:model.t_shock_transparent_index]
			# prepare scatter plot for time ticks:
			time_array = model.time_arr[start_index:model.t_shock_transparent_index]
			x_arr, y_arr = compute_xy_time_ticks(time_array, 5.*year, mag_Hband, (mag_Jband-mag_Hband))
			plt.scatter(x_arr, y_arr, c="#"+mean_color(colour_arr[i],"#2f4f4f"), marker="o", s=5.0,zorder=10)
			if j==0:
				plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i], label=model_name_arr[i])
			
			plt.plot(mag_Hband, (mag_Jband-mag_Hband), linewidth=1.5, linestyle=linestyls_array[j], c=colour_arr[i])

	for k in range(len(redshift_array)):
		plt.plot([-100,-99], [-100,-99], linewidth=1.5, linestyle=linestyls_array[k], c="black", label="$z=$"+str(redshift_array[k]))
		
	handles, labels = plt.gca().get_legend_handles_labels()
	#order = [0,4,1,5,2,6,3,7]
	order = [0,2,1,3]
	#plt.legend(loc="lower center", ncol=4, fontsize = 8)
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower center", ncol=2, fontsize = 8)
	
	xarr = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound, filter_EUCLID_NISP_Hband.filter_magnitude_bound, 100]
	yarr1 = [filter_EUCLID_NISP_Jband.filter_magnitude_bound-10., filter_EUCLID_NISP_Jband.filter_magnitude_bound-filter_EUCLID_NISP_Hband.filter_magnitude_bound, -10, -10]
	yarr2 = [10, 10, 10, 10]
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.text(29.5, -0.5, "detection bound\n(SN deep survey)", fontsize=8, color="#404040")
	# detection bound for roman SN medium field survey:
	xarr_2 = [10., filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3, filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3, 100]
	yarr1_2 = [filter_EUCLID_NISP_Jband.filter_magnitude_bound-1.7-10., filter_EUCLID_NISP_Jband.filter_magnitude_bound-1.7-(filter_EUCLID_NISP_Hband.filter_magnitude_bound-1.3), -10, -10]
	#yarr2 = [10, 10, 10, 10]
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.3, color="grey")
	plt.plot(xarr_2, yarr1_2, alpha = 0.3, color="grey", lw=2, ls="--")
	plt.text(20.5, 4.5, "detection bound\n(SN medium survey)", fontsize=8, color="#404040")

	model_name="FujH1"
	plt.title("Colour magnitude diagram", fontsize=14) # add title to the whole plots
	plt.xlabel("RST WFI $mag_\mathrm{H–band}$", fontsize=14)
	plt.ylabel("RST WFI $mag_\mathrm{J–band}-mag_\mathrm{H–band}$", fontsize=14)
	#max_xlim=xrange[1] # year
	plt.xlim(17, 33) # AB magnitude
	plt.ylim(-2,5) # AB magnitude
	plt.grid(alpha=0.15, linestyle="--", c="black")

	picturename = figname #'paper_plots/test/eta_model_filter_tracks_RST_Jband_Hband_vs_Hband_at_redshift_with_time_ticks_eta_large.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.2) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)




def SMS_star_grid_max_luminosity():

	lenarrays = 100 # may take 5-10 minutes to run!
	vals_Meje = np.geomspace(2e3*M_sun_gram,3.5e5*M_sun_gram,lenarrays)
	vals_Ekin_eje = np.geomspace(1e53, 5e56, lenarrays)
	  
	# Creating 2-D grid of features
	[X, Y] = np.meshgrid(vals_Meje/M_sun_gram, vals_Ekin_eje)
	Zvals = (X*X) * 1.
	Zvals2 = (X*X) * 1.
	fig, ax = plt.subplots(1, 1)

	for i in range(lenarrays):
		for j in range(lenarrays):
			R_0 = 1e4*R_sun_cm
			M_eje = vals_Meje[i]; E_kin_eje = vals_Ekin_eje[j]
			#
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 300.*year
			model.stop_after_opt_thick_phase = True
			model.max_timestep = 20.*year
			model.integrate_model()
			Zvals[j][i] = model.shock_opt_thick_Lbol_peak_luminosity
			Zvals2[j][i] = model.shock_opt_thick_Lbol_peak_luminosity/model.shock_transparent_Lbol_initial_luminosity # normalize for stability of contourf
			# catch some exceptions to make the plot look nicer:
			if Zvals[j][i] > 6e48: Zvals[j][i]=6e48
			if Zvals[j][i] < 1e42: Zvals[j][i]=1.01e42
			if Zvals2[j][i] < 1.1e-8: Zvals2[j][i]=1.1e-8
			#
			print(i, j, Zvals[j][i], M_eje, E_kin_eje)
	  

	print(Zvals)
	print(Zvals2)
	  
	# plots filled contour plot 
	contourf_ = ax.contourf(X, Y, np.log10(Zvals),levels=40,extend='max', cmap="viridis")
	fig_cbar = fig.colorbar(contourf_, ticks=range(43, 49, 1))
	fig_cbar.set_label(label=r'$log_{10} \left(L_\mathrm{bol,peak}\, [erg\,s^{-1}] \right)$', fontsize=14)

	ct = ax.contour(X, Y, Zvals, levels=[1e44, 1e45, 1e46, 1e47, 1e48], colors="black",linestyles='dashed')
	ax.text(2300, 7.4e55, r"$10^{48}$", fontsize = 10, rotation=8, c="black")
	ax.text(2300, 9.5e54, r"$10^{47}$", fontsize = 10, rotation=10, c="black")
	ax.text(2300, 1.35e54, r"$10^{46}$", fontsize = 10, rotation=5, c="black")
	ax.text(1.4e4, 2.6e53, r"$10^{45}$", fontsize = 10, rotation=12, c="black")
	ax.text(4.5e4, 1.5e53, r"$10^{44}$", fontsize = 10, rotation=20, c="black")
	#fmt = ticker.LogFormatterMathtext()
	#fmt.create_dummy_axis()
	#ax.clabel(ct, inline=1, fmt=fmt)
	# insert hatches region where Lshock> Lbol,max and thus wher eout modle kind of breaks down:
	ct2 = ax.contourf(X, Y, Zvals2, levels=[1e-8,1.], colors="grey",hatches=["\\\\", "."], extend="lower", alpha=0.7)
	ct3 = ax.contour(X, Y, Zvals2, levels=[1.], colors="black",linestyles='-')

	#fmt = ticker.LogFormatterMathtext()
	#fmt.create_dummy_axis()
	#ax.clabel(ct, inline=True)

	collapsing_H_models = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2] # Fujibayashi+Nagele
	collapsing_He_models = [FujHe1,FujHe2,FujHe4] # Fujibayashi
	exploding_models = [NagExp] # Nagele
	pulsating_models = [NagPul1,NagPul2] # Nagele
	
	
	x_mass = [x[2]/M_sun_gram for x in collapsing_H_models]
	y_Ekineje = [x[1] for x in collapsing_H_models]
	ax.scatter(x_mass, y_Ekineje, marker="d", c="black")
	#
	x_mass = [x[2]/M_sun_gram for x in collapsing_He_models]
	y_Ekineje = [x[1] for x in collapsing_He_models]
	ax.scatter(x_mass, y_Ekineje, marker="s", c="black")
	#
	x_mass = [x[2]/M_sun_gram for x in exploding_models]
	y_Ekineje = [x[1] for x in exploding_models]
	ax.scatter(x_mass, y_Ekineje, marker="*", c="black")
	#
	x_mass = [x[2]/M_sun_gram for x in pulsating_models]
	y_Ekineje = [x[1] for x in pulsating_models]
	ax.scatter(x_mass, y_Ekineje, marker="o", c="black")

	# put a marker circle around some models, becasue they could be obstructed by self-ionization of the CSM:
	ax.scatter([FujHe2[2]/M_sun_gram], [FujHe2[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujHe4[2]/M_sun_gram], [FujHe4[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujH2[2]/M_sun_gram], [FujH2[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujH4[2]/M_sun_gram], [FujH4[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujDif1[2]/M_sun_gram], [FujDif1[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	  
	ax.set_title(r'Peak $L_\mathrm{bol}$ in optically thick shock phase', fontsize=14)
	ax.set_xlabel(r'$M_\mathrm{eje}\, [M_\odot]$', fontsize=14) 
	ax.set_ylabel(r'$E_\mathrm{kin,eje}\, [erg]$', fontsize=14)
	ax.set_yscale('log')
	ax.set_xscale('log')
	
	picturename = 'Fig5_grid_Meje_Ekin_eje_Lbol.pdf'
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.05) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)


def SMS_star_grid_luminosity_variability_time():

	lenarrays = 70 # good: 47, 57, 70  ; ugly unphysical artifacts at: 48, 49, 50, 51, 52(small bump), 53, 54, 55, 56
	vals_Meje = np.geomspace(2e3*M_sun_gram,3.5e5*M_sun_gram,lenarrays)
	vals_Ekin_eje = np.geomspace(1e53, 5e56, lenarrays)
	  
	# Creating 2-D grid of features 
	[X, Y] = np.meshgrid(vals_Meje/M_sun_gram, vals_Ekin_eje)
	Zvals = (X*X) * 1.
	Zvals2 = (X*X) * 1.
	fig, ax = plt.subplots(1, 1)

	for i in range(lenarrays):
		for j in range(lenarrays):
			R_0 = 1e4*R_sun_cm
			M_eje = vals_Meje[i]; E_kin_eje = vals_Ekin_eje[j]
			#
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 30.*year
			model.stop_after_opt_thick_phase = True
			model.max_timestep = 2.0*year #0.0175*year
			#model.use_smooth_opacity_decrease = True
			model.integrate_model()
			Zvals[j][i] = model.shock_opt_thick_Lbol_peak_variability_timescale/year
			Zvals2[j][i] = model.shock_opt_thick_Lbol_peak_luminosity/model.shock_transparent_Lbol_initial_luminosity # normalize for stability of contourf
			# to make the plot look nicer:
			if Zvals[j][i] > 10.5: Zvals[j][i] = 10.5
			if Zvals[j][i] < 1e-5: Zvals[j][i] = 1e-5
			if Zvals2[j][i] < 1.1e-8: Zvals2[j][i]=1.1e-8
			print(i, j, Zvals[j][i], M_eje, E_kin_eje)
	  

	print(Zvals)
	print(Zvals2)
	#exit()
	# plots filled contour plot
	contourf_ = ax.contourf(X, Y, Zvals, levels=42, extend='max',vmin = 0, vmax= 10.5, cmap="rainbow")
	fig_cbar = fig.colorbar(contourf_, ticks=np.linspace(0.,10.,11)) #(0.,5.,11)
	fig_cbar.set_label(label=r'$t_\mathrm{var,peak}\, [years]$', fontsize=14)

	ct = ax.contour(X, Y, Zvals, levels=[0.5,1,1.5,3,6,9], colors="black",linestyles='dashed')
	ax.text(1.4e4, 1.6e53, r"$0.5$", fontsize = 10, rotation=25, c="black")
	ax.text(3000, 6.05e53, r"$1.0$", fontsize = 10, rotation=20, c="black")
	ax.text(3000, 1.6e54, r"$1.5$", fontsize = 10, rotation=40, c="black")
	ax.text(6780, 5e55, r"$3.0$", fontsize = 10, rotation=65, c="black")
	ax.text(2.67e4, 1.0e56, r"$6.0$", fontsize = 10, rotation=65, c="black")
	ax.text(5.43e4, 1.5e56, r"$9.0$", fontsize = 10, rotation=65, c="black")
	#fmt = ticker.LogFormatterMathtext()
	#fmt.create_dummy_axis()
	#ax.clabel(ct, inline=1, fmt=fmt)
	# insert hatches region where Lshock> Lbol,max and thus wher eout modle kind of breaks down:
	ct2 = ax.contourf(X, Y, Zvals2, levels=[1e-8,1.], colors="grey",hatches=["\\\\", "."], extend="lower", alpha=0.7)
	ct3 = ax.contour(X, Y, Zvals2, levels=[1.], colors="black",linestyles='-')

	#fmt = ticker.LogFormatterMathtext()
	#fmt.create_dummy_axis()
	#ax.clabel(ct, inline=True)

	collapsing_H_models = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2] # Fujibayashi+Nagele
	collapsing_He_models = [FujHe1,FujHe2,FujHe4] # Fujibayashi
	exploding_models = [NagExp] # Nagele
	pulsating_models = [NagPul1,NagPul2] # Nagele
	
	
	x_mass = [x[2]/M_sun_gram for x in collapsing_H_models]
	y_Ekineje = [x[1] for x in collapsing_H_models]
	ax.scatter(x_mass, y_Ekineje, marker="d", c="black")
	#
	x_mass = [x[2]/M_sun_gram for x in collapsing_He_models]
	y_Ekineje = [x[1] for x in collapsing_He_models]
	ax.scatter(x_mass, y_Ekineje, marker="s", c="black")
	#
	x_mass = [x[2]/M_sun_gram for x in exploding_models]
	y_Ekineje = [x[1] for x in exploding_models]
	ax.scatter(x_mass, y_Ekineje, marker="*", c="black")
	#
	x_mass = [x[2]/M_sun_gram for x in pulsating_models]
	y_Ekineje = [x[1] for x in pulsating_models]
	ax.scatter(x_mass, y_Ekineje, marker="o", c="black")

	# put a marker circle around model FujHe4, becasue it could be obstructed by self-ionization of the CSM:
	ax.scatter([FujHe2[2]/M_sun_gram], [FujHe2[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujHe4[2]/M_sun_gram], [FujHe4[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujH2[2]/M_sun_gram], [FujH2[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujH4[2]/M_sun_gram], [FujH4[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujDif1[2]/M_sun_gram], [FujDif1[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	  
	ax.set_title(r'$L_\mathrm{bol}$ peak variability timescale in rest-frame', fontsize=14)
	ax.set_xlabel(r'$M_\mathrm{eje}\, [M_\odot]$', fontsize=14)
	ax.set_ylabel(r'$E_\mathrm{kin,eje}\, [erg]$', fontsize=14)
	ax.set_yscale('log')
	ax.set_xscale('log')
	
	picturename = 'Fig6_grid_Meje_Ekin_eje_tvar_newmodel_new.pdf'
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.05) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)


def SMS_star_grid_max_eta_factor():

	lenarrays = 61 #for good resolution, should be around 20+, 41 is ok (51 not)
	vals_Meje = np.geomspace(2e3*M_sun_gram,3.5e5*M_sun_gram,lenarrays)
	vals_Ekin_eje = np.geomspace(1e53, 5e56, lenarrays)
	  
	# Creating 2-D grid of features 
	[X, Y] = np.meshgrid(vals_Meje/M_sun_gram, vals_Ekin_eje)
	Zvals = (X*X) * 1.
	Zvals2 = (X*X) * 1.
	fig, ax = plt.subplots(1, 1)

	for i in range(lenarrays):
		for j in range(lenarrays):
			R_0 = 1e4*R_sun_cm
			M_eje = vals_Meje[i]; E_kin_eje = vals_Ekin_eje[j]
			#
			model = SMSclasses.SMSlightcurve_sphericalCSMshock_modified(E_kin_eje, M_eje, R_0, Tion, opacity)
			model.max_integration_time = 300.*year
			model.stop_after_opt_thick_phase = True
			model.max_timestep = 20.*year
			model.integrate_model()
			Zvals[j][i] = model.shock_opt_thick_maximum_eta
			if Zvals[j][i] < 1e-9 or np.isnan(Zvals[j][i]): Zvals[j][i] = 1e-9
			if Zvals[j][i] > 1e3 or np.isnan(Zvals[j][i]): Zvals[j][i] = 1e3
			Zvals2[j][i] = model.shock_opt_thick_Lbol_peak_luminosity/model.shock_transparent_Lbol_initial_luminosity # normalize for stability of contourf
			if Zvals2[j][i] < 1e-5 or np.isnan(Zvals2[j][i]): Zvals2[j][i] = 1e-5
			print(i, j, Zvals[j][i], M_eje, E_kin_eje)
	  

	print(Zvals)
	print(Zvals2)
	#exit()
	# plots filled contour plot
	contourf_ = ax.contourf(X, Y, np.log10(Zvals), levels=28, extend='max',vmin = -8, vmax= 3, cmap="autumn")
	fig_cbar = fig.colorbar(contourf_, ticks=range(-8,3,1))
	fig_cbar.set_label(label=r'$log_{10}(\eta_\mathrm{max})$', fontsize=14)

	ct = ax.contour(X, Y, np.log10(Zvals), levels=[-2,0,2], colors="black",linestyles=['--', "-", "--"])
	#ax.text(1.4e4, 1.6e53, r"$0.5$", fontsize = 10, rotation=25, c="black")
	#ax.text(3000, 7e53, r"$1.0$", fontsize = 10, rotation=20, c="black")
	ax.text(3000, 4.0e54, r"$1$", fontsize = 10, rotation=5, c="black")
	ax.text(3000, 5.3e55, r"$10^2$", fontsize = 10, rotation=5, c="black")
	ax.text(6000, 3.4e53, r"$10^{-2}$", fontsize = 10, rotation=30, c="black")
	#fmt = ticker.LogFormatterMathtext()
	#fmt.create_dummy_axis()
	#ax.clabel(ct, inline=1, fmt=fmt)
	# insert hatches region where Lshock> Lbol,max and thus wher eout modle kind of breaks down:
	ct2 = ax.contourf(X, Y, Zvals2, levels=[1e-8,1.], colors="grey",hatches=["\\\\", "."], extend="lower", alpha=0.7)
	ct3 = ax.contour(X, Y, Zvals2, levels=[1.], colors="black",linestyles='-')

	#fmt = ticker.LogFormatterMathtext()
	#fmt.create_dummy_axis()
	#ax.clabel(ct, inline=True)

	collapsing_H_models = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2] # Fujibayashi+Nagele
	collapsing_He_models = [FujHe1,FujHe2,FujHe4] # Fujibayashi
	exploding_models = [NagExp] # Nagele
	pulsating_models = [NagPul1,NagPul2] # Nagele
	
	
	x_mass = [x[2]/M_sun_gram for x in collapsing_H_models]
	y_Ekineje = [x[1] for x in collapsing_H_models]
	ax.scatter(x_mass, y_Ekineje, marker="d", c="black")
	#
	x_mass = [x[2]/M_sun_gram for x in collapsing_He_models]
	y_Ekineje = [x[1] for x in collapsing_He_models]
	ax.scatter(x_mass, y_Ekineje, marker="s", c="black")
	#
	x_mass = [x[2]/M_sun_gram for x in exploding_models]
	y_Ekineje = [x[1] for x in exploding_models]
	ax.scatter(x_mass, y_Ekineje, marker="*", c="black")
	#
	x_mass = [x[2]/M_sun_gram for x in pulsating_models]
	y_Ekineje = [x[1] for x in pulsating_models]
	ax.scatter(x_mass, y_Ekineje, marker="o", c="black")

	# put a marker circle around model FujHe4, becasue it could be obstructed by self-ionization of the CSM:
	ax.scatter([FujHe2[2]/M_sun_gram], [FujHe2[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujHe4[2]/M_sun_gram], [FujHe4[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujH2[2]/M_sun_gram], [FujH2[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujH4[2]/M_sun_gram], [FujH4[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])
	ax.scatter([FujDif1[2]/M_sun_gram], [FujDif1[1]], marker = "o", facecolors='none', edgecolors='black', sizes=[125])	
	  
	ax.set_title(r'$\eta_{max}$ in optically thick shock phase', fontsize=14)
	ax.set_xlabel(r'$M_\mathrm{eje}\, [M_\odot]$', fontsize=14)
	ax.set_ylabel(r'$E_\mathrm{kin,eje}\, [erg]$', fontsize=14)
	ax.set_yscale('log')
	ax.set_xscale('log')
	
	picturename = 'Fig4_grid_Meje_Ekin_eje_eta_max.pdf'
	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.05) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	print("Picture saved as: " + picturename)

#
def get_comoving_volume_between_redshifts(z1,z2):
	# get cosmology:
	# present radiation density; present matter density; curvature; dark energy density
	Omega_r = 0.0; Omega_k = 0.0; Omega_m = 0.315; Omega_Lambda = 0.685
	H_0 = 67.4
	# integrate:
	def E(z_in): # dimensionless Hubble parameter as function of redshift:
		return 1. / np.sqrt(Omega_r*pow(1.+z_in,4.) + Omega_m*pow(1.+z_in,3.) + Omega_k*pow(1.+z_in,2.) + Omega_Lambda)
	# compute comoving distance d_C and the luminosity distance d_L = (1+z)*d_C:
	res1, err = integrate.quad(E, 0., z1, epsabs=1e-10)
	res2, err = integrate.quad(E, 0., z2, epsabs=1e-10)
	d_H = c0 / (H_0 * 3.2407793e-20) # Hubble distance c/H_0 in cm
	# compute comoving distance in Mpc:
	cm_to_Mpc = 3.086e24
	d_C_z1 = d_H*res1 / cm_to_Mpc
	d_C_z2 = d_H*res2 / cm_to_Mpc
	# comoving volume:
	Vz1 = 4.*np.pi* d_C_z1**3 / 3.
	Vz2 = 4.*np.pi* d_C_z2**3 / 3.
	Vz1z2 = ( Vz2 - Vz1 )

	print(d_C_z1, d_C_z2, d_H*res1, d_H*res2)

	# for Omega_k=0, we can compute the luminosity distance like this:
	#d_lum  = (1+self.z_redshift)*d_C
	#print("res, d_C, d_lum, z:")
	#cm_to_Gly = 9.461e26
	#print(res, d_C/cm_to_Gly, self.d_lum/cm_to_Gly, self.z_redshift)
	return Vz1z2

def compute_rate():
	z1 = 7.; z2= 11.5
	Adeg2 = 53. # sky area in deg^2
	dt_survey = 6. # in years

	V_comoving = get_comoving_volume_between_redshifts(z1,z2)

	V_survey = V_comoving * (Adeg2 / 41252.96) # whole sky normalized over square degrees
	rate_survey = 1. / V_survey / dt_survey
	print("Vcomov [Mpc^3] and rate [1/Mpc^3/yr]:")
	print("{0:.4E}".format(V_survey), "{0:.4E}".format(rate_survey))

# make a plot SMSm rate in Mpc^-3 yr^-1 vs redshift. plot the surveys in this work, 
# plot cosmo simulation results, plot Harikane SFR density+IMF estimate
# see https://arxiv.org/pdf/2309.12049 and https://iopscience.iop.org/article/10.3847/1538-4365/acaaa9#apjsacaaa9eqn12
def plot_cosmological_SMS_rates_with_surveys_and_observations():
	#
	plt.figure(figsize=(6.4,4.8)) #change later maybe
	plt.text(14.8,2e-6, "(a)", fontsize=14)
	#### put our bounds from surveys:
	JWST_col="#ebab2f"; EUCLID_col="#980033"; RST_col="#0b4199"
	errbar_linewidth=1.5
	# brightest cases:
	#EUCLID EWF survey
	z=[7,9.5]; rate=[6.31e-13]
	#plt.plot(z,rate, linewidth=2,color="red",linestyle="-",marker="|",markersize=20)
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#EUCLID EWF+single–filter
	z=[7,12]; rate=[3.53e-13]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#EUCLID EDF survey
	z=[7,10.5]; rate=[1.29e-10]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#EUCLID EDF+single–filter
	z=[7,14]; rate=[7.46e-11]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#RST SN med+deep
	z=[7,10.5]; rate=[1.47e-9]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	#RST SN med+deep+single–filter
	z=[7,13]; rate=[9.51e-10]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	#
	# medium brightest cases:
	#EUCLID EWF survey
	z=[5,5.5]; rate=[22.9e-13]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	line[-1][0].set_linestyle(':')
	#EUCLID EWF+single–filter
	z=[5,6.5]; rate=[8.07e-13]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	line[-1][0].set_linestyle(':')
	#EUCLID EDF survey
	z=[5,7]; rate=[1.70e-10]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	line[-1][0].set_linestyle(':')
	#EUCLID EDF+single–filter
	z=[7,9]; rate=[21.1e-11]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	line[-1][0].set_linestyle(':')
	#RST SN med 
	z=[7,8.5]; rate=[4.84e-9]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	line[-1][0].set_linestyle(':')
	#RST SN med+single–filter
	z=[7,9.5]; rate=[3.05e-9]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	line[-1][0].set_linestyle(':')
	#RST SN deep 
	z=[7,9]; rate=[6.70e-9]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	line[-1][0].set_linestyle(':')
	#RST SN deep+single–filter 
	z=[7,10.5]; rate=[4.11e-9]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	line[-1][0].set_linestyle(':')
	#
	# JWST bound by Moriya:
	z=[10,15]; rate=[7.0e-7]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=JWST_col)
	plt.plot([50,51],[10,10],ls="-",lw=errbar_linewidth, color=JWST_col, label="Moriya 2023")
	#
	#
	# put cosmological simulations values:
	# Agarwal et al.
	xpos=[6]; ypos=[6e-10]
	plt.scatter(xpos,ypos, marker="o", color="grey", label="Agarwal 2012",zorder=10)
	# Chiaki et al.
	xpos=[10]; ypos=[5e-8]
	plt.scatter(xpos,ypos, marker="s", color="grey", label="Chiaki 2023",zorder=10)
	#
	#
	# for the legend of EUCLID and RST:
	plt.plot([50,51],[10,10],ls="-",lw=errbar_linewidth,color=RST_col, label="Roman ST")
	plt.plot([50,51],[10,10],ls="-",lw=errbar_linewidth,color=EUCLID_col, label="EUCLID")
	#
	#
	# harikane star formation rate measurements in different redshifts:
	colour_surfaces=["#7fc4ed","#7fcd8d","#e5fb7f","#ffad7f"]
	colour_edge_lines = [None]*len(colour_surfaces)
	for i in range(len(colour_surfaces)): colour_edge_lines[i] = "#"+mean_color(colour_surfaces[i],"#"+mean_color(colour_surfaces[i],"#2f4f4f"))
	#print(colour_surfaces, colour_edge_lines)
	# for the brightest cases:
	SFR_factor= 1.8e-8 #4.6e-7#
	#
	xarr = [5, 6.5]
	yarr1 = [10**(-1.8) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-1.4) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[0])
	#
	xarr = [7, 9.5]
	yarr1 = [10**(-2) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-2.6) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[1],zorder=1)
	#
	xarr = [7, 13]
	yarr1 = [10**(-2) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-3.5) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[2],zorder=0)
	#
	xarr = [10, 15]
	yarr1 = [10**(-2.8) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-3.8) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[3])
	#
	# for the medium bright cases:
	SFR_factor= 4.6e-7#
	#
	xarr = [5, 6.5]
	yarr1 = [10**(-1.8) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-1.4) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.8, facecolor="none", hatch="//",lw=errbar_linewidth, edgecolor=colour_edge_lines[0])
	#
	xarr = [7, 9.5]
	yarr1 = [10**(-2) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-2.6) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.8, facecolor="none", hatch="//", lw=errbar_linewidth, edgecolor=colour_edge_lines[1],zorder=1)
	#
	xarr = [7, 13]
	yarr1 = [10**(-2) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-3.5) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.8, facecolor="none", hatch="/", lw=errbar_linewidth, edgecolor=colour_edge_lines[2],zorder=0)
	#
	xarr = [10, 15]
	yarr1 = [10**(-2.8) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-3.8) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.8, facecolor="none", hatch="//", lw=errbar_linewidth, edgecolor=colour_edge_lines[3])

	plt.semilogy()
	#plt.title(r"rate", fontsize=14) # add title to the whole plots
	plt.xlabel(r"redshift $z$", fontsize=14)
	plt.ylabel(r"SMS explosion rate [Mpc$^{-3}$yr$^{-1}$]", fontsize=14)
	plt.xlim(4.5,15.5) # time in yr
	plt.xticks(np.arange(5, 15+1, 1.0))
	plt.ylim(1e-13,1e-5)

	plt.legend(loc="upper left", ncol=1, fontsize = 8)

	picturename = 'paper_plots/SMS_explosion_rate_fig_a.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.1) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	#print("Picture saved as: " + picturename)
	return 0

def plot_cosmological_SMS_rates_with_surveys_and_observations_split_plot():
	#
	plt.figure(figsize=(6.4,3.84)) #change later maybe
	plt.text(14.8,2.5e-6, "(a)", fontsize=14)
	plt.text(13.0,1.e-13, r"brightest SMSs", fontsize=10)
	#### put our bounds from surveys:
	JWST_col="#ebab2f"; EUCLID_col="#980033"; RST_col="#0b4199"
	errbar_linewidth=1.5
	# brightest cases:
	#EUCLID EWF survey
	z=[7,11.5]; rate=[0.43e-13]
	#plt.plot(z,rate, linewidth=2,color="red",linestyle="-",marker="|",markersize=20)
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#EUCLID EWF+single–filter
	z=[7,14]; rate=[0.16e-13]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#EUCLID EDF survey
	z=[7,11.5]; rate=[0.63e-11]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#EUCLID EDF+single–filter
	z=[7,14]; rate=[0.45e-11]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#RST SN med+deep
	z=[7,10.5]; rate=[2.93e-11]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	#RST SN med+deep+single–filter
	z=[7,13]; rate=[1.90e-11]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	#
	#
	# JWST bound by Moriya:
	z=[10,15]; rate=[1.6e-8]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=JWST_col)
	plt.plot([50,51],[10,10],ls="-",lw=errbar_linewidth, color=JWST_col, label="Moriya 2023")
	#
	#
	# put cosmological simulations values:
	# Agarwal et al.
	xpos=[6]; ypos=[6e-10]
	plt.scatter(xpos,ypos, marker="o", color="grey", label="Agarwal 2012",zorder=10)
	# Chiaki et al.
	xpos=[10]; ypos=[5e-8]
	plt.scatter(xpos,ypos, marker="s", color="grey", label="Chiaki 2023",zorder=10)
	#
	#
	# for the legend of EUCLID and RST:
	plt.plot([50,51],[10,10],ls="-",lw=errbar_linewidth,color=RST_col, label="Roman ST")
	plt.plot([50,51],[10,10],ls="-",lw=errbar_linewidth,color=EUCLID_col, label="EUCLID")
	#
	#
	# harikane star formation rate measurements in different redshifts:
	colour_surfaces=["#7fc4ed","#7fcd8d","#e5fb7f","#ffad7f"]
	colour_edge_lines = [None]*len(colour_surfaces)
	for i in range(len(colour_surfaces)): colour_edge_lines[i] = "#"+mean_color(colour_surfaces[i],"#"+mean_color(colour_surfaces[i],"#2f4f4f"))
	#print(colour_surfaces, colour_edge_lines)
	# for the brightest cases:
	SFR_factor= 1.8e-8 #4.6e-7#
	#
	xarr = [5, 6.5]
	yarr1 = [10**(-1.8) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-1.4) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[0])
	#
	xarr = [7, 9.5]
	yarr1 = [10**(-2) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-2.6) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[1],zorder=1)
	#
	xarr = [7, 13]
	yarr1 = [10**(-2) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-3.5) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[2],zorder=0)
	#
	xarr = [10, 15]
	yarr1 = [10**(-2.8) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-3.8) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[3])
	#
	# add text descriptions for the bounds:
	plt.text(5.95,2.0e-14, r"EWF", fontsize=10, color=EUCLID_col)
	plt.text(6.01,3e-12, r"EDF", fontsize=10, color=EUCLID_col)
	plt.text(5.51,1.7e-11, r"RST SN", fontsize=10, color=RST_col)
	plt.text(11,2.3e-8, r"JWST", fontsize=10, color=JWST_col)
	
	plt.semilogy()
	#plt.title(r"rate", fontsize=14) # add title to the whole plots
	#plt.xlabel(r"redshift $z$", fontsize=14)
	plt.ylabel(r"SMS explosion rate [Mpc$^{-3}$yr$^{-1}$]", fontsize=14)
	plt.xlim(4.5,15.5) # time in yr
	plt.xticks(np.arange(5, 15+1, 1.0))
	# y ticks:
	ax = plt.gca()
	nticks = 10
	maj_loc = ticker.LogLocator(numticks=nticks)
	min_loc = ticker.LogLocator(subs='all', numticks=nticks)
	ax.yaxis.set_major_locator(maj_loc)
	ax.yaxis.set_minor_locator(min_loc)
	plt.ylim(1e-14,1e-5)

	plt.legend(loc="upper left", ncol=1, fontsize = 7.5)

	picturename = 'Fig11a_SMS_explosion_rate_eta_model.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.03) #bbox_inches='tight', pad_inches=0.016
	plt.close()
	#print("Picture saved as: " + picturename)
	#
	# --------------------------------------------------------------------------------------------------------
	#
	plt.figure(figsize=(6.4,3.84)) #change later maybe
	plt.text(14.8,2.5e-6, "(b)", fontsize=14)
	plt.text(11.6,1.e-13, r"moderately bright SMSs", fontsize=10)
	#### put our bounds from surveys:
	JWST_col="#ebab2f"; EUCLID_col="#980033"; RST_col="#0b4199"
	errbar_linewidth=1.5
	# medium brightest cases:
	#EUCLID EWF survey
	z=[5,5.5]; rate=[13.7e-13]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#line[-1][0].set_linestyle(':')
	#EUCLID EWF+single–filter
	z=[5,6.5]; rate=[4.85e-13]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#line[-1][0].set_linestyle(':')
	#EUCLID EDF survey
	z=[5,7]; rate=[10.2e-11]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#line[-1][0].set_linestyle(':')
	#EUCLID EDF+single–filter
	z=[7,9]; rate=[12.6e-11]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=EUCLID_col)
	#line[-1][0].set_linestyle(':')
	#RST SN med 
	z=[7,8]; rate=[1.42e-9]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	#line[-1][0].set_linestyle(':')
	#RST SN med+single–filter
	z=[7,9.5]; rate=[6.08e-10]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	#line[-1][0].set_linestyle(':')
	#RST SN deep 
	z=[7,9]; rate=[1.34e-9]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	#line[-1][0].set_linestyle(':')
	#RST SN deep+single–filter 
	z=[7,10.5]; rate=[8.20e-10]
	line=plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True,capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=RST_col)
	#line[-1][0].set_linestyle(':')
	#
	# JWST bound by Moriya:
	z=[10,15]; rate=[1.6e-7]
	plt.errorbar((z[1]+z[0])/2., rate, xerr=(z[1]-z[0])/2., yerr=0.3*rate[0], lolims=True, capsize=2*errbar_linewidth, elinewidth=errbar_linewidth, markeredgewidth=errbar_linewidth, color=JWST_col)
	plt.plot([50,51],[10,10],ls="-",lw=errbar_linewidth, color=JWST_col, label="Moriya 2023")
	#
	#
	# put cosmological simulations values:
	# Agarwal et al.
	xpos=[6]; ypos=[6e-10]
	plt.scatter(xpos,ypos, marker="o", color="grey", label="Agarwal 2012",zorder=10)
	# Chiaki et al.
	xpos=[10]; ypos=[5e-8]
	plt.scatter(xpos,ypos, marker="s", color="grey", label="Chiaki 2023",zorder=10)
	#
	#
	# for the legend of EUCLID and RST:
	plt.plot([50,51],[10,10],ls="-",lw=errbar_linewidth,color=RST_col, label="Roman ST")
	plt.plot([50,51],[10,10],ls="-",lw=errbar_linewidth,color=EUCLID_col, label="EUCLID")
	#
	#
	# harikane star formation rate measurements in different redshifts:
	colour_surfaces=["#7fc4ed","#7fcd8d","#e5fb7f","#ffad7f"]
	colour_edge_lines = [None]*len(colour_surfaces)
	for i in range(len(colour_surfaces)): colour_edge_lines[i] = "#"+mean_color(colour_surfaces[i],"#"+mean_color(colour_surfaces[i],"#2f4f4f"))
	#print(colour_surfaces, colour_edge_lines)
	# for the medium bright cases:
	SFR_factor= 4.6e-7#
	#
	xarr = [5, 6.5]
	yarr1 = [10**(-1.8) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-1.4) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.8, facecolor="none", hatch="//",lw=errbar_linewidth, edgecolor=colour_edge_lines[0])
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[0])
	#
	xarr = [7, 9.5]
	yarr1 = [10**(-2) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-2.6) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.8, facecolor="none", hatch="//", lw=errbar_linewidth, edgecolor=colour_edge_lines[1],zorder=1)
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[1])
	#
	xarr = [7, 13]
	yarr1 = [10**(-2) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-3.5) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.8, facecolor="none", hatch="/", lw=errbar_linewidth, edgecolor=colour_edge_lines[2],zorder=0)
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[2])
	#
	xarr = [10, 15]
	yarr1 = [10**(-2.8) *SFR_factor]*2 # lower bound = rho_SFR*IMF
	yarr2 = [10**(-3.8) *SFR_factor]*2 # upper bound = rho_SFR*IMF
	#plt.fill_between(xarr, yarr1, yarr2, alpha = 0.8, facecolor="none", hatch="//", lw=errbar_linewidth, edgecolor=colour_edge_lines[3])
	plt.fill_between(xarr, yarr1, yarr2, alpha = 0.4, color=colour_surfaces[3])
	#
	# add text descriptions for the bounds:
	plt.text(6.7,5e-13, r"EWF", fontsize=10, color=EUCLID_col)
	plt.text(6.7,2.5e-11, r"EDF", fontsize=10, color=EUCLID_col)
	plt.text(8.3,6e-9, r"RST SN", fontsize=10, color=RST_col)
	plt.text(11,2.3e-7, r"JWST", fontsize=10, color=JWST_col)

	plt.semilogy()
	#plt.title(r"rate", fontsize=14) # add title to the whole plots
	plt.xlabel(r"redshift $z$", fontsize=14)
	plt.ylabel(r"SMS explosion rate [Mpc$^{-3}$yr$^{-1}$]", fontsize=14)
	plt.xlim(4.5,15.5) # time in yr
	plt.xticks(np.arange(5, 15+1, 1.0))
	# y ticks:
	ax = plt.gca()
	nticks = 10
	maj_loc = ticker.LogLocator(numticks=nticks)
	min_loc = ticker.LogLocator(subs='all', numticks=nticks)
	ax.yaxis.set_major_locator(maj_loc)
	ax.yaxis.set_minor_locator(min_loc)
	plt.ylim(1e-14,1e-5)

	plt.legend(loc="upper left", ncol=1, fontsize = 7.5)

	picturename = 'Fig11b_SMS_explosion_rate_eta_model.pdf'

	#plt.show()
	plt.savefig(picturename,dpi=300,bbox_inches='tight', pad_inches=0.03) #bbox_inches='tight', pad_inches=0.016
	plt.close()

	return 0

# ==============================================================================================================================
def make_figure_1(): # nice looking cartoon plot
	SMS_explosion_summary_graphic()
#
def make_figure_2(): # nice looking cartoon plot
	density_distribution_summary_graphic()
#
def make_figure_3(): # Lbol plots:
	##collapsing_H_models = [FujH1,FujH2,FujH4,FujDif1,NagCol1,NagCol2] # Fujibayashi+Nagele
	##collapsing_He_models = [FujHe1,FujHe2,FujHe4] # Fujibayashi
	##exploding_models = [NagExp] # Nagele
	##pulsating_models = [NagPul1,NagPul2] # Nagele
	model_arr = [FujH1,FujHe1,NagCol1,NagCol2,NagExp,NagPul1,NagPul2]
	name_arr = ["FujH1","FujHe1","NagCol1","NagCol2","NagExp","NagPul1","NagPul2"]
	picname='Fig3a_Lbol_eta_small.pdf'
	multiple_star_Luminosity_evolution(model_arr, name_arr,picname,"upper right", 2)
	# models with eta>1 somewhere:
	model_arr = [FujH2,FujH4,FujHe4,FujDif1,FujHe2]
	name_arr = ["FujH2","FujH4","FujHe4","FujDif1","FujHe2"]
	picname='Fig3b_Lbol_eta_large.pdf'
	multiple_star_Luminosity_evolution(model_arr, name_arr,picname,"lower center", 1)
	
#
def make_figure_4(): # max eta grid plot
	SMS_star_grid_max_eta_factor()
#
def make_figure_5(): # peak Lbol grid plot
	SMS_star_grid_max_luminosity()
#
def make_figure_6(): # peak Lbol timescale grid plot
	SMS_star_grid_luminosity_variability_time()
#
def make_figure_7(): # single AB magnitude plots at different z
	
	single_star_colour_evolution_JWST(FujH1, "FujH1", 7, [0,120], [30,18], 6)
	single_star_colour_evolution_JWST(FujH1, "FujH1", 10, [0,160], [32,20], 5)
	single_star_colour_evolution_JWST(FujH1, "FujH1", 15, [0,200], [32,20], 4)
	single_star_colour_evolution_JWST(FujH1, "FujH1", 20, [0,225], [32,20], 3)
	#
	single_star_colour_evolution_JWST(NagPul2, "NagPul2", 7, [0,100], [32,22], 5)
	single_star_colour_evolution_JWST(NagPul2, "NagPul2", 10, [0,100], [32,22], 4)
	single_star_colour_evolution_JWST(NagPul2, "NagPul2", 15, [0,120], [32,22], 3)
	single_star_colour_evolution_JWST(NagPul2, "NagPul2", 20, [0,140], [32,22], 2)
	
	# dont use FujHe4, use either FujHe2 or Fuj H4:
	#single_star_colour_evolution_JWST(FujHe2, "FujHe2", 7, [0,120], [30,18], 7)
	#single_star_colour_evolution_JWST(FujHe2, "FujHe2", 10, [0,160], [32,20], 5)
	#single_star_colour_evolution_JWST(FujHe2, "FujHe2", 15, [0,225], [32,22], 4)
	#single_star_colour_evolution_JWST(FujHe2, "FujHe2", 20, [0,225], [32,22], 3)
	#
	single_star_colour_evolution_JWST(FujH4, "FujH4", 7, [0,1400], [30,16], 7, use_insert=True) # tune the plot ranges again
	single_star_colour_evolution_JWST(FujH4, "FujH4", 10, [0,2000], [32,18], 5)
	single_star_colour_evolution_JWST(FujH4, "FujH4", 15, [0,2500], [32,20], 4)
	single_star_colour_evolution_JWST(FujH4, "FujH4", 20, [0,3500], [32,20], 3)
	#
	#
	##single_star_colour_evolution_JWST(NagCol1, "NagCol1", 7, [0,120], [30,18], 7)
	##single_star_colour_evolution_JWST(NagCol1, "NagCol1", 10, [0,160], [32,20], 5)
	##single_star_colour_evolution_JWST(NagCol1, "NagCol1", 15, [0,225], [32,22], 4)
	##single_star_colour_evolution_JWST(NagCol1, "NagCol1", 20, [0,225], [32,22], 3)
#
def make_figure_8(): # color magnitude tracks in JWST
	picname = "Fig8a.pdf"
	tracks_F277W_F444W_vs_F444_diagram_at_different_redshifts_with_time_ticks_eta0(picname)
	picname = "Fig8b.pdf"
	tracks_F277W_F444W_vs_F444_diagram_at_different_redshifts_with_time_ticks_eta1(picname)


def make_figure_9(): # color magnitude tracks for EUCLID, and Roman space telescope. use different z-range and models?
	# new:
	picname = "Fig9_EUCLID_a"
	tracks_EUCLID_Jband_Hband_vs_Hband_diagram_at_different_redshifts_with_time_ticks_eta0(picname)
	picname = "Fig9_EUCLID_b"
	tracks_EUCLID_Jband_Hband_vs_Hband_diagram_at_different_redshifts_with_time_ticks_eta1(picname)
	picname = "Fig9_RST_a"
	tracks_RST_Jband_Hband_vs_Hband_diagram_at_different_redshifts_with_time_ticks_eta0(picname)
	picname = "Fig9_RST_b"
	tracks_RST_Jband_Hband_vs_Hband_diagram_at_different_redshifts_with_time_ticks_eta1(picname)

#
def make_figure_10(): # ionized region plot
	model_arr = [FujH4,FujHe4,FujH2,FujH1,NagCol1,NagPul2] # collection of models
	name_arr = ["FujH4","FujHe4","FujH2","FujH1","NagCol1","NagPul2"]
	picname='Fig10_Stroemgen_sphere.pdf'
	multiple_star_StroemgenSphere_CSM(model_arr,name_arr,picname)

def make_figure_11():
	#plot_cosmological_SMS_rates_with_surveys_and_observations()
	plot_cosmological_SMS_rates_with_surveys_and_observations_split_plot()
#
def make_figure_appendix_B(): # figures for model/system validation and varying CSM
	picname='FigAppendixB1_Lbol_vary_Ekin_eje.pdf'
	multiple_star_Luminosity_evolution_systematics_vary_Ekin(picname, [1e-3,8e2], [1e42,5e48])
	picname='FigAppendixB2_Lbol_vary_Meje.pdf'
	multiple_star_Luminosity_evolution_systematics_vary_Meje(picname, [4e-3,7e2], [5e41,5e48])
	picname='FigAppendixB3_Lbol_vary_R0.pdf'
	multiple_star_Luminosity_evolution_systematics_vary_R0(picname, [1e-4,1e3], [1e41,1e48])
	picname='FigAppendixB4_Lbol_vary_n_density.pdf'
	multiple_star_Luminosity_evolution_systematics_vary_n_density(picname, [1e-2,2e2], [1e42,1e48])
	picname='FigAppendixB5_Lbol_vary_CSM.pdf'
	multiple_star_Luminosity_evolution_systematics_vary_CSM(picname, [1e-2,2e2], [1e42,1e48])
	#
	# figure including the early time luminosity peak:
	picname='FigAppendixB6_eta_model_filter_tracks_F277W_F444W_vs_F444_with_timeticks_at_redshift_eta_small_init_peak.pdf'
	tracks_F277W_F444W_vs_F444_diagram_at_different_redshifts_with_time_ticks_eta0(picname, cut_t0_peak=False)
#
#
#
def make_data_discussion_VMS():
	VMS_880 = [1e3*R_sun_cm, 2.9e53, 88*M_sun_gram]
	VMS_6600 = [1e3*R_sun_cm, 2.2e54, 660*M_sun_gram]
	model_arr = [VMS_880,VMS_6600]
	name_arr = ["VMS_880","VMS_6600"]
	picname='Lbol_VMS_eta_model.pdf'
	multiple_star_Luminosity_evolution(model_arr, name_arr,picname)
	#
	#tracks_EUCLID_Jband_Hband_vs_Hband_diagram_at_different_redshifts() # need to modify the model arr by hand to find luminosity
#
#
# ==============================================================================================================================
if __name__ == "__main__":
	
	# fig 5 6 and 7 might take 10min-1hour to run. the others take less than a minute each.
	make_figure_1()
	make_figure_2()
	make_figure_3()

	make_figure_4()
	make_figure_5()
	make_figure_6()
	
	make_figure_7()
	make_figure_8()
	make_figure_9()
	make_figure_10()
	make_figure_11()
	make_figure_appendix_B()
	
	#make_data_discussion_VMS()

	#----------------------------------------------------------------
	exit()