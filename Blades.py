
# Author: Felipe M. Pasquali
# Contact: felipeme-at-buffalo.edu
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import csv


# TODO:
# Create class for cylinder
# Integrate cylinder stations into the mesh
# Create output for Abaqus file or CAD
# 

def PlotGrid(X,Y):

	fig = plt.figure()
	m,n = np.shape(X)
	for i in range(0,m):
		plt.plot(X[i,:],Y[i,:])
	for j in range(0,n):
		plt.plot(X[:,j],Y[:,j])

	plt.show()

class Cylinder(object):
	"""docstring for Cylinder"""
	def __init__(self, arg):
		super(Cylinder, self).__init__()
		self.arg = arg
		

class Spline(object):
	"""Defines a cubic interpolated spline
		INPUT: 
		points - list of points to interpolate the spline with [need to be sorted from min to max, non repeating]
		numinterpol - number of representations when evaluating.
	"""
	def __init__(self,points,numinterpol=10000):
		self.points = points # Input as a list containint X[0] and Y[1]
		self.numrepres = numinterpol
		# print self.points[0], self.points[1]
		self.curve = interpolate.splrep(self.points[0],self.points[1],s=0,k=3)
		self.xnew = np.linspace(np.min(points[0]),np.max(points[0]),self.numrepres)
		self.ynew =interpolate.splev(self.xnew,self.curve)

	def plot(self):
		# plt.figure()
		plt.plot(self.xnew,self.ynew)
		# plt.show()
	def GetPoints(self,s):
		s = np.min(self.xnew) + s*(np.max(self.xnew)-np.min(self.xnew))
		return np.array([s,interpolate.splev(s,self.curve)])


class SimpleVLine(object):
	"""Defines a simple line between two points
		INPUT: 
		points - list of two points to interpolate
		numinterpol - number of representations when evaluating.
	"""
	def __init__(self, points, numinterpol=100):
		super(SimpleVLine, self).__init__()
		self.points = points # Sorted in descending Y
		self.numrepres = numinterpol
		self.X = points[0]
		self.Y = points[1]
		self.xnew = np.linspace(self.X[1],self.X[0],self.numrepres)
		self.ynew = self.LinInterp(self.xnew)
		# # self.ynew = self.Y[1] + (self.Y[0] - self.Y[1])*np.linspace(0,1,self.numrepres)
		# self.ynew = self.Y[1] + (self.Y[0] - self.Y[1])*
	def plot(self):
		plt.plot(self.xnew,self.ynew)
	
	def LinInterp(self,s):
		return self.Y[1] + (self.Y[0] - self.Y[1])*(s-self.X[1])/(self.X[0]-self.X[1]) 

	def GetPoints(self,s):
		s = self.X[1] +(s*(self.X[0]-self.X[1]))
		return np.array([s , self.LinInterp(s)])
		

class Airfoil(object):
	"""Defines normalized Airfoil Shape by importing the points from txtfile
		INPUT - path to airfoil file with normalized coordinates X/C and Y/C
	"""
	def __init__(self, data):
		super(Airfoil, self).__init__()
		self.data = data
		self.X, self.Y = np.loadtxt(data,float,unpack=True,comments='#')

	def split(self):
		return np.argmin(self.X) #102 for NACA64618, denotes the middle of airfoil
	def length(self):
		return self.X.size
	def plot(self):
		# plt.figure()
		plt.plot(self.X,self.Y,'o')
		# plt.show()

class AirfoilTop(Spline):
	"""
	Defines a representation of the airfoil top surface for mapping the structured mesh
	"""
	def __init__(self,airfoil):
		self.X = airfoil.X[airfoil.split()-1::-1]
		self.Y = airfoil.Y[airfoil.split()-1::-1]
		self.points = [self.X,self.Y]
		# print self.points
		Spline.__init__(self,self.points,numinterpol=10000)

class AirfoilBottom(Spline):
	"""
	Defines a representation of the airfoil bottom surface for mapping the structured mesh
	"""
	def __init__(self, airfoil):
		self.X = airfoil.X[airfoil.split()+1:airfoil.length()]
		self.Y = airfoil.Y[airfoil.split()+1:airfoil.length()] 
		self.points = [self.X,self.Y]
		# print self.points
		Spline.__init__(self,self.points,numinterpol=10000)


class AirfoilLeft(SimpleVLine):
	"""
	Defines a representation of the airfoil left surface for mapping the structured mesh
	"""
	def __init__(self, airfoil):
		self.airfoil = airfoil
		self.X = airfoil.X[[airfoil.split()-1,airfoil.split()+1]]
		self.Y = airfoil.Y[[airfoil.split()-1,airfoil.split()+1]]
		self.points = [self.X,self.Y]
		SimpleVLine.__init__(self,self.points,numinterpol=10000)

class AirfoilRight(SimpleVLine):
	"""	
	Defines a representation of the airfoil left surface for mapping the structured mesh
	"""
	def __init__(self,airfoil):
		self.airfoil = airfoil
		self.X = airfoil.X[[0,airfoil.length()-1]]
		self.Y = airfoil.Y[[0,airfoil.length()-1]]
		# print self.X, self.Y
		self.points = [self.X,self.Y]
		SimpleVLine.__init__(self,self.points,numinterpol=10000)	
		

class Station(object):
	"""
	Defines Station of the Airfoil
	Input - Parameters read from blade schedule file.
	"""
	def __init__(self,nnx,nny,RNodes,AeroTwist,Chord,FoilName):
		super(Station, self).__init__()
		self.nnx, self.nny = nnx, nny
		self.RNodes = float(RNodes)
		self.AeroTwist = float(AeroTwist)
		self.Chord = float(Chord)
		self.FoilName = FoilName
		self.Airfoil = Airfoil("BladeData/"+self.FoilName+".txt")
		# Scale with Chord and center the X with Chord/2
		self.Airfoil.X = self.Airfoil.X*self.Chord - (self.Chord/2)
		self.Airfoil.Y = self.Airfoil.Y*self.Chord
		# Rotate with twist angle
		self.TwistAngle = self.AeroTwist*np.pi/180.0
		self.Airfoil.X = self.Airfoil.X*np.cos(self.TwistAngle) - self.Airfoil.Y*np.sin(self.TwistAngle)
		self.Airfoil.Y = self.Airfoil.X*np.sin(self.TwistAngle) + self.Airfoil.Y*np.cos(self.TwistAngle)

		#Representation of Airfoil
		self.Top = AirfoilTop(airfoil=self.Airfoil)
		self.Bottom = AirfoilBottom(airfoil=self.Airfoil)	
		self.Left = AirfoilLeft(self.Airfoil)
		self.Right = AirfoilRight(self.Airfoil)
		# self.Top.plot()
		# self.Bottom.plot()
		# self.Right.plot()
		# self.Left.plot()
		self.MeshX, self.MeshY = self.TFI(self.nnx, self.nny,self.Bottom,self.Top,self.Left,self.Right)
		# PlotGrid(self.MeshX, self.MeshY)

	def TFI(self,n,m,BottomRep,TopRep,Left,Right):
		#Performs transfinite interpolation given the four representations of blade.
		#Returns X,Y coordinates of the grid. 
		X = np.zeros([n,m])
		Y = np.zeros([n,m])
		xi = np.linspace(0.,1,n)
		eta = np.linspace(0.,1.,m)


		for i in range(0,n):
			Xi = xi[i] 
			for j in range(0,m):
				Eta = eta[j]
				XY = (1-Eta)*BottomRep.GetPoints(Xi)+Eta*TopRep.GetPoints(Xi) + (1-Xi)*Left.GetPoints(Eta) + Xi*Right.GetPoints(Eta) - (Xi*Eta*TopRep.GetPoints(1) + Xi*(1-Eta)*BottomRep.GetPoints(1) + Eta*(1-Xi)*TopRep.GetPoints(0) + (1-Xi)*(1-Eta)*BottomRep.GetPoints(0))
				X[i,j] = float(XY[0])
				Y[i,j] = float(XY[1])
		return X,Y


class Blade(object):
	"""Defines a blade based on blade data"""
	def __init__(self,schedule,mesh=[24,8,100]):
		super(Blade, self).__init__()
		self.schedule = schedule
		self.nelx,self.nely,self.nelz = mesh
		self.nnx,self.nny,self.nnz = self.nelx+1,self.nely+1,self.nelz+1
		# Read the blade schedule
		self.stations = []
		with open (self.schedule, 'rb') as csvfile :
			spamreader = csv.reader (csvfile , delimiter ="\t", quotechar ='|')
			for row in spamreader:
				# Only read rows that do NOT start with the "#"character .
				if ( row [0][0] != '#'):
					RNodes = float(row[0])
					AeroTwist = float(row[1])
					DRNodes = float(row[2])
					Chord = float(row[3])
					NFoil = row[4]
					if NFoil != 'Cylinder1' and NFoil != 'Cylinder2':
						s = Station(self.nnx,self.nny,RNodes, AeroTwist, Chord,NFoil)
						self.stations.append(s)

	def Plot3D(self):
		plt.figure()
		ax = plt.gca(projection='3d')
		for i in range(0,len(self.stations)):
			ax.plot(self.stations[i].Airfoil.X,self.stations[i].Airfoil.Y,self.stations[i].RNodes)
		plt.show()
	def Plot2D(self):
		plt.figure()
		for i in range(0,len(self.stations)):
			plt.plot(self.stations[i].Airfoil.X,self.stations[i].Airfoil.Y)
		plt.show()

class Mesh(object):
	"""Creates a lofted mesh of the wind turbine blade
	Node Numberng
	
	3-----6-----9-----12
	| (2) | (4) | (6)  |
	2-----5-----8-----11	 
 	| (1) | (3) | (5)  |
	1-----4-----7-----10


	"""
	def __init__(self,blade,WriteAbaqus=0):
		super(Mesh, self).__init__()
		self.blade = blade
		self.nelx,self.nely, self.nelz = self.blade.nelx,self.blade.nely,self.blade.nelz
		self.nnx,self.nny, self.nnz = self.blade.nnx,self.blade.nny,self.blade.nnz
		self.nodesPerSection = self.nelz/len(self.blade.stations)
		self.numSections = len(self.blade.stations)-1
		# print "nodes per section", self.nodesPerSection
		#CORRECTED NUMBER OF NODES IN Z
		self.nnz = (self.nodesPerSection)+(self.numSections-1)*(self.nodesPerSection-1)
		# self.nnz = 4
		print self.nnz
		if WriteAbaqus:
			f = open('blade.inp','w+')
			f.write('*Heading\n')
			f.write('** Job name: blade Model name: blade\n')	
			f.write('**\n')
			f.write('** PARTS\n')
			f.write('**\n')		
			f.write('*Part, name=Blade\n')
			f.write('*Node\n')
			f.close()


		plt.figure()
		ax = plt.gca(projection='3d')
		SecHeight = 0
		CurrentH = 0
		nodenum = 0
		for st in range(0,len(self.blade.stations)-1):
		# 	plt.plot(self.stations[i].Airfoil.X,self.stations[i].Airfoil.Y)
			#To do, fix the number of elements in Z
			tau = np.linspace(0.,1,self.nodesPerSection)

			SecHeight = self.blade.stations[st+1].RNodes - self.blade.stations[st].RNodes
			# print "CurrentHeight:", CurrentH
			for i in tau:
				X = (1-i)*self.blade.stations[st].MeshX + i*self.blade.stations[st+1].MeshX
				Y = (1-i)*self.blade.stations[st].MeshY + i*self.blade.stations[st+1].MeshY
				if i ==0 and st == 0:
					ax.scatter(X,Y,i*SecHeight + CurrentH,color = 'blue')
					if WriteAbaqus:
						for xi in range(0,X.shape[0]):
							for eta in range(0,X.shape[1]):
								nodenum += 1
								f = open('blade.inp','a+')
								f.write('%i,%12.7f,%12.7f,%12.7f\n' %(nodenum,X[xi,eta],Y[xi,eta],i*SecHeight + CurrentH))
								f.close()
				if i != 0:
					ax.scatter(X,Y,i*SecHeight + CurrentH, color='blue')
					if WriteAbaqus:
						for xi in range(0,X.shape[0]):
							for eta in range(0,X.shape[1]):
								nodenum += 1
								f = open('blade.inp','a+')
								f.write('%i,%12.7f,%12.7f,%12.7f\n' %(nodenum,X[xi,eta],Y[xi,eta],i*SecHeight + CurrentH))
								f.close()

			CurrentH += self.blade.stations[st+1].RNodes - self.blade.stations[st].RNodes

			
		plt.show()

		if WriteAbaqus:
			f = open('blade.inp','a+')
			f.write('*ELEMENT, TYPE=C3D8, ELSET=ALL\n')
			f.close()
			elNum = 0
			
			for k in range(0,self.nnx*self.nny*(self.nnz-1),self.nnx*self.nny):
				for i in range(0,(self.nnx-1)*self.nny,self.nny):
					for j in range(1,self.nely+1):
						elNum +=1 
						f = open('blade.inp','a+')
						f.write('%d, %d,%d,%d,%d,%d,%d,%d,%d\n' %(elNum,0+i+j+k,self.nny+i+j+k,self.nny+1+i+j+k,1+i+j+k,(self.nnx*self.nny)+i+j+k,(self.nnx*self.nny+self.nny)+i+j+k,(self.nnx*self.nny+self.nny+1)+i+j+k,(self.nnx*self.nny+1)+i+j+k))
						f.close()

if __name__ == '__main__':

	# NACA64618 = Airfoil("BladeData/DU30_A17.txt")
	# # NACA64618.plot()

	# TopRep = AirfoilTop(airfoil=NACA64618)
	# TopRep.plot()
	# BottomRep = AirfoilBottom(airfoil=NACA64618)
	# BottomRep.plot()
	# Left = AirfoilLeft(airfoil=NACA64618)
	# Right = AirfoilRight(airfoil=NACA64618)
	# Left.plot()
	# Right.plot()

	# # Blade Tests
	Blade = Blade("BladeData/BladeSchedule.txt",mesh=[48,16,280])
	Mesh = Mesh(Blade,WriteAbaqus=1)
	# Blade.Plot2D()











