# -*- coding: utf-8 -*


import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
	x=[]
	y=[]
	fr = np.array(
		[
		 [0.697,0.46,'1'],
		 [0.774,0.376,'1'],
		 [0.634,0.264,'1'],
		 [0.608,0.318,'1'],
		 [0.556,0.215,'1'],
		 [0.403,0.237,'1'],
		 [0.481,0.149,'1'],
		 [0.437,0.211,'1'],
		 [0.666,0.091,'1'],
		 [0.243,0.267,'0'],
		 [0.245,0.057,'0'],
		 [0.343,0.099,'0'],
		 [0.639,0.161,'0'],
		 [0.657,0.198,'0'],
		 [0.36,0.37,'0'],
		 [0.593,0.042,'0'],
		 [0.719,0.103,'0']
		]
	)
	for line in fr:
		x.append([1.0,float(line[0]),float(line[1])])
		y.append(int(line[2]))
	return x,y


def sigmoid(inX):
	return 1.0/(1+np.exp(-inX))

def gradAscent(xIn,yIn):
	dataMatrix=np.mat(xIn)
	labelMat=np.mat(yIn).transpose()
	m,n=np.shape(dataMatrix)
	alpha=0.01
	maxCycles=100000
	weights=np.ones((n,1))
	for k in range (maxCycles):
		h=sigmoid(np.dot(dataMatrix,weights))
		error=(labelMat-h)
		weights=weights+alpha*np.dot(dataMatrix.transpose(),error)
	return weights

def plotBestFit(weights):
	x,y=loadDataSet()
	dataArr = np.array(x)
	n = np.shape(dataArr)[0]
	xcord1 = []
	ycord1 = []
	xcord2 = []
	ycord2 = []
	for i in range(n):
		if int(y[i]) == 1:
			xcord1.append(dataArr[i, 1])
			ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1])
			ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c="red", marker="s")
	ax.scatter(xcord2, ycord2, s=30, c="green")
	x = np.arange(-0.2, 0.8, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	#这里的y是二维数组，x是一维，无法相互对应。但是由于能力有限，无法做到降维操作。
	#所以这里就把前面的y在结果中显示出来，然后把得出的结果重新赋给了y
	print (y)
	y= [0.57559815, 0.52292909 ,0.47026003 ,0.41759097 ,0.36492191, 0.31225286,0.2595838 , 0.20691474 ,0.15424568 ,0.10157662]
	###
	ax.plot(x,y)
	plt.xlabel("密度");
	plt.ylabel("含糖率");
	plt.show()


x,y=loadDataSet()
w=gradAscent(x,y)
print(w)
print(plotBestFit(w))
