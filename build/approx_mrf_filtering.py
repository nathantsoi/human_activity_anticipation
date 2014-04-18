import sys
import getopt
import os
import pickle 
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib.backends.backend_pdf import PdfPages

def getEntropy(ArrL,ArrV):
	UArr=dict()
	CArr=dict()
	for i in range(ArrL.size):
		if ArrL[i] in UArr.keys():
			UArr[ArrL[i]]=(UArr[ArrL[i]]*CArr[ArrL[i]]+ArrV[i])/float(CArr[ArrL[i]]+1)
			CArr[ArrL[i]]+=1
		else:
			UArr[ArrL[i]]=ArrV[i]
			CArr[ArrL[i]]=1
	Ent = 0
	Summ=0
	PDF = np.array(UArr.values())
	PDF = PDF/PDF.sum()
	return - np.sum(PDF * np.log(PDF))
def ComputeMacro(MAT,NUM):
	PR = 0
	REC = 0
	for i in range(NUM):
		FN=0
		FP=0
		TP = 0
		TP += MAT[i,i]
		for j in range(NUM):
			if ( not (j == i)):
				FN += MAT[i,j]
				FP += MAT[j,i]
		PR += TP/float(TP+FP)
		REC += TP/float(TP+FN)
	return PR/float(NUM),REC/float(NUM)

def Plot(prec,out_name):
	y1=[];y2=[];y3=[];y4=[];
	for i in prec:
		y1.append(prec[i][0])
		y2.append(prec[i][1])
		y3.append(prec[i][2])
		y4.append(prec[i][3])

	y1 = np.array(y1)
	y2 = np.array(y2)
	y3 = np.array(y3)
	y4 = np.array(y4)

	print "VAR:",np.std(y1)
	print "VAR:",np.std(y2)
	print "VAR:",np.std(y3)
	print "VAR:",np.std(y4)

	plt.clf()
	x = np.array(range(1,len(prec)+1))
	y = np.ones(len(prec))*1.05
	for i in range(0,len(prec),2):
		y[i]=0
	labels = prec.keys()

	plt.bar(x-0.5,y,width=1,linewidth=0,alpha=0.4,color='gray')
	plt.axhline(linewidth=3,color='r',y= y1.mean(),alpha=0.8)
	plt.axhline(linewidth=3,color='g',y= y2.mean(),alpha=0.8)
	plt.axhline(linewidth=3,color='b',y= y3.mean(),alpha=0.8)
	plt.axhline(linewidth=3,color='m',y= y4.mean(),alpha=0.8)

	plt.plot(x, y1-0.002,'ro', label='W/O Temporal',alpha=0.75)
	plt.plot(x,y2 ,'go',label='W/ Temporal',alpha=0.75)
	plt.plot(x,y3+0.005,'bo' ,label='Oracle',alpha=0.75)
	plt.plot(x,y4+0.002,'mo' ,label='Anticipation w/ Samples',alpha=0.75)
	print "DMBEST:",y4.mean()
	print "MRF:",y2.mean()
	print "ORACLE:",y3.mean()
	print "MAP",y1.mean()
	plt.ylim([0,1.05])
	plt.xlim([0.5,31.5])
	#plt.plot(x, yO1, 'ro',label='W/O Temporal',alpha=0.75)
	#plt.plot(x, yO2, 'go',label='W/ Temporal',alpha=0.75)
	#plt.plot(x, yO3, 'bo',label='Div10Best',alpha=0.75)
	plt.xlabel('video id')
	plt.ylabel('precision')
	plt.legend(loc=4)
	# You can specify a rotation for the tick labels in degrees or with keywords.
	plt.xticks(x, labels, rotation='vertical')
	# Pad margins so that markers don't get clipped by the axes
	plt.margins(0.2)
	# Tweak spacing to prevent clipping of tick-labels
	plt.subplots_adjust(bottom=0.15)
	# plt.show()
	pp = PdfPages(out_name)
	plt.savefig(pp, format='pdf')
	pp.close()

def testFile(f_name,Pred,PredMRF,Labels,M):
	#Exponentiate and normalize probabilities
	Summ = 0
	for i in xrange(M):
		Pred['val'][i] = math.exp(Pred['val'][i]) 
		Summ = Summ + Pred['val'][i]
	for i in xrange(10):
		Pred['val'][i]=(Pred['val'][i])/float(Summ)

	[PAMap,PAT1,POMap,POT1,CMAPA,CMAPO]=solveMAP(f_name,Labels,Pred)
	[PAMrf,PAT2,POMrf,POT2,CMRFA,CMRFO]=solveMAP(f_name,Labels,PredMRF)
	[PADBest,PAT3,PODBest,POT3,CMBESTA,CMBESTO]=SolveSuccEst(f_name,Labels,Pred,False)
	[PADBestO,PAT4,PODBestO,POT4,CORA,CORO]=SolveSuccEst(f_name,Labels,Pred,True)


	return PAMap/PAT1,POMap/POT1,PAMrf/PAT2,POMrf/POT2,PADBest/PAT3,PODBest/POT3,PADBestO/PAT4,PODBestO/POT4,PAT1,PAT2,PAT3,PAT4,POT1,POT2,POT3,POT4,CMAPA,CMAPO,CMRFA,CMRFO,CMBESTA,CMBESTO,CORA,CORO


def SinglePassSuccEstF1(Labels,f_name,Pred,NumObj,ProbA,ProbO,TranA,TranO):
	for i in xrange(len(Labels[1][f_name])):
		Res = Labels[1][f_name][i+1]			
		if (i<1):
			for k in xrange(10):
				if Pred['labels'][k][2*(i+1)-1][0]==Pred['labels'][0][2*(i+1)-1][0]:
					ProbA[k,i]=1
				else:
					ProbA[k,i]=0
				for m in xrange(NumObj):
					ProbO[m,k,i]=1
		else:
			for k in xrange(10):
				ProbA[k,i]=0
				for m in xrange(NumObj):
					ProbO[m,k,i]=0
				for l in xrange(10):
					ProbA[k,i]+=ProbA[l,i-1]*TranA[Pred['labels'][l][2*(i)-1][0]-1,Pred['labels'][k][2*(i+1)-1][0]-1]
					for m in xrange(NumObj):
						ProbO[m,k,i]+=ProbO[m,l,i-1]*TranO[Pred['labels'][l][2*(i-1)][m]-1,Pred['labels'][k][2*(i)][0]-1]
		if True: #(i < len(Labels[1][f_name]) - 2):
			for k in xrange(10):
				ProbA[k,i]*=Pred['val'][k]
				for m in xrange(NumObj):
					ProbO[m,k,i]*=Pred['val'][k]
	#			if Pred['val'][k] < 0 :
	#				print "VAL",Pred['val']

		Summ = 0
		Summa = np.zeros((NumObj,1))
		for k in xrange(10):
			Summ+=ProbA[k,i]
			for m in xrange(NumObj):
				Summa[m,0]+=ProbO[m,k,i]
		if Summ<0.00001:
			Summ=1
		for m in xrange(NumObj):
			if Summa[m,0]<0.00001:
				Summa[m,0]=1
		for k in xrange(10):
			ProbA[k,i]=ProbA[k,i]/float(Summ)
			for m in xrange(NumObj):
				ProbO[m,k,i]=ProbO[m,k,i]/Summa[m,0]
	return ProbA,ProbO

def SinglePassSuccEstB1(Labels,f_name,Pred,NumObj,ProbA,ProbO,inTranA,inTranO):
	for i in range(len(Labels[1][f_name])-1,-1,-1):	
		#Res = Labels[1][f_name][i+1]		
		if (i==(len(Labels[1][f_name])-1)):
			for k in xrange(10):
				if Pred['labels'][k][2*(i+1)-1][0]==Pred['labels'][0][2*(i+1)-1][0]:
					ProbA[k,i]=1
				else:
					ProbA[k,i]=0
				for m in xrange(NumObj):
					ProbO[m,k,i]=1
		else:
			for k in xrange(10):
				ProbA[k,i]=0
				for m in xrange(NumObj):
					ProbO[m,k,i]=0
				for l in xrange(10):
					ProbA[k,i]+=ProbA[l,i+1]*inTranA[Pred['labels'][l][2*(i+2)-1][0]-1,Pred['labels'][k][2*(i+1)-1][0]-1]
					for m in xrange(NumObj):
						ProbO[m,k,i]+=ProbO[m,l,i+1]*inTranO[Pred['labels'][l][2*(i+1)][m]-1,Pred['labels'][k][2*(i)][m]-1]

		if True: #(i < len(Labels[1][f_name])-2):					
			for k in xrange(10):
				ProbA[k,i]*=Pred['val'][k]
				for m in xrange(NumObj):
					ProbO[m,k,i]*=Pred['val'][k]

		Summ = 0
		Summa = np.zeros((NumObj,1))
		for k in xrange(10):
			Summ+=ProbA[k,i]
			for m in xrange(NumObj):
				Summa[m,0]+=ProbO[m,k,i]
		if Summ<0.00001:
			Summ=1
		for m in xrange(NumObj):
			if Summa[m,0]<0.00001:
				Summa[m,0]=1
		for k in xrange(10):
			ProbA[k,i]=ProbA[k,i]/float(Summ)
			for m in xrange(NumObj):
				ProbO[m,k,i]=ProbO[m,k,i]/Summa[m,0]
	return ProbA,ProbO

def SinglePassSuccEstFR(Labels,f_name,Pred,NumObj,ProbA,ProbO,TranA,TranO):
	for i in xrange(len(Labels[1][f_name])):
		if (i==0):
			Res = 0
		else:
			for k in xrange(10):
				ProbA[k,i]=0
				for l in xrange(10):
					ProbA[k,i]+=ProbA[l,i-1]*TranA[Pred['labels'][l][2*(i)-1][0]-1,Pred['labels'][k][2*(i+1)-1][0]-1]
					for m in xrange(NumObj):
						ProbO[m,k,i]=0
					for m in xrange(NumObj):
						ProbO[m,k,i]+=ProbO[m,l,i-1]*TranO[Pred['labels'][l][2*(i-1)][m]-1,Pred['labels'][k][2*(i)][m]-1]
		
		if True: #(i < len(Labels[1][f_name]) - 1):	
			for k in xrange(10):
				ProbA[k,i]*=Pred['val'][k]
				for m in xrange(NumObj):
					ProbO[m,k,i]*=Pred['val'][k]

#				if Pred['val'][k] < 0 :
#					print "VAL",Pred['val']
		Summ = 0
		Summa = np.zeros((NumObj,1))
		for k in xrange(10):
			Summ+=ProbA[k,i]
			for m in xrange(NumObj):
				Summa[m,0]+=ProbO[m,k,i]
		if Summ<0.00001:
			Summ=1
		for m in xrange(NumObj):
			if Summa[m,0]<0.00001:
				Summa[m,0]=1
		for k in xrange(10):
			ProbA[k,i]=ProbA[k,i]/float(Summ)
			for m in xrange(NumObj):
				ProbO[m,k,i]=ProbO[m,k,i]/Summa[m,0]
	return ProbA,ProbO

def SinglePassSuccEstBR(Labels,f_name,Pred,NumObj,ProbA,ProbO,inTranA,inTranO):
	for i in range(len(Labels[1][f_name])-1,-1,-1):	
		if (i==(len(Labels[1][f_name])-1)):
			Res = 0
		else:
			for k in xrange(10):
				ProbA[k,i]=0
				for m in xrange(NumObj):
					ProbO[m,k,i]=0
				for l in xrange(10):
					ProbA[k,i]+=ProbA[l,i+1]*inTranA[Pred['labels'][l][2*(i+2)-1][0]-1,Pred['labels'][k][2*(i+1)-1][0]-1]
					for m in xrange(NumObj):
						ProbO[m,k,i]+=ProbO[m,l,i+1]*inTranO[Pred['labels'][l][2*(i+1)][m]-1,Pred['labels'][k][2*(i)][m]-1]

		if True: ##(i < len(Labels[1][f_name]) - 2):			
			for k in xrange(10):
				ProbA[k,i]*=Pred['val'][k]
				for m in xrange(NumObj):
					ProbO[m,k,i]*=Pred['val'][k]

		Summ = 0
		Summa = np.zeros((NumObj,1))
		for k in xrange(10):
			Summ+=ProbA[k,i]
			for m in xrange(NumObj):
				Summa[m,0]+=ProbO[m,k,i]
		if Summ<0.00001:
			Summ=1
		for m in xrange(NumObj):
			if Summa[m,0]<0.00001:
				Summa[m,0]=1
		for k in xrange(10):
			ProbA[k,i]=ProbA[k,i]/float(Summ)
			for m in xrange(NumObj):
				ProbO[m,k,i]=ProbO[m,k,i]/Summa[m,0]
	return ProbA,ProbO



def SolveSuccEst(f_name,Labels,Pred,Oracle):
	f = open('TranO.bn','rb')
	TranO = pickle.load(f)
	f.close() 

	f = open('TranA.bn','rb')
	TranA = pickle.load(f)
	f.close() 

	f = open('inTranO.bn','rb')
	inTranO = pickle.load(f)
	f.close() 

	f = open('inTranA.bn','rb')
	inTranA = pickle.load(f)
	f.close() 

	#Create matrixes and auxilary variables
	ProbA = np.zeros((10,len(Labels[1][f_name])))
	NumObj = len(Labels[1][f_name][1])-1 
	ProbO = np.zeros((NumObj,10,len(Labels[1][f_name])))

	if False:
		for i in xrange(len(Labels[1][f_name])-1,len(Labels[1][f_name])):
			for k in xrange(10):
				Pred['labels'][k][2*(i+1)-1][0]=Pred['labels'][k][2*i-1][0]
				for m in xrange(NumObj):
					Pred['labels'][k][2*i][m]=Pred['labels'][k][2*(i-1)][m]



	[ProbA,ProbO]=SinglePassSuccEstF1(Labels,f_name,Pred,NumObj,ProbA,ProbO,TranA,TranO)
	#[ProbA,ProbO]=SinglePassSuccEstB1(Labels,f_name,Pred,NumObj,ProbA,ProbO,inTranA,inTranO)

	for iterA in xrange(5):
		[ProbA,ProbO]=SinglePassSuccEstBR(Labels,f_name,Pred,NumObj,ProbA,ProbO,inTranA,inTranO)
		[ProbA,ProbO]=SinglePassSuccEstFR(Labels,f_name,Pred,NumObj,ProbA,ProbO,TranA,TranO)
	

	for i in range(len(Labels[1][f_name])):
		Vec = ProbA[:,i]
		LL =[]
		for j in range(10):
			LL.append(Pred['labels'][j][2*(i)-1][0])
		print 'ENT',i,getEntropy(np.array(LL),Vec)

	ConfMO = np.zeros((12,12))
	ConfMA = np.zeros((10,10))

#	print ProbA
	scipy.io.savemat('test.mat',dict(PA=ProbA))
	# Choose the best
	TPA = 0
	TOTA = 0
	TPO = 0
	TOTO = 0
	for i in xrange(len(Labels[1][f_name])):
		Res = Labels[1][f_name][i+1]
		if i==len(Labels[1][f_name])-1:
			fl = False
			resC = 0
			for k  in xrange(10):
				if (ProbA.max(0)[i-1]==ProbA[k,i-1]):
					resC = Pred['labels'][k][2*(i)-1][0]-1
					if (Pred['labels'][k][2*(i)-1][0]==Res[0]):
						fl=True
			if(fl):
				TPA = TPA + 0
				ConfMA[Res[0]-1,Res[0]-1]+=1
			else:
				Res[0]-1
				ConfMA[Res[0]-1,resC]+=1
			TOTA = TOTA + 0
		else:
			fl = False
			for k  in xrange(10):
				if (Oracle or (ProbA.max(0)[i]==ProbA[k,i])):
					resC = Pred['labels'][k][2*(i+1)-1][0]-1
					if (Pred['labels'][k][2*(i+1)-1][0]==Res[0]):
						fl=True
			if(fl):
				TPA = TPA + 1
				ConfMA[Res[0]-1,Res[0]-1]+=1
			elif(Oracle):
				ConfMA[Res[0]-1,Pred['labels'][0][2*(i+1)-1][0]-1]+=1
			else:
				ConfMA[Res[0]-1,resC]+=1
			TOTA = TOTA + 1
		for m in xrange(NumObj):
			fl = False
			for k  in xrange(10):
				if(Oracle or (ProbO[m].max(0)[i]==ProbO[m,k,i])):
					if(Pred['labels'][k][2*i][m]==Res[1+m]):
						fl=True
						resC = Pred['labels'][k][2*i][m] - 1

			if(fl):
				TPO = TPO + 1
				ConfMO[Res[1+m]-1,Res[1+m]-1]+=1
			elif(Oracle):
				ConfMO[Res[1+m]-1,Pred['labels'][0][2*i][m] - 1]+=1				
			else:
				ConfMO[Res[1+m]-1,resC]+=1
			TOTO = TOTO + 1			
	return TPA,float(TOTA),TPO,float(TOTO),ConfMA,ConfMO

def solveMAP(f_name,Labels,Pred):
	TPO   = 0
	TOTO = 0
	TPA = 0
	TOTA = 0
	ConfMO = np.zeros((12,12))
	ConfMA = np.zeros((10,10))
	for i in xrange(len(Labels[1][f_name])):
		Res = Labels[1][f_name][i+1]
		NumObj = len(Res)-1
		for j in xrange(len(Res)-1):
			ConfMO[Res[1+j]-1,Pred['labels'][0][2*i][j]-1]+=1
			fl = False
			for k  in xrange(1):
				if(Pred['labels'][k][2*i][j]==Res[1+j]):
					fl=True
			if(fl):
				TPO = TPO + 1
			TOTO = TOTO + 1
		ConfMA[Res[0]-1,Pred['labels'][0][2*(i+1)-1][0]-1]+=1
		fl = False
		for k  in xrange(1):
			if(Pred['labels'][k][2*(i+1)-1][0]==Res[0]):
				fl=True
		if(fl):
			TPA = TPA + 1
		TOTA = TOTA + 1

	return TPA,float(TOTA),TPO,float(TOTO),ConfMA,ConfMO


def SolveViterbiMRF(f_name,Labels,Pred):
	f = open('TranO.bn','rb')
	TranO = pickle.load(f)
	f.close() 

	f = open('TranA.bn','rb')
	TranA = pickle.load(f)
	f.close() 


	#Create matrixes and auxilary variables
	ProbA = np.zeros((10,len(Labels[1][f_name]),2))
	NumObj = len(Labels[1][f_name][1])-1 
	ProbO = np.zeros((NumObj,10,len(Labels[1][f_name]),2))


	[ProbA,ProbO]=ViterbiMRFRec(Labels,f_name,Pred,NumObj,ProbA,ProbO,TranA,TranO)

	soll = np.zeros((len(Labels[1][f_name]),1))
	mL = 0
	for m in range(10):
		if( ProbA[m,len(Labels[1][f_name])-2,0]> ProbA[mL,len(Labels[1][f_name])-2,0]):
			mL=m
	soll[len(Labels[1][f_name])-2,0]=mL 
	soll[len(Labels[1][f_name])-1,0]=mL 

	for m in range(len(Labels[1][f_name])-3,-1,-1):
		soll[m,0]=ProbA[soll[m+1,0],m+1,1]

	# Choose the best
	TPA = 0
	TOTA = 0
#	TPO = 0
#	TOTO = 0
	for i in xrange(len(Labels[1][f_name])):
		Res = Labels[1][f_name][i+1]
		fl = False
		if (Pred['labels'][int(soll[i,0])][2*(i+1)-1][0] ==Res[0]):
			fl=True
		if(fl):
			TPA = TPA + 1
		TOTA = TOTA + 1
#		for m in xrange(NumObj):
#			fl = False
#			for k  in xrange(10):
#				if(ProbO[m].max(0)[i]==ProbO[m,k,i]):
#					if(Pred['labels'][k][2*i][m]==Res[1+m]):
#						fl=True
#			if(fl):
#				TPO = TPO + 1
#			TOTO = TOTO + 1			
#	return ,TPO/float(TOTO)
	#print soll[:,0]
	return TPA,float(TOTA),0

def ViterbiMRFRec(Labels,f_name,Pred,NumObj,ProbA,ProbO,TranA,TranO):
	for i in xrange(len(Labels[1][f_name])):
		if (i==0):
			for k in xrange(10):
				ProbA[k,i,0]=1
				ProbA[k,i,1]=0
		else:
			for k in xrange(10):
				ProbA[k,i,0]=0
				Bes = 0
				for l in xrange(10):
					if(ProbA[l,i-1,0]*TranA[Pred['labels'][l][2*(i)-1][0]-1,Pred['labels'][k][2*(i+1)-1][0]-1]>=ProbA[Bes,i-1,0]*TranA[Pred['labels'][Bes][2*(i)-1][0]-1,Pred['labels'][k][2*(i+1)-1][0]-1]):
						Bes = l
				ProbA[k,i,0]=ProbA[Bes,i-1,0]*TranA[Pred['labels'][Bes][2*(i)-1][0]-1,Pred['labels'][k][2*(i+1)-1][0]-1]
				ProbA[k,i,1]=Bes

				for m in xrange(NumObj):
					ProbO[m,k,i,0]=0
				for m in xrange(NumObj):
					ProbO[m,k,i,0]+=ProbO[m,l,i-1,0]*TranO[Pred['labels'][l][2*(i-1)][m]-1,Pred['labels'][k][2*(i)][m]-1]
			
		for k in xrange(10):
			ProbA[k,i,0]*=Pred['val'][k]
			for m in xrange(NumObj):
				ProbO[m,k,i,0]*=Pred['val'][k]

		Summ = 0
		Summa = np.zeros((NumObj,1))
		for k in xrange(10):
			Summ+=ProbA[k,i,0]
			for m in xrange(NumObj):
				Summa[m,0]+=ProbO[m,k,i,0]
		if Summ<0.00001:
			Summ=1
		for m in xrange(NumObj):
			if Summa[m,0]<0.00001:
				Summa[m,0]=1
		for k in xrange(10):
			ProbA[k,i,0]=ProbA[k,i,0]/float(Summ)
			for m in xrange(NumObj):
				ProbO[m,k,i,0]=ProbO[m,k,i,0]/Summa[m,0]
	return ProbA,ProbO
	

def main(files_name,fold):
	f = open(files_name,'r')
	lines = f.read().split(',')
	lines.pop(len(lines)-1)
	f.close

	precA = dict()
	precO = dict()

	dur = 0
	durc =0
	cF = 0
	PT = np.zeros((1,16))
	GCMAPA=np.zeros((10,10))
	GCMAPO=np.zeros((12,12))
	GCMRFA=np.zeros((10,10))
	GCMRFO=np.zeros((12,12))
	GCMBESTA=np.zeros((10,10))
	GCMBESTO=np.zeros((12,12))
	GCORA=np.zeros((10,10))
	GCORO=np.zeros((12,12))

	for f_name in lines:
		if (not os.path.exists('./'+fold+f_name+'_dmbest.bn')):
			print 'PO',f_name
			continue
		elif(not os.path.exists('./'+fold+f_name+'_mrf.bn')):
			print 'PO',f_name
			continue

		cF+=1
		f = open('./'+fold+f_name+'_dmbest.bn','rb')
		Pred = pickle.load(f)
		f.close

		f = open('./'+fold+f_name+'_mrf.bn','rb')
		PredMRF = pickle.load(f)
		f.close

		f = open('LabelingBinary.bn','rb')
		Labels = pickle.load(f)
		f.close() 

		[PAMap,POMap,PAMrf,POMrf,PADBest,PODBest,PADBestO,PODBestO,PAT1,PAT2,PAT3,PAT4,POT1,POT2,POT3,POT4,CMAPA,CMAPO,CMRFA,CMRFO,CMBESTA,CMBESTO,CORA,CORO]=testFile(f_name,Pred,PredMRF,Labels,10)
		precA[f_name]=[PAMap,PAMrf,PADBestO,PADBest]
		precO[f_name]=[POMap,POMrf,PODBestO,PODBest]

		GCMAPA+=CMAPA
		GCMAPO+=CMAPO
		GCMRFA+=CMRFA
		GCMRFO+=CMRFO
		GCMBESTA+=CMBESTA
		GCMBESTO+=CMBESTO
		GCORA+=CORA
		GCORO+=CORO

		dur+=len(Labels[1][f_name])
		durc+=1
		#print len(Labels[1][f_name])
		PT[0,0]+=PAT1*PAMap
		PT[0,1]+=PAT1
		PT[0,2]+=POT1*POMap
		PT[0,3]+=POT1
		PT[0,4]+=PAT2*PAMrf
		PT[0,5]+=PAT2
		PT[0,6]+=POT2*POMrf
		PT[0,7]+=POT2
		PT[0,8]+=PAT3*PADBest
		PT[0,9]+=PAT3
		PT[0,10]+=POT3*PODBest
		PT[0,11]+=POT3
		PT[0,12]+=PAT4*PADBestO
		PT[0,13]+=PAT4
		PT[0,14]+=POT4*PODBestO
		PT[0,15]+=POT4
#	print "M",dur/float(durc)
	print 'Activity'
	Plot(precA,'ActRes2.pdf')
	print 'Object'
	Plot(precO,'ObjRes2.pdf')
	print cF
	for i in range(8):
		print i,PT[0,i*2]/PT[0,i*2+1]
	print 'MAP',ComputeMacro(GCMAPA,10),ComputeMacro(GCMAPO,12)
	print 'MRF',ComputeMacro(GCMRFA,10),ComputeMacro(GCMRFO,12)
	print 'MBEST',ComputeMacro(GCMBESTA,10),ComputeMacro(GCMBESTO,12)
	print 'ORACLE',ComputeMacro(GCORA,10),ComputeMacro(GCORO,12)
if __name__ == "__main__":
	print "Test 11"
	main('files.txt','Test11/')
	print "Test 12"
	main('files.txt','Test12/')
	print "Test 21"
	main('files.txt','Test21/')
	print "Test 22"
	main('files.txt','Test22/')