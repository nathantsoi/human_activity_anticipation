import sys
import getopt
import os
import pickle 
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import scipy
import scipy.io
import time


def getEntropy(ArrL,ArrV):
	UArr=dict()
	CArr=dict()
	for i in range(ArrL.size):
		if True:#ArrV[i]>0:
			if ArrL[i] in UArr.keys():
				UArr[ArrL[i]]=(UArr[ArrL[i]]*CArr[ArrL[i]]+ArrV[i])/float(CArr[ArrL[i]]+1)
				CArr[ArrL[i]]+=1.0
			else:
				UArr[ArrL[i]]=float(ArrV[i])
				CArr[ArrL[i]]=1.0
	PDF = np.array(UArr.values())
	PDF = PDF/PDF.sum()
	#if math.isnan(- np.sum(PDF * np.log(PDF))):
	#	print 'V'
	#	print ArrL,ArrV
	return - np.sum(PDF * np.log(PDF))



def ComputeMacro(MAT,NUM):
	PR = 0
	REC = 0
	nn=NUM
	for i in range(NUM):
		FN=0
		FP=0
		TP = 0
		TP += MAT[i,i]
		for j in range(NUM):
			if ( not (j == i)):
				FN += MAT[i,j]
				FP += MAT[j,i]
		if TP+FP ==0:
			nn-=1
			continue
		if TP+FN ==0:
			nn-=1
			continue
		PR += TP/float(TP+FP)
		REC += TP/float(TP+FN)
	return PR/float(nn),REC/float(nn)

def SinglePassSuccEstF1(Labels,f_name,Pred,NumObj,ProbA,ProbO,TranA,TranO,N):
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
		if i < (len(Labels[1][f_name]) - N):
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
		if Summ<1e-10:
			Summ=1
		for m in xrange(NumObj):
			if Summa[m,0]<0.00000001:
				Summa[m,0]=1
		for k in xrange(10):
			ProbA[k,i]=ProbA[k,i]/float(Summ)
			for m in xrange(NumObj):
				ProbO[m,k,i]=ProbO[m,k,i]/Summa[m,0]
	return ProbA,ProbO

def SinglePassSuccEstFR(Labels,f_name,Pred,NumObj,ProbA,ProbO,TranA,TranO,N):
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
		
		if (i < len(Labels[1][f_name]) - N):	
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
		if Summ<1e-10:
			Summ=1
		for m in xrange(NumObj):
			if Summa[m,0]<0.00001:
				Summa[m,0]=1
		for k in xrange(10):
			ProbA[k,i]=ProbA[k,i]/float(Summ)
			for m in xrange(NumObj):
				ProbO[m,k,i]=ProbO[m,k,i]/Summa[m,0]
	return ProbA,ProbO

def SinglePassSuccEstBR(Labels,f_name,Pred,NumObj,ProbA,ProbO,inTranA,inTranO,N):
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

		if (i < len(Labels[1][f_name]) - N):			
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
		if Summ<1e-10:
			Summ=1
		for m in xrange(NumObj):
			if Summa[m,0]<0.00001:
				Summa[m,0]=1
		for k in xrange(10):
			ProbA[k,i]=ProbA[k,i]/float(Summ)
			for m in xrange(NumObj):
				ProbO[m,k,i]=ProbO[m,k,i]/Summa[m,0]
	return ProbA,ProbO



def SolveSuccEst(f_name,Labels,Pred,Oracle,N):
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

	if True:
		for i in xrange(len(Labels[1][f_name])-N+1,len(Labels[1][f_name])):#xrange(len(Labels[1][f_name])-1,len(Labels[1][f_name])-N,-1):
			for k in xrange(10):
				Pred['labels'][k][2*(i+1)-1][0]=k+1#Pred['labels'][k][2*(i)-1][0]#k+1#Pred['labels'][k][2*(i)-1][0] #? Might change
				for m in xrange(NumObj):
					Pred['labels'][k][2*i][m]=Pred['labels'][k][2*(i-1)][m] #[2*(len(Labels[1][f_name])-4)][m]


#	if True:
#		for i in xrange(4,len(Labels[1][f_name])):
#			for k in xrange(10):
#				Pred['labels'][k][2*(i+1)-1][0]=k+1#Pred['labels'][k][2*(len(Labels[1][f_name])-3)-1][0]
#				for m in xrange(NumObj):
#					Pred['labels'][k][2*i][m]=Pred['labels'][k][2*3][m] #[2*(len(Labels[1][f_name])-4)][m]


	[ProbA,ProbO]=SinglePassSuccEstF1(Labels,f_name,Pred,NumObj,ProbA,ProbO,TranA,TranO,N)
	[ProbA,ProbO]=SinglePassSuccEstBR(Labels,f_name,Pred,NumObj,ProbA,ProbO,inTranA,inTranO,N)

	for iterA in xrange(4):
		#[ProbA,ProbO]=SinglePassSuccEstBR(Labels,f_name,Pred,NumObj,ProbA,ProbO,inTranA,inTranO)
		[ProbA,ProbO]=SinglePassSuccEstFR(Labels,f_name,Pred,NumObj,ProbA,ProbO,TranA,TranO,N)
	
	ConfMO = np.zeros((12,12))
	ConfMA = np.zeros((10,10))


#	Ent = np.zeros(len(Labels[1][f_name]))
#	for i in range(4,len(Labels[1][f_name])):
#		for m in xrange(NumObj):
#			Vec = ProbA[:,i]
#			LL =[]
#			for j in range(10):
#				LL.append(Pred['labels'][j][2*(i+1)-1][0])
#			Ent[i]+=getEntropy(np.array(LL),Vec)

#	print f_name+'_obj_ent.bn'
#	f=open(f_name+'_obj_ent.bn','wb')
#	pickle.dump(Ent/NumObj,f)
#	f.close()
	scipy.io.savemat('test.mat',dict(PA=ProbA))
	# Choose the best
	TPA = 0
	TOTA = 0
	TPO = 0
	TOTO = 0
	for i in xrange(len(Labels[1][f_name])-N-2,len(Labels[1][f_name])-1):
		Res = Labels[1][f_name][i+1]
		if True:
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
	return TPA/float(TOTA),TPO/float(TOTO),ConfMA,ConfMO

def solveMAPAnt(f_name,Labels,Pred,K):
	TPO   = 0
	TOTO = 0
	TPA = 0
	TOTA = 0
	ConfMO = np.zeros((12,12))
	ConfMA = np.zeros((10,10))

#	print f_name,Pred
	Res = Labels[1][f_name][K-1+1]
	NumObj = len(Res)-1
	for j in xrange(len(Res)-1):
		fl = False
		if len(Pred['labels'][0])<=2*(K-1):
			return -1,-1,-1,-1
		ConfMO[Res[1+j]-1,Pred['labels'][0][2*(K-1)][j]-1]+=1
		if(Pred['labels'][0][2*(K-1)][j]==Res[1+j]):
			fl=True
		if(fl):
			TPO = TPO + 1
		TOTO = TOTO + 1
	ConfMA[Res[0]-1,Pred['labels'][0][2*(K)-1][0]-1]+=1
	fl = False
	for k  in xrange(1):
		if(Pred['labels'][k][2*(K-1+1)-1][0]==Res[0]):
			fl=True
	if(fl):
		TPA = TPA + 1
	TOTA = TOTA + 1
	return TPA/float(TOTA),TPO/float(TOTO),ConfMA,ConfMO


def solveDMBESTAnt(f_name,Labels,Pred,i,N):
	for k in  Labels[1][f_name].keys():
		if k>(i+1):
			Labels[1][f_name].pop(k)
	return SolveSuccEst(f_name,Labels,Pred,False,N)


def main(files_name,fold):
	f = open('f.txt')
	lines = f.read().split()
	linesH = lines

	f = open('LabelingBinary.bn','rb')
	Labels = pickle.load(f)
	f.close() 

	PRH = dict()
	PRM = dict()

	CFMA = np.zeros((10,10))
	CFMO = np.zeros((12,12))

	PTA=0
	PTO=0
	PTTA=0
	if False:
		for ll in lines:
			f_name = ll[0:10]
			print fold+ll
			f = open(fold+ll,'rb')
			PredMRF = pickle.load(f)
			f.close
			#print ll,PredMRF['labels']
			if f_name in PRH.keys():
				RES0,RES1,ConfMA,ConfMO = solveMAPAnt(f_name,Labels,PredMRF,int(ll.split('_')[1])) 
				if (RES0)==-1:
					continue
			 	PRH[f_name].append([int(ll.split('_')[1]),[RES0,RES1]])
			 	CFMA+=ConfMA
			 	CFMO+=ConfMO
				PTO+= RES1
				PTA+=RES0
				PTTA +=1
			else:
				PRH[f_name] = []
				RES0,RES1,ConfMA,ConfMO = solveMAPAnt(f_name,Labels,PredMRF,int(ll.split('_')[1])) 
				if (RES0)==-1:
					continue
			 	PRH[f_name].append([int(ll.split('_')[1]),[RES0,RES1]])
			 	CFMA+=ConfMA
			 	CFMO+=ConfMO
				PTO+= RES1
				PTA+=RES0
				PTTA +=1
		#print solveMAP(f_name,Labels,PredMRF,int(ll.split('_')[1]))
		#print PRH.keys()
		print 'P',PRH
		print PTO/PTTA
		print PTA/PTTA
		print ComputeMacro(CFMO,12)
		print ComputeMacro(CFMA,10)
	#exit()
	#for f_name in PRH.keys():
	f = open('files.txt','r')
	lines = f.read().split(',')
	lines.pop(len(lines)-1)
	f.close
	TA=0
	TO=0
	TTA=0
	GXA=np.zeros((10,10))
	GXO=np.zeros((12,12))

	if True:
		#for f_name in lines:
		if True:
			f_name = '0510172333'
			f = open('./'+'Test21/'+f_name+'_dmbest.bn','rb')
			Pred = pickle.load(f)
			Summ = 0
			for i in xrange(10):
				Pred['val'][i] = math.exp(Pred['val'][i]) 
				Summ = Summ + Pred['val'][i]
			for i in xrange(10):
				Pred['val'][i]=(Pred['val'][i])/float(Summ)
			f.close
			N = 3
			PRM[f_name]=[]
#			for i in xrange(len(Labels[1][f_name])-1,(N+1),-1):
			for i in xrange(len(Labels[1][f_name])-1,len(Labels[1][f_name])-2,-1):
				if len(Labels[1][f_name]) < (N+1):
					continue
				f = open('LabelingBinary.bn','rb')
				Labels = pickle.load(f)
				f.close() 
				start = time.clock()
				AHM0,AHM1,AHM2,AHM3=solveDMBESTAnt(f_name,Labels,Pred,7,N)
				print "My Time:" ,time.clock() - start
				PRM[f_name].append([i,[AHM0,AHM1]])
				GXA+=AHM2
				GXO+=AHM3
				TO+=AHM1
				TA+=AHM0
				TTA+=1
		print 'PRM',TTA,PTTA
		#f = open('PRM.bn','wb')
		#pickle.dump(PRM,f)
		#f.close()
		print "O",TA/TTA,TO/TTA
		#print "H",PTA/PTTA,PTO/PTTA
		print ComputeMacro(GXA,10)
		print ComputeMacro(GXO,12)
		exit()
	PAR = np.zeros(30)
	PC = np.zeros(30)
	PARA = np.zeros(30)
	
	print 'HH',linesH
	for ll in linesH:
		f_name = ll[0:10]
		print f_name
		if(len(PRH[f_name])>0):
			for i in PRH[f_name]:
					PARA[i[0]]+=i[1][0]
					PAR[i[0]]+=i[1][1]
					PC[i[0]]+=1
	for j in range(30):
			PAR[j]=PAR[j]/PC[j]
			PARA[j]=PARA[j]/PC[j]
	print 'HemaA',PARA
	print 'HemaO',PAR
	print 'C',PC

	exit()
	PAR = np.zeros(25)
	PC = np.zeros(25)
	PARA = np.zeros(25)


	for f_name in lines:
		if(len(PRM[f_name])>0):
			for i in PRH[f_name]:
#					print 'I',i
					PARA[i[0]]+=i[1][0]
					PAR[i[0]]+=i[1][1]
					PC[i[0]]+=1
	for j in range(25):
			PAR[j]=PAR[j]/PC[j]
			PARA[j]=PARA[j]/PC[j]
	PO = dict()
	PO['A']=PARA
	PO['O']=PAR
	PO['C']=PC
	print PO
	print ComputeMacro(GXA,10)
	print ComputeMacro(GXO,12)

#	with open('ant.bn', 'wb') as f:
#		pickle.dump(PO, f)
#	print CFMA
#	print CFMO
if __name__ == "__main__":
   main('files.txt','BOB/')