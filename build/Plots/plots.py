import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('p1.pdf')

f = open('files.txt','r')
lines = f.read().split(',')
lines.pop(len(lines)-1)
f.close

f = open('LabelingBinary.bn','rb')
Labels = pickle.load(f)
f.close


#0 MAP 1 MRF 2 Oracle
precO = dict()
precA = dict()

for f_name in lines:
	if (not os.path.exists(f_name+'_dmbest.bn')):
		continue
	elif(not os.path.exists(f_name+'_mrf.bn')):
		continue

	precO[f_name]=[-1,-1,-1]
	precA[f_name]=[-1,-1,-1]

	f = open(f_name+'_dmbest.bn','rb')
	Pred = pickle.load(f)
	f.close

	f = open(f_name+'_mrf.bn','rb')
	PredMRF = pickle.load(f)
	f.close

	TPO   = 0
	TOTO = 0
	TPA = 0
	TOTA = 0
	for i in xrange(len(Labels[1][f_name])):
		Res = Labels[1][f_name][i+1]
		NumObj = len(Res)-1
		for j in xrange(len(Res)-1):
			fl = False
			for k  in xrange(10):
				if(Pred['labels'][k][2*i][j]==Res[1+j]):
					fl=True
			if(fl):
				TPO = TPO + 1
			TOTO = TOTO + 1
		fl = False
		for k  in xrange(10):
			if(Pred['labels'][k][2*(i+1)-1][0]==Res[0]):
				fl=True
		if(fl):
			TPA = TPA + 1
		TOTA = TOTA + 1
	print f_name,"DMBEST Precision Object,Activity:",TPO,TOTO,TPA,TOTA
	precO[f_name][2] = TPO/float(TOTO)
	precA[f_name][2] = TPA/float(TOTA)

	TPO   = 0
	TOTO = 0
	TPA = 0
	TOTA = 0
#	print "R",Labels[1][f_name]
	for i in xrange(len(Labels[1][f_name])):
		Res = Labels[1][f_name][i+1]
		NumObj = len(Res)-1
		for j in xrange(len(Res)-1):
			fl = False
			for k  in xrange(1):
				if(PredMRF['labels'][k][2*i][j]==Res[1+j]):
					fl=True
			if(fl):
				TPO = TPO + 1
			TOTO = TOTO + 1
		fl = False
		for k  in xrange(1):
			if(PredMRF['labels'][k][2*(i+1)-1][0]==Res[0]):
				fl=True
		if(fl):
			TPA = TPA + 1
		TOTA = TOTA + 1
	print f_name,"MRF Precision Object,Activity:",TPO,TOTO,TPA,TOTA
	precO[f_name][1] = TPO/float(TOTO)
	precA[f_name][1] = TPA/float(TOTA)


	TPO   = 0
	TOTO = 0
	TPA = 0
	TOTA = 0

	for i in xrange(len(Labels[1][f_name])):
		Res = Labels[1][f_name][i+1]
		NumObj = len(Res)-1
		for j in xrange(len(Res)-1):
			fl = False
			for k  in xrange(1):
				if(Pred['labels'][k][2*i][j]==Res[1+j]):
					fl=True
			if(fl):
				TPO = TPO + 1
			TOTO = TOTO + 1
		fl = False
		for k  in xrange(1):
			if(Pred['labels'][k][2*(i+1)-1][0]==Res[0]):
				fl=True
		if(fl):
			TPA = TPA + 1
		TOTA = TOTA + 1
	print f_name,"MAP Precision Object,Activity:",TPO,TOTO,TPA,TOTA
	precO[f_name][0] = TPO/float(TOTO)
	precA[f_name][0] = TPA/float(TOTA)

	print ""


yO1=[];yO2=[];yO3=[];yA1=[];yA2=[];yA3=[];
for i in precO:
	yO1.append(precO[i][0])
	yO2.append(precO[i][1])
	yO3.append(precO[i][2])
	yA1.append(precA[i][0])
	yA2.append(precA[i][1])
	yA3.append(precA[i][2])
yO1 = np.array(yO1)
yO2 = np.array(yO2)
yO3 = np.array(yO3)

yA1 = np.array(yA1)
yA2 = np.array(yA2)
yA3 = np.array(yA3)

x = np.array(range(1,len(precO)+1))
y = np.ones(len(precO))*1.05
for i in range(0,len(precO),2):
	y[i]=0
labels = precO.keys()

plt.bar(x-0.5,y,width=1,linewidth=0,alpha=0.4,color='gray')
plt.axhline(linewidth=3,color='r',y=np.mean(yO1),alpha=0.8)
plt.axhline(linewidth=3,color='g',y=np.mean(yO2),alpha=0.8)
plt.axhline(linewidth=3,color='b',y=np.mean(yO3),alpha=0.8)

plt.plot(x, yO1-0.005,'ro', label='W/O Temporal',alpha=0.75)
plt.plot(x,yO2 ,'go',label='W/ Temporal',alpha=0.75)
plt.plot(x,yO3+0.005,'bo' ,label='Div10Best',alpha=0.75)

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
pp = PdfPages('pObject.pdf')
plt.savefig(pp, format='pdf')
pp.close()

plt.clf()

plt.bar(x-0.5,y,width=1,linewidth=0,alpha=0.4,color='gray')
plt.axhline(linewidth=3,color='r',y=np.mean(yA1),alpha=0.8)
plt.axhline(linewidth=3,color='g',y=np.mean(yA2),alpha=0.8)
plt.axhline(linewidth=3,color='b',y=np.mean(yA3),alpha=0.8)

plt.plot(x, yA1-0.005,'ro', label='W/O Temporal',alpha=0.75)
plt.plot(x,yA2 ,'go',label='W/ Temporal',alpha=0.75)
plt.plot(x,yA3+0.005,'bo' ,label='Div10Best',alpha=0.75)

plt.ylim([0,1.05])
plt.xlim([0.5,31.5])

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
pp = PdfPages('pAct.pdf')
plt.savefig(pp, format='pdf')
pp.close()


print len(precO),len(precA)
# Plot precisions