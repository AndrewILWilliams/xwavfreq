#Script written for python 2

import numpy as np
from numpy import linalg as LA
import scipy.io as io
import matplotlib as mpl
from scipy import signal
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import date, timedelta
from scipy import io as io
from scipy import stats
from mpl_toolkits.basemap import Basemap
import datetime as dt

'''Code performs spectrum and regression analysis used by
https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.3709

Paths to data files and your own data formats will require revision.
'''



date1=date(1974,6,1)
L=14000
datelist=[]
datestr=[]
for x in np.arange(L+70):
     datelist.append(date1+dt.timedelta(days=x))
     datestr.append(datelist[x].strftime('%d-%b'))

datearray=np.array(datelist)
cmap=plt.cm.get_cmap("bwr")

Llist=['Lm15','Lm125','Lm10','Lm75','Lm5','Lm25','L0','L25','L5','L75','L10','L125','L15']

#class band is used to hold the phase speed ranges of the 3 filters.
# slow: 2-5 m/s, intermediate: 6-11 m/s, and fast, 12-30 m/s. 
class band: pass

intermediate=band()
intermediate.minspeed=6
intermediate.maxspeed=11

slow=band()
slow.minspeed=2
slow.maxspeed=5

fast=band()
fast.minspeed=12
fast.maxspeed=30


RE=6.371e6

#Load OLR anomaly data (replicate with your own data arrays)
OLRarray=np.zeros((13,14000,144))
for y in np.arange(13):
   file='/roundylab_rit/roundy/aroniadata/olr/Ldata/%s.mat'%Llist[y]
   filed=io.loadmat(file)
   A=filed['A'][:14000,:]
   OLRarray[y,:A.shape[0],:]=A

#Create arrays filtered for symmetry and antisymmetry across the equator.
OLRsymarray=np.zeros((7,14000,144))
OLRantisymarray=np.zeros((6,14000,144))
for y in np.arange(7):
     if y==0:
          OLRsymarray[y,:,:]=OLRarray[7,:,:]
     else:
          OLRsymarray[y,:,:]=(OLRarray[y,:,:]+OLRarray[-y,:,:])/2
          OLRantisymarray[y-1,:,:]=(OLRarray[y,:,:]-OLRarray[-y,:,:])/2

#Create power spectra of OLR data
twindow=np.arange(96) #96 day time windows
nwins=289  #number of windows included (adjust for the length of dataset)
spectrasym=np.zeros((nwins,96,144)) #will hold the set of spectra
spectraantisym=np.zeros((nwins,96,144))

#Construct cosine bell for tapering data windows in time
cosbell=np.expand_dims((((1-np.cos(2*np.pi*twindow/96))+1)/2 - .5),1)
print cosbell.shape
cosbell=np.repeat(cosbell,144,axis=1)
for y in np.arange(7):
     for n in np.arange(nwins):
          print n
	  #detrend window
          window=signal.detrend(OLRsymarray[y,(twindow)+(n*96/2),:],axis=0)
	  #taper window
          window=window*cosbell
	  #Take 2-d discrete Fourier transform of window
	  fftwin=np.fft.fft2(window)
	  spectrasym[n,:,:]=np.fft.fftshift((fftwin*np.conj(fftwin)).real)
	  if y<6:
               window=signal.detrend(OLRantisymarray[y,(twindow)+(n*96/2),:],axis=0)
               window=window*cosbell
	       fftwin=np.fft.fft2(window)
	       spectraantisym[n,:,:]=np.fft.fftshift((fftwin*np.conj(fftwin)).real)

Iout=np.where(spectraantisym[:,10,10]!=0)[0]

ks=np.arange(-72,72)
freqs=-(np.arange(-96/2,96/2)/96.)

background=(spectraantisym[Iout,:,:].mean(axis=0)+spectrasym.mean(axis=0))/2
backgrnd0=np.delete(background,(48),axis=0)

def one21(X):
     '''1-2-1 filter'''
     x=np.arange(1,143)
     f=np.arange(1,94)
     X[:,x]=(X[:,x-1]+2*X[:,x]+X[:,x+1])/4
     X[f,:]=(X[f-1,:]+2*X[f,:]+X[f+1,:])/4
     return X


#Run one21 30 times to create background spectrum.
for nn in np.arange(30):
    backgrnd0=one21(backgrnd0)


background=np.insert(backgrnd0,(48),0,axis=0)
plt.figure()
plt.contourf(ks,freqs,np.log(background),20,cmap=cmap,vmin=12,vmax=20)
plt.title('Smoothed Background Spectrum')
plt.xlabel('Westward    Zonal Wavenumber    Eastward')
plt.ylabel('Frequency (CPD)')
plt.axis([-14,14,0,.5])
plt.savefig('/pr11/roundy/public_html/exam5.png')

plt.figure()
plt.contourf(ks,freqs,np.log(spectrasym.mean(axis=0)),20,cmap=cmap,vmin=12,vmax=20)
plt.colorbar()
plt.axis([-14,14,0,.5])
plt.savefig('/pr11/roundy/public_html/exam4.png')
plt.clf()
plt.contourf(ks,freqs,np.log(spectraantisym[Iout,:,:].mean(axis=0)),20,cmap=cmap,vmin=12,vmax=20)
plt.colorbar()
plt.axis([-14,14,0,.5])
plt.savefig('/pr11/roundy/public_html/exam3.png')


#speedsdiag is an array of phase speeds corresponding to each 
#wavenumber and frequency (it's only correct for eastward-moving disturbances)
speedsdiag=np.zeros((96,144))
for x in np.arange(144):
    for t in np.arange(96):
         speedsdiag[t,x]=freqs[t]/ks[x]*2*np.pi*RE/86400



Ks=np.arange(144)
N=OLRarray.shape[1]
freqslong=(np.arange(N)/float(N))
speeds=np.zeros((N,144))
for x in np.arange(144/2):
    for t in np.arange(N/2):
         speeds[t,-x]=freqslong[t]/Ks[x]*2*np.pi*RE/86400
	 if t!=0 and x!=0:
              speeds[-t,x]=freqslong[t]/Ks[x]*2*np.pi*RE/86400
plt.figure()
plt.contourf(Ks,freqslong,speeds,[10,40],cmap=cmap)
plt.axis([-14,14,0,.5])
plt.savefig('/pr11/roundy/public_html/exam6.png')


plt.figure()
plt.contourf(ks,freqs,np.log(spectrasym.mean(axis=0)),20,cmap=cmap)
symnorm=spectrasym.mean(axis=0)/background
plt.contour(ks,freqs,symnorm,np.arange(2.2,4.4,.4),colors='k')
fil1=np.logical_and(speedsdiag>fast.minspeed,speedsdiag<fast.maxspeed)
fil2=np.logical_and(speedsdiag>intermediate.minspeed,speedsdiag<intermediate.maxspeed)
fil3=np.logical_and(speedsdiag>slow.minspeed,speedsdiag<slow.maxspeed)

for x in np.arange(96):
   if np.abs(freqs[x])>.3:
        fil1[x,:]=0
        fil2[x,:]=0
        fil3[x,:]=0
for x in np.arange(144):
     if np.abs(ks[x])<2:
        fil1[:,x]=0
        fil2[:,x]=0
        fil3[:,x]=0
     if np.abs(ks[x])>10:
        fil1[:,x]=0
        fil2[:,x]=0
        fil3[:,x]=0
        

plt.contour(ks,freqs,fil1.astype(float),[0.5],colors='g',linestyles='dashdot')
plt.contour(ks,freqs,fil2.astype(float),[0.5],colors='r')
plt.contour(ks,freqs,fil3.astype(float),[0.5],colors='b',linestyles='dashed')
plt.axis([-14,14,0,.5])
plt.plot([-15,15],[0.03333,0.03333],linewidth=2,color='k',linestyle='--')
plt.xlabel('Westward   Zonal Wavenumber   Eastward')
plt.ylabel('Frequency (CPD)')
plt.title('Spectrum of Equatorially Symmetric\n OLR Anomalies')
plt.savefig('/pr11/roundy/public_html/exam2.png')
plt.savefig('figure1.eps',format='eps')


eqolr=np.mean(OLRarray[3:10,:,:],axis=0)
ffteqolr=np.fft.fft2(eqolr)
ffteqolrfil=np.zeros_like(ffteqolr)

I=np.logical_and(speeds>slow.minspeed,speeds<slow.maxspeed)
Ix=np.where(np.abs(Ks)==1)[0]
If=np.where(np.logical_and(freqslong>0.3,freqslong<0.5))[0] 
I[:,Ix]=0
I[:,-Ix]=0
I[If,:]=0
I[-If,:]=0
for x in np.arange(144):
     if np.abs(Ks[x])<2:
        I[:,x]=0
     if np.abs(Ks[x])>10:
        I[:,x]=0
Ispeeds=np.where(I)
ffteqolrfil[np.where(I)]=ffteqolr[np.where(I)]
plt.figure()

""" 
# AILW: Not sure why this is here, 
# it just replicates np.fft.fftfreq(),
# and isn't used anywhere...

Ksind=Ks
Ik=np.where(Ksind>72)[0]
Ksind[Ik]=144-Ksind[Ik]
freqsind=freqslong
If=np.where(freqsind>0.5)[0]
freqsind[If]=1-freqsind[If]
Ksind=np.fft.fftshift(Ksind)
freqsind=np.fft.fftshift(freqsind)
"""

filtered=np.fft.ifft2(ffteqolrfil)

print 'slowest'
print np.sum(I)

X=np.arange(0,360,2.5)
slow.filtered=np.fft.ifft2(ffteqolrfil).real
plt.figure(10,figsize=(9.5,7.5))
plt.subplot(1,3,3)
plt.contourf(X,datearray[5000:5100],slow.filtered[5000:5100,:],30,vmin=-30,vmax=30,cmap=cmap)
plt.title('c. Filtered OLR Anomaly, '+str(slow.minspeed)+' - '+str(slow.maxspeed)+' m/s')
plt.xlabel('Longitude (Degrees East)')
plt.tick_params(axis='both', which='both', bottom='true', top='false', labelbottom='true', right='false', left='true', labelleft='false')
plt.colorbar()


I=np.logical_and(speeds>intermediate.minspeed,speeds<intermediate.maxspeed)
Ix=np.where(np.abs(Ks)==1)[0]
If=np.where(np.logical_and(freqs>0.3,freqs<0.5))[0]
I[:,Ix]=0
I[:,-Ix]=0
I[If,:]=0
I[-If,:]=0
Ix=np.where(np.abs(Ks)>10)[0]
I[:,Ix]=0
I[:,-Ix]=0
for x in np.arange(144):
     if np.abs(Ks[x])<2:
        I[:,x]=0
     if np.abs(Ks[x])>10:
        I[:,x]=0


ffteqolrfil=np.zeros_like(ffteqolr)
Ispeeds=np.where(I)
ffteqolrfil[np.where(I)]=ffteqolr[np.where(I)]
plt.subplot(1,3,2)
intermediate.filtered=np.fft.ifft2(ffteqolrfil).real
print 'slow'
print np.sum(I)


plt.contourf(X,np.arange(datearray[5000:5100].shape[0]),intermediate.filtered[5000:5100,:],30,vmin=-30,vmax=30,cmap=cmap)
plt.xlabel('Longitude (Degrees East)')
plt.tick_params(axis='both', which='both', bottom='true', top='false', labelbottom='true', right='false', left='true', labelleft='false')
plt.title('b. Filtered OLR Anomaly, '+str(intermediate.minspeed)+' - '+str(intermediate.maxspeed)+' m/s')
#plt.colorbar()

I=np.logical_and(speeds>fast.minspeed,speeds<fast.maxspeed)
Ix=np.where(np.abs(Ks)==1)[0]
If=np.where(np.logical_and(freqs>0.3,freqs<0.5))[0]
I[:,Ix]=0
I[:,-Ix]=0
I[If,:]=0
I[-If,:]=0
Ix=np.where(np.abs(Ks)>10)[0]
I[:,Ix]=0
I[:,-Ix]=0
for x in np.arange(144):
     if np.abs(Ks[x])<2:
        I[:,x]=0
     if np.abs(Ks[x])>10:
        I[:,x]=0


ffteqolrfil=np.zeros_like(ffteqolr)
Ispeeds=np.where(I)
ffteqolrfil[np.where(I)]=ffteqolr[np.where(I)]
plt.subplot(1,3,1)
fast.filtered=np.fft.ifft2(ffteqolrfil).real
print 'fast'
print np.sum(I)
plt.contourf(X,np.arange(datearray[5000:5100].shape[0]),fast.filtered[5000:5100,:],30,vmin=-30,vmax=30,cmap=cmap)
plt.xlabel('Longitude (Degrees East)')
plt.tick_params(axis='both', which='both', bottom='true', top='false', labelbottom='true', right='false', left='true', labelleft='true')
plt.title('a. Filtered OLR Anomaly, '+str(fast.minspeed)+' - '+str(fast.maxspeed)+' m/s')
#plt.colorbar()
plt.savefig('/pr11/roundy/public_html/exam9.png')


'''Read in Data. The manuscript used OLR and reanalysis data from the
author's storage format. Replicating the analysis requires filling 
25x14000x144 arrays (latitude x time x longitude). There is no necessity
to replicate the exact period included (14000 days following June 1, 1974). 
Replication with more modern reanalysis products is encouraged, but these can
require different periods of record. I encourage others to load the data
directly from netcdf or grib files, remove the mean and seasonal cycle, then
place the anomaly data into the arrays.'''


OLRarray=np.zeros((25,14000,144)) # Fill with NOAA interpolated OLR anomaly data
htarray=np.zeros((25,14000,144))  # 200 hPa geopotential height anomaly data.
uarray=np.zeros((25,14000,144))   # 200 hPa zonal wind anomaly data.
varray=np.zeros((25,14000,144))   # 200 hPa meridional wind anomaly data.

hcutter=(date(1974,6,1)-date(1974,1,1)).days #Number of days to trim from the 
                                             #reanalysis data that happened 
					     #to begin Jan 1 1974. 

'''The following lines load the data from the author's arrays on disk. 
Replace them with your own data resources.'''

Llist=['Lm30','Lm275','Lm250','Lm225','Lm20','Lm175','Lm15','Lm125','Lm10','Lm75','Lm5','Lm25','L0','L25','L5','L75','L10','L125','L15','L175','L20','L225','L250','L275','L30']
for y in np.arange(25):
    file='/roundylab_rit/roundy/aroniadata/olr/Ldata/%s.mat'%Llist[y]
    filed=io.loadmat(file)
    A=filed['A'][:14000,:]
    OLRarray[y,:A.shape[0],:]=A
    file='/roundylab_rit/roundy/aroniadata/height/data/h200/Ldata/%s.mat'%Llist[y]
    filed=io.loadmat(file)
    A=filed['A'][hcutter:,:][:14000,:]
    htarray[y,:A.shape[0],:]=signal.detrend(A,axis=0)

    file='/roundylab_rit/roundy/aroniadata/winds/uwnd/h200/Ldata/%s.mat'%Llist[y]
    filed=io.loadmat(file)
    A=filed['A'][hcutter:,:][:14000,:]
    uarray[y,:A.shape[0],:]=signal.detrend(A,axis=0)

    file='/roundylab_rit/roundy/aroniadata/winds/vwnd/h200/Ldata/%s.mat'%Llist[y]
    filed=io.loadmat(file)
    A=filed['A'][hcutter:,:][:14000,:]
    varray[y,:A.shape[0],:]=signal.detrend(A,axis=0)

def regress(x,y):
     print x.shape
     print y.shape
     if x.ndim==1:
          x=np.expand_dims(x,1)
     C=np.linalg.inv(x.T.dot(x)).dot((x.T.dot(y)).transpose(1,0,2))*-2*np.std(x.squeeze())
     return C.squeeze()


def sigtest(x,y):
    '''Uses stats.pearsonr to find statistical significance of the correlation
    between the predictor and predictand.'''
    sigmap=np.zeros((y.shape[0],y.shape[2]))
    for i in np.arange(y.shape[0]):
         for j in np.arange(y.shape[2]):
	      yy=y[i,:,j]
              dummy,sigmap[i,j]=stats.pearsonr(x,yy) 

    return sigmap

base=int(80/2.5)  #Set Base longitude to 80 degrees east

#Perform regression analysis and statistical significance test.
fast.compolr=regress(intermediate.filtered[:,base],OLRarray)
intermediate.compolr=regress(intermediate.filtered[:,base],OLRarray)
intermediate.compht=regress(intermediate.filtered[:,base],htarray)
intermediate.compu=regress(intermediate.filtered[:,base],uarray)
intermediate.compv=regress(intermediate.filtered[:,base],varray)
intermediate.compusig=sigtest(intermediate.filtered[:,base],uarray)
intermediate.compvsig=sigtest(intermediate.filtered[:,base],varray)
intermediate.compu[np.logical_and(intermediate.compusig>0.05,intermediate.compvsig>0.05)]=np.nan
intermediate.compv[np.logical_and(intermediate.compusig>0.05,intermediate.compvsig>0.05)]=np.nan

slow.compolr=regress(slow.filtered[:,base],OLRarray)
slow.compht=regress(slow.filtered[:,base],htarray)
slow.compu=regress(slow.filtered[:,base],uarray)
slow.compv=regress(slow.filtered[:,base],varray)
slow.compusig=sigtest(slow.filtered[:,base],uarray)
slow.compvsig=sigtest(slow.filtered[:,base],varray)
slow.compu[np.logical_and(slow.compusig>0.05,slow.compvsig>0.05)]=np.nan
slow.compv[np.logical_and(slow.compusig>0.05,slow.compvsig>0.05)]=np.nan

fast.compolr=regress(fast.filtered[:,base],OLRarray)
fast.compht=regress(fast.filtered[:,base],htarray)
fast.compu=regress(fast.filtered[:,base],uarray)
fast.compv=regress(fast.filtered[:,base],varray)
fast.compusig=sigtest(fast.filtered[:,base],uarray)
fast.compvsig=sigtest(fast.filtered[:,base],varray)
fast.compu[np.logical_and(fast.compusig>0.05,fast.compvsig>0.05)]=np.nan
fast.compv[np.logical_and(fast.compusig>0.05,fast.compvsig>0.05)]=np.nan


#grids for mapping
xgrid=np.arange(0,360,2.5)
ygrid=np.arange(-30,32.5,2.5)
for y in np.arange(25):
    if y==0:
       Xgrid=xgrid.reshape(1,144)
    else:
       Xgrid=np.vstack((Xgrid,xgrid.reshape(1,144)))
for x in np.arange(144):
    if x==0:
       Ygrid=ygrid.reshape(25,1)
    else:
       Ygrid=np.hstack((Ygrid,ygrid.reshape(25,1)))

Xgrid,Ygrid=np.meshgrid(xgrid,ygrid)



plt.figure(figsize=(7,9))
plt.subplot(3,1,1)
map=Basemap(projection='cyl',lon_0=0,llcrnrlat=-30,urcrnrlat=30,llcrnrlon=0,urcrnrlon=180,resolution='l')
Xgrid,Ygrid=map(Xgrid,Ygrid)
map.drawcoastlines(linewidth=1.)
map.drawmeridians(np.arange(0,360,20),labels=[False,False,False,False])
map.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
map.contourf(Xgrid,Ygrid,fast.compolr,np.arange(-40,45,5),vmin=-20,vmax=20,cmap=cmap)
plt.colorbar(shrink=0.75)
map.contour(Xgrid,Ygrid,fast.compht,np.arange(1,6,1),colors='r',linewidth=2)
map.contour(Xgrid,Ygrid,fast.compht,np.arange(-5,0,1),colors='b',linewidth=2)

map.quiver(Xgrid[::2,::2],Ygrid[::2,::2],fast.compu[::2,::2],fast.compv[::2,::2])


plt.title('a. Regressed 200 hPa Wave Pattern, '+str(fast.minspeed)+' - '+str(fast.maxspeed)+' m/s')
plt.subplot(3,1,2)

map.drawcoastlines(linewidth=1.)
map.drawmeridians(np.arange(0,360,20),labels=[False,False,False,False])
map.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
map.contourf(Xgrid,Ygrid,intermediate.compolr,np.arange(-40,45,5),vmin=-20,vmax=20,cmap=cmap)
plt.colorbar(shrink=0.75)
map.contour(Xgrid,Ygrid,intermediate.compht,np.arange(1,6,1),colors='r',linewidth=2)
map.contour(Xgrid,Ygrid,intermediate.compht,np.arange(-5,0,1),colors='b',linewidth=2)

map.quiver(Xgrid[::2,::2],Ygrid[::2,::2],intermediate.compu[::2,::2],intermediate.compv[::2,::2])
plt.title('b. Regressed 200 hPa Wave Pattern, '+str(intermediate.minspeed)+' - '+str(intermediate.maxspeed)+' m/s')

plt.subplot(3,1,3)

map.drawcoastlines(linewidth=1.)
map.drawmeridians(np.arange(0,360,20),labels=[False,False,False,True])
map.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
map.contourf(Xgrid,Ygrid,slow.compolr,np.arange(-40,45,5),vmin=-20,vmax=20,cmap=cmap)
plt.colorbar(shrink=0.75)
map.contour(Xgrid,Ygrid,slow.compht,np.arange(1,6,1),colors='r',linewidth=2)
map.contour(Xgrid,Ygrid,slow.compht,np.arange(-5,0,1),colors='b',linewidth=2)

map.quiver(Xgrid[::2,::2],Ygrid[::2,::2],slow.compu[::2,::2],slow.compv[::2,::2])
plt.title('c. Regressed 200 hPa Wave Pattern, '+str(slow.minspeed)+' - '+str(slow.maxspeed)+' m/s')
plt.savefig('/pr11/roundy/public_html/exam10.png')
