# -*- coding: utf-8 -*-
#May 23 2016, Mads Poulsen

def find_isopycnal( rho, z, dzw, sigma = 1.026, axis = 0, interp = 'none' ):
    """
    Finds the depth of a specified isopycnal
    """

    import numpy as np
    rhodev = rho - sigma #Subtract the value of the isopycnal surface from the potential density field 
    kk = np.argmin(np.abs(rhodev), axis = axis) #Find the minimum value of the absolute value rho structure and return the index.
    s = kk.shape #Find shape of k
    jj,ii = np.mgrid[0:s[0],0:s[1]] #Create indexing arrays to call corresponding values in depth matrix   
    z = np.expand_dims(np.expand_dims(z, axis = 1), axis = 2) #Expand depth array to the size of rho
    z = np.tile(z,(1,s[0],s[1])) #Repeat depth array in x and y
    rhodepth = z[kk,jj,ii] #Extract the depth of the isopycnal
    if interp == 'none':
        rhodepth[rho[0,:,:] > 1e30] = np.nan 
        return rhodepth #Output depth field of isopycnal without correct values using interpolation
    elif interp == 'linear':
        rhoprim = rhodev[kk,jj,ii] #Extract minimum deviations from perfect match
        kkup = kk + 1 #Vertical index +1
        kkdn = kk - 1 #Vertical index -1
        kkup[kkup == np.shape(rho)[0]] = np.shape(rho)[0] - 1 #If vertical index +1 is above number of layers, set it back to number of layers
        kkdn[kkdn < 0] = 0 #If vertical index -1 is below zero, set to zero
        dzw = np.expand_dims(np.expand_dims(dzw, axis = 1), axis = 2) #Expand vertical grid spacing array to the size of rho
        dzw = np.tile(dzw,(1,s[0],s[1])) #Repeat vertical space increment array in x and y
        num = rho[kkup,jj,ii] - rho[kkdn,jj,ii] #Compute the denominator of the derivative dz/drho
        num[num == 0.0] = np.inf #Set to infinte if denominator is equal or less than some user-defined threshold to avoid division by small number
        dz = -1.0 * rhoprim * (dzw[kkup,jj,ii] + dzw[kk,jj,ii]) / num #Compute numerator of derivative and divide by denominator
#        dz[dz > dzw[kk,jj,ii] / 2.0] = 0.0 #If dz is bigger than half of the cell height, the adjustment is unjustificed. This happens if the water column is constant in density with depth, because num becomes small
        dz[np.abs(dz) > dzw[kk,jj,ii]] = 0.0 #If dz is bigger than half of the cell height, the adjustment is unjustificed. This happens if the water column is constant in density with depth, because num becomes small
        rhodepth = rhodepth + dz #Add correction to zero order field
        rhodepth[rhodepth < 0] = 0 #If deviation (rhoprim) becomes so large that the correction makes the isopycnal height go less than zero, set to zero.
        rhodepth[rho[0,:,:] > 1e30] = np.nan
        return rhodepth #Output depth field of isopycnal using linear interpolation

def dft(t, dt):
    """
    VERSION1. 
    Description: A simple discrete fourier transformation. 
    Comments: N MUST be an even number. 
    
    VERSION2. 
    Description: Now written as a function and with a more pythonic for loop for the fourier coefficients. 
    Inputs: the time series 't' and the sampling interval 'dt'
    Outputs: The squares fourier coefficients 'fq' and their corresponding frequencies 'f'. 
    Comments: N MUST still be an even number
    """
    import numpy as np
    
    N = len(t) #The amount of sample points
    T = N * dt #The length of the time series in time
    sigma = 2. * np.pi / T #Fundamental frequency
    fn = 1. / (2. * dt) #Nyquist frequency
    f = np.arange(0., int(N / 2) + 1, dtype = 'float') / int(N / 2) * fn #Frequency array for plotting
    
    #Now we compute Aq and Bq components for q=1 to q=N/2-1. Bq for q=0 and q=N/2 is equal to zero.
    Aq = [2. / N * np.sum(t * np.cos(2. * np.pi / N * q * np.arange(1., N + 1., dtype = 'float'))) for q in range(1, int(N / 2))]
    Bq = [2. / N * np.sum(t * np.sin(2. * np.pi / N * q * np.arange(1., N + 1., dtype = 'float'))) for q in range(1, int(N / 2))]
    Aq = [1. / N * np.sum(t)] + Aq + [2. / N * np.sum(t * np.cos(np.arange(1., N + 1., dtype = 'float') * np.pi))] #Concatenate the Aq's with the zero and end value
    Bq = [0] + Bq +[0] #Concatenate the Bq's with the zero and end value
    Aq = np.array(Aq) #Convert to np array
    Bq = np.array(Bq) #Convert to np array

    fq = (Aq ** 2 + Bq ** 2) * N * dt #Compute spectral energy
    
    return fq, f

def sbx1(A,w):
    """
    VERSION1
    Description: A box car filter. The function receives a 1D array, and the time axis has to be the first. Also, the width of the boxcar is w = 2 * m + 1, where w is user defined.
    """
    import numpy as np
    
    m = (w - 1.) / 2. #Half width -1 of boxcar 
    nt = np.shape(A) #Determine the shape of the input data
    Asbx = np.zeros(nt) #Output array
    nt = nt[0] #Overwrite nt with length of array
    if np.mod(w,2) != 0:
        m = np.int(m)
        for n in np.arange(nt):
            if n - m < 0 or n + m + 1 > nt:
                Asbx[n] = np.nan
            else:
                Asbx[n] = np.mean(A[n - m:n + m + 1])
    elif np.mod(w,2) == 0:
        i = np.int(np.floor(m))
        for n in np.arange(nt):
            if n - m < 0 or n + m + 1 > nt:
                Asbx[n] = np.nan
            else:
                Asbx[n] = (np.sum(A[n - i:n + i + 1]) + 0.5 * A[n - i - 1] + 0.5 * A[n + i + 1]) / w

    return Asbx 

def sbx3(A,m):
    """
    VERSION1
    Description: A box car filter. The function receives a 3D array, and the time axis has to be the first. Also, the width of the boxcar is 2 * m + 1, where m is user defined.
    """
    import numpy as np
    
    nt = np.shape(A) #Determine the shape of the input data
    Asbx = np.zeros(nt)
    nt = nt[0]
    for n in np.arange(nt):
        if n - m < 0 or n + m + 1 > nt:
            Asbx[n,:,:] = np.nan
        else:
            Asbx[n,:,:] = np.mean(A[n - m:n + m + 1,:,:], axis = 0)

    return Asbx 

def regrid_pop(tlat,tlong,data,latnew,lonnew,thrs):
    """
    Regridding tool for POP output. It regrids from POP's iregular grid to a regular grid that is easier to use for plotting
    """
    import numpy as np
    from mpl_toolkits.basemap import Basemap, addcyclic
    from matplotlib.mlab import griddata 
    lat = tlat.flatten()
    lon = tlong.flatten()
    data = data.flatten()
    data[data > thrs] = np.nan

    lat = np.append(lat,lat[lon < -170]) 
    data = np.append(data,data[lon < -170])
    lonlow = lon[lon<-170]
    lont = np.append(lon,lonlow)
    lon = np.append(lon,lonlow + 360)
    

    lat = np.append(lat,lat[lont > 170])
    data = np.append(data,data[lont > 170])
    lonhi = lont[lont>170]
    lont = np.append(lont,lonhi)
    lon = np.append(lon,lonhi - 360)

    return np.array(griddata(lon,lat,data,lonnew,latnew,interp='linear'))

def so_bsf_pop(UVEL,dz,DYU,w,thrs): #Calculate BSF in Southern Ocean
    """
    Calculates the poor-man's version of the barotropic streamfunction (BSF) for the Southern Ocean. Takes partial bottom cells into account through the weight (w) term. Threshold helps to remove flagvalues
    """
    import numpy as np
    field=np.array(UVEL,copy=True)
    field[field> thrs] = 0.0 #Set fillvalues to zero
    s = np.shape(field) #Find dimensions of UVEL
    #Prepare grid variables for integration
    dz = np.expand_dims(np.expand_dims(dz, axis = 1), axis = 2)
    DYU = np.expand_dims(DYU, axis = 0)
    dz = np.tile(dz, [1,s[1],s[2]])
    DYU = np.tile(DYU, [s[0],1,1])
    
    BSF = np.sum(field * dz * DYU * w, axis = 0)
    BSF = np.cumsum(BSF, axis = 0)
    BSF[w[0,:,:]==0] = np.nan
    BSF = 1e-12 * BSF
    return BSF

def find_vertindex(field, value, axis = 0):
    import numpy as np
    fielddev = field - value #Subtract the value from the field of interest
    vertindex = np.argmin(np.abs(fielddev), axis = axis) #Find the minimum value of the absolute difference and return the vertical index.
    return vertindex #Output the vertical index field 

def hist2d(x1,x1val,x2,x2val,norm='False'):
    """
    Creates a two-dimensional histogram / probability density function
    """
    import numpy as np

    if x1val.shape[0] != x2val.shape[0]:
        raise IndexError('The two input arrays are not of equal length')

    c = 0
    hist = np.zeros([x2.shape[0],x1.shape[0]])
    for n in np.arange(x1val.shape[0]):
        if np.isnan(x1val[n])==0 and np.isnan(x2val[n])==0: 
            i = np.argmin(np.abs(x1 - x1val[n]))
            j = np.argmin(np.abs(x2 - x2val[n]))
            hist[j,i] = hist[j,i] + 1.0
            c = c + 1.0
    
    if norm == 'True':
        hist = hist / c

    return hist
    
def findcontour(cn):
    """ 
    Find longest contour of collection of contours from the plt.contour function
    """
    import numpy as np
    p = cn.collections[0].get_paths()
    contour = []
    for i in np.arange(len(p)): 
        cand = np.array(p[i].vertices)
        if np.max(np.shape(cand)) > np.max(np.shape(contour)):
            contour = cand

    return contour
#    import numpy as np
#    contours = []
#    # for each contour line
#    for cc in cn.collections:
#        paths = []
#        # for each separate section of the contour line
#        for pp in cc.get_paths():
#            xy = []
#            # for each segment of that section
#            for vv in pp.iter_segments():
#                xy.append(vv[0])
#            paths.append(np.vstack(xy))
#        contours.append(paths)
#    
#    contours = np.array(contours)
#    n = 0
#    for i in np.arange(len(contours[0,:]) - 1):
#        if len(contours[0,i + 1]) > len(contours[0,n]):
#            n = i + 1
#
#    contour = contours[0,n]
#    return contour

def nlcol(levels, cmap = 'viridis'):
    """
    Non-linear colormap generator
    """

    import matplotlib, numpy as np
    lincol = matplotlib.cm.get_cmap(cmap)
    lincol = lincol(np.linspace(0,1,len(levels)))
    y = np.arange(np.min(levels),np.max(levels) + np.min(levels[1:] - levels[0:-1]), np.min(levels[1:] - levels[0:-1]))
    nlincol = np.ones([len(y),4])
    nlincol[:,0] = np.interp(y, levels, lincol[:,0])
    nlincol[:,1] = np.interp(y, levels, lincol[:,1])
    nlincol[:,2] = np.interp(y, levels, lincol[:,2])
    nlincmap = matplotlib.colors.ListedColormap(nlincol)
    return nlincmap

def xylatlon(lon,lat,coords):
    """
    Converts a pair of lon,lat coordinates (saved in coords) to the corresponding index on the provided lon,lat grid.
    """

    import numpy as np
    s = coords.shape
    xy = np.zeros(s,dtype=np.int64)
    for n in np.arange(s[0]):
        xy[n,0] = np.nanargmin(np.abs(lon - coords[n,0]))
        xy[n,1] = np.nanargmin(np.abs(lat - coords[n,1]))         

    return xy

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    import numpy as np
    # haversine formula 
    f = np.pi / 180.0
    dlon = ( lon2 - lon1 ) * f
    dlat = ( lat2 - lat1 ) * f
    a = np.sin(dlat/2)**2 + np.cos(lat1 * f) * np.cos(lat2 * f) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371.0 # Radius of earth in kilometers.
    return c * r

def verticalmean(field,dz,w,thrs): #Calculates weighted vertical mean
    """
    Compute a weighted vertical integral
    """
    import numpy as np
    field[field > thrs] = 0.0 #Set fillvalues to zero
    s = np.shape(field) #Find dimensions of field
    #Prepare grid variables for integration
    dz = np.expand_dims(np.expand_dims(dz, axis = 1), axis = 2)
    dz = np.tile(dz, [1,s[1],s[2]])
    
    vfield = np.nansum(field * dz * w, axis = 0)
    denom = np.sum(dz * w, axis = 0)
    denom[denom==0.0]=np.inf
    vfield = vfield / denom
#    vfield[w[0,:,:]==0] = np.nan
    return vfield 
       
def ACCmask(BSF,cl,cu):
    """
    Computes a mask for the ACC based on the BSF and two circumpolar BSF contours that constrain the current.
    """
    import numpy as np, scipy.ndimage.measurements as scim
    #Create binary map of current (ones) and environment (zeros)
    field=np.array(BSF,copy=True)
    field[field>cu]=0.0
    field[field<cl]=0.0
    field[field>=cl]=1.0
    field[np.isnan(field)==1.0]=0.0
    field = np.array(field,dtype=np.int64) #Transform to integer
    [laba,labn]=scim.label(field) #Now label all individual patches that are different from zeros (all closed circulation cells outside ACC envelope.
    c = np.zeros(labn, dtype=np.int64) 
    for n in np.arange(labn):
        c[n]=np.sum(laba==n)

    c[0]=0 #Label zero is environment
    i=np.argmax(c) #Find label of ACC envelope (here assumed largest area after environment!)
    laba[laba!=i]=0 #When laba values are different from ACC envelope label, set equal to environment label. 
    #Now we need to treat all recirculation cells within ACC envelope. We start by defining ACC envelope as new environment.
    laba[laba==0]=-1 #Old environment is now abitrarily chosen to -1. 
    laba[laba==i]=0 #ACC envelope is defined as background. 
    [laba,labn]=scim.label(laba) #Find new labels
    ##We now ACC evelope is now zero. All recirculation cells WITHIN ACC envelope must be set to zero as well. First we need to identify the environment outside the ACC that we wish to leave untouched. 
    label1 = laba[0,0] #Southwestern most corner. Assumed to be outside ACC and to be a member of environment south of ACC.
    label2 = laba[-1,0] #Northwestern most corner. Assumed to be outside ACC and to be a member of environment north of ACC.
    for n in np.arange(labn): #Now set all labels, except southern and northern environments, equal to ACC envelope label.
        if n!=0 and n!=label1 and n!=label2:
            laba[laba==n]=0

    laba[laba!=0]=2 #All that is different from ACC envelope, set abitrarily to 2.
    laba[laba==0]=1 #If ACC envelope, set to 1.
    laba[laba==2]=0 #All other values to 0.

    return laba

def alongcontourmean(field,weight,wsize,xy):
    """
    This function takes a field, an associated weight field, a window size and contour indices and takes the horizontal mean along the contour with the spatial extent determined by the window size. 
    """
    import numpy as np
    s = field.shape #Shape of input field
    fieldsec = np.zeros([s[0],np.max(xy.shape[0])]) #Output file
    for n in np.arange(np.shape(xy)[0]):
        if xy[n,0] - wsize >= 0 and xy[n,0] + wsize <= s[2] - 1:
            fieldsec[:,n] = np.sum(np.sum(field[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, xy[n,0] - wsize:xy[n,0] + wsize + 1] * weight[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, xy[n,0] - wsize:xy[n,0] + wsize + 1], axis = 2), axis = 1) #Pick field below streamline. Done on non-rolled grid. 
            areasec = np.sum(np.sum(weight[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, xy[n,0] - wsize:xy[n,0] + wsize + 1], axis = 2), axis = 1)
            areasec[areasec==0.0]=np.inf
            fieldsec[:,n] = fieldsec[:,n] / areasec
        elif xy[n,0] - wsize < 0: 
            fieldsec[:,n] = np.sum(np.sum(np.concatenate([field[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, xy[n,0] - wsize:] * weight[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, xy[n,0] - wsize:],field[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, 0:xy[n,0] + wsize + 1] * weight[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, 0:xy[n,0] + wsize + 1]], axis = 2), axis = 2), axis = 1) #Pick field below streamline. Done on non-rolled grid. 
            areasec = np.sum(np.sum(np.concatenate([weight[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, xy[n,0] - wsize:],weight[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, 0:xy[n,0] + wsize + 1]], axis = 2), axis = 2), axis = 1)
            areasec[areasec==0.0]=np.inf
            fieldsec[:,n] = fieldsec[:,n] / areasec
        elif xy[n,0] + wsize > s[2] - 1:
            fieldsec[:,n] = np.sum(np.sum(np.concatenate([field[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, xy[n,0] - wsize:] * weight[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, xy[n,0] - wsize:],field[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, 0:xy[n,0] - s[2] + wsize + 1] * weight[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, 0:xy[n,0] - s[2] + wsize + 1]], axis = 2), axis = 2), axis = 1) #Pick field below streamline. Done on non-rolled grid. 
            areasec = np.sum(np.sum(np.concatenate([weight[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, xy[n,0] - wsize:],weight[:,xy[n,1] - wsize:xy[n,1] + wsize + 1, 0:xy[n,0] - s[2] + wsize + 1]], axis = 2), axis = 2), axis = 1)
            areasec[areasec==0.0]=np.inf
            fieldsec[:,n] = fieldsec[:,n] / areasec
     
    return fieldsec

def normpdf(data,nres,lb,ub):
    """
    This function takes a 1-D data array and produces the normalized probability density function of the data with resolution set by nres between lower (lb) and upper boundary (ub).
    """
    import numpy as np
    hist = np.histogram(data,np.linspace(lb,ub,nres), normed = True)
    x = 0.5 * ( hist[1][0:-1] + hist[1][1:] )
    return hist[0],x

def poisson2D( data, domain, dx, dy, cyclic=[False, True]):
    """
    Solves nabla**2 psi = f(x,y) in a rectangular domain with continents and islands. 

    INPUT:
    data: the field to which we seek the potential psi, f(x,y).
    domain: A boolean of same shape as field, which is true when f(x,y) is inside domain. 
    dx: The longitudinal width of a T-cell.
    dy: The latitudinal width of a T-cell.
    cyclic: Determines whether domain is periodic in latitude, longitude directions. True for periodic. 

    OUTPUT:
    psi: The scalar potential psi. 
    """

    import numpy as np
    import scipy.sparse as scs
    import scipy.sparse.linalg as scsalg
    import matplotlib.pyplot as plt
    
    #Determine shape
    [ny,nx] = np.shape(data)

    #Handle boundary conditions
    if np.all(cyclic):
        psidom=np.zeros([ny,nx],dtype=bool)
        psidom = domain
        d = data
        dxu = dx
        dyu = dy
    elif cyclic[1]:
        ny = ny + 2
        psidom = np.zeros([ny,nx],dtype=bool) 
        d = np.zeros([ny,nx],dtype=np.float64) 
        dxu = np.zeros([ny,nx],dtype=np.float64) 
        dyu = np.zeros([ny,nx],dtype=np.float64) 
        psidom[1:-1,:] = domain
        d[1:-1,:] = data
        dxu[1:-1,:] = dx
        dyu[1:-1,:] = dy
    elif cyclic[0]:
        nx = nx + 2
        psidom = np.zeros([ny,nx],dtype=bool)
        d = np.zeros([ny,nx],dtype=np.float64) 
        dxu = np.zeros([ny,nx],dtype=np.float64) 
        dyu = np.zeros([ny,nx],dtype=np.float64) 
        psidom[:,1:-1] = domain
        d[:,1:-1] = data
        dxu[:,1:-1] = dx
        dyu[:,1:-1] = dy
    else:
        nx = nx + 2
        ny = ny + 2
        psidom = np.zeros([ny,nx], dtype=bool)
        d = np.zeros([ny,nx],dtype=np.float64) 
        dxu = np.zeros([ny,nx],dtype=np.float64) 
        dyu = np.zeros([ny,nx],dtype=np.float64) 
        psidom[1:-1,1:-1] = domain
        d[1:-1,1:-1] = data
        dxu[1:-1,1:-1] = dx
        dyu[1:-1,1:-1] = dy
    
    #Determine where we don't want to calculate psi
    psidombnd = np.zeros(np.shape(psidom), dtype=bool)
    psidomshift = np.concatenate([psidom[:,-1:],psidom,psidom[:,0:1]], axis = 1)
    psidomshift = np.concatenate([psidomshift[-1:,:],psidomshift,psidomshift[0:1,:]], axis = 0)
    for i in np.arange(nx):
        for j in np.arange(ny):
            if np.all(psidomshift[j:j + 3,i:i + 3]):
                psidombnd[j,i]=True
        
    #Compute index for psi
    indpsi = np.reshape(np.cumsum(psidombnd), (ny,nx))         
    lenpsi = indpsi[-1,-1]
    indpsi[psidombnd==False]=0 
    
    ##Determine index for operator matrix S
    indpsi00 = indpsi[psidombnd]
    valpsi00 = ( -1.0 / ( 0.5 * ( dxu[psidombnd] + np.roll(dxu, -1, axis = 1)[psidombnd] ) ) ) * ( 1.0 / dxu[psidombnd] + 1.0 / np.roll( dxu, -1, axis = 1 )[psidombnd] ) + ( -1.0 / ( 0.5 * ( dyu[psidombnd] + np.roll(dyu, -1, axis = 0)[psidombnd] ) ) ) * ( 1.0 / dyu[psidombnd] + 1.0 / np.roll(dyu, -1, axis = 0)[psidombnd] )
    indpsi0p1 = np.roll(indpsi, -1, axis = 1)[psidombnd]
    valpsi0p1 = 1.0 / ( 0.5 * np.roll(dxu, -1, axis = 1)[psidombnd] * ( dxu[psidombnd] + np.roll(dxu, -1, axis = 1)[psidombnd] ) )
    indpsi0m1 = np.roll(indpsi, 1, axis = 1)[psidombnd]
    valpsi0m1 = 1.0 / ( 0.5 * dxu[psidombnd] * ( dxu[psidombnd] + np.roll(dxu, -1, axis = 1)[psidombnd] ) )
    indpsi1p0 = np.roll(indpsi, -1, axis = 0)[psidombnd]
    valpsi1p0 = 1.0 / ( 0.5 * np.roll(dyu, -1, axis = 0)[psidombnd] * ( dyu[psidombnd] + np.roll(dyu, -1, axis = 0)[psidombnd] ) ) 
    indpsi1m0 = np.roll(indpsi, 1, axis = 0)[psidombnd]
    valpsi1m0 = 1.0 / ( 0.5 * dyu[psidombnd] * ( dyu[psidombnd] + np.roll(dyu, -1, axis = 0)[psidombnd] ) ) 
    
    ###BUILD MATRIX###
    #Row index
    rind = np.concatenate([indpsi00,indpsi00[indpsi0p1!=0],indpsi00[indpsi0m1!=0],indpsi00[indpsi1p0!=0],indpsi00[indpsi1m0!=0]]) - 1
    
    #Column index
    cind = np.concatenate([indpsi00,indpsi0p1[indpsi0p1!=0],indpsi0m1[indpsi0m1!=0],indpsi1p0[indpsi1p0!=0],indpsi1m0[indpsi1m0!=0]]) - 1
    
    #Value associated with row/column index 
    val = np.concatenate([valpsi00,valpsi0p1[indpsi0p1!=0],valpsi0m1[indpsi0m1!=0],valpsi1p0[indpsi1p0!=0],valpsi1m0[indpsi1m0!=0]])
    
    #create sparse matrix
    S = scs.csc_matrix((val, (rind,cind)), shape = (lenpsi,lenpsi))
    
    ###SOLVE SYSTEM###
    d = d[psidombnd]
    m = scsalg.spsolve(S,d)
    resL2 = np.sqrt( np.sum((S.dot(m) - d)**2))
    resLinf = np.max(S.dot(m) - d)
    print('The L2 residual is',resL2)
    print('The Linf residual is',resLinf)
    
    #Rearange solution array to obtain potential 2D field
    psi = np.zeros(np.shape(psidom))
    psi[psidombnd]=m
    
    #Trim solution
    if np.all(cyclic):
        psi = psi 
    elif cyclic[0]:
        psi = psi[:,1:-1]
    elif cyclic[1]:
        psi = psi[1:-1,:]
    else:
        psi = psi[1:-1,1:-1]
    
    psi[domain==0]=np.nan
    return psi

def gradient3D(x,dx,ax,cyclic = 'False'):
    
    """
    Computes the horizontal gradient of f(x,y,z) on a irregular grid on the ax dimension given the grid increment dx(x,y,z). Through the cyclic option, it is possible to specify how the routine treats matrix boundaries. Land values need to be given as NaN's. 
    """
    import numpy as np

    if ax == 1 and cyclic == 'True':
        dxds = ( np.concatenate([x[:,1:,:],x[:,0:1,:]], axis = ax ) - np.concatenate([x[:,-1:,:],x[:,0:-1,:]], axis = ax ) ) / ( np.concatenate([dx[:,1:,:],dx[:,0:1,:]], axis = ax ) + dx ) #gradient
        meanval = np.fmax( np.concatenate([dxds[:,1:,:],dxds[:,0:1,:]], axis = ax ), np.concatenate([dxds[:,-1:,:],dxds[:,0:-1,:]], axis = ax ) ) #Which one of the neighbour values in ax direction is largest and non-nan? To be used at boundaries where i,j derivative is nan. Replace with neighbour value. 
        meanval[np.isnan(meanval)!=np.isnan(x)] = 0.0 #In notches bounded by nans, meanval in i,j will be nan as well. Set to zero derivative here. 
    elif ax == 1 and cyclic == 'False':
        dxds = ( np.concatenate([x[:,1:,:],x[:,-1:,:]], axis = ax ) - np.concatenate([x[:,0:1,:],x[:,0:-1,:]], axis = ax ) ) / ( np.concatenate([dx[:,1:,:],dx[:,-1:,:]], axis = ax ) + dx )
        meanval = np.fmax( np.concatenate([dxds[:,1:,:],dxds[:,-1:,:]], axis = ax ), np.concatenate([dxds[:,0:1,:],dxds[:,0:-1,:]], axis = ax ) )
        meanval[np.isnan(meanval)!=np.isnan(x)] = 0.0 #In notches bounded by nans, meanval in i,j will be nan as well. Set to zero derivative here. 
    elif ax == 2 and cyclic == 'True':
        dxds = ( np.concatenate([x[:,:,1:],x[:,:,0:1]], axis = ax ) - np.concatenate([x[:,:,-1:],x[:,:,0:-1]], axis = ax ) ) / ( np.concatenate([dx[:,:,1:],dx[:,:,0:1]], axis = ax ) + dx )
        meanval = np.fmax( np.concatenate([dxds[:,:,1:],dxds[:,:,0:1]], axis = ax ), np.concatenate([dxds[:,:,-1:],dxds[:,:,0:-1]], axis = ax ) )
        meanval[np.isnan(meanval)!=np.isnan(x)] = 0.0 #In notches bounded by nans, meanval in i,j will be nan as well. Set to zero derivative here. 
    elif ax == 2 and cyclic == 'False':
        dxdx = ( np.concatenate([x[:,:,1:],x[:,:,-1:]], axis = ax ) - np.concatenate([x[:,:,0:1],x[:,:,0:-1]], axis = ax ) ) / ( np.concatenate([dx[:,:,1:],dx[:,:,-1:]], axis = ax ) + dx )
        meanval = np.fmax( np.concatenate([dxds[:,:,1:],dxds[:,:,-1:]], axis = ax ), np.concatenate([dxds[:,:,0:1],dxds[:,:,0:-1]], axis = ax ) )
        meanval[np.isnan(meanval)!=np.isnan(x)] = 0.0 #In notches bounded by nans, meanval in i,j will be nan as well. Set to zero derivative here. 

    dxds[np.isnan(dxds)!=np.isnan(x)] = meanval[np.isnan(dxds)!=np.isnan(x)] #Replace ocean NaNs with nearest neighbour dxds value. 

    return dxds        

def gradient2D(x,dx,ax,cyclic = 'False'):
    
    """
    Computes the horizontal gradient of f(x,y) on a irregular grid on the ax dimension given the grid increment dx(x,y). Through the cyclic option, it is possible to specify how the routine treats matrix boundaries. Land values need to be given as NaN's. 
    """
    import numpy as np

    if ax == 0 and cyclic == 'True': 
        dxds = ( np.concatenate([x[1:,:],x[0:1,:]], axis = ax ) - np.concatenate([x[-1:,:],x[0:-1,:]], axis = ax ) ) / ( np.concatenate([dx[1:,:],dx[0:1,:]], axis = ax ) + dx )
        meanval = np.fmax( np.concatenate([dxds[1:,:],dxds[0:1,:]], axis = ax ), np.concatenate([dxds[-1:,:],dxds[0:-1,:]], axis = ax ) )
        meanval[np.isnan(meanval)!=np.isnan(x)] = 0.0 #In notches bounded by nans, meanval in i,j will be nan as well. Set to zero derivative here. 
    elif ax == 0 and cyclic == 'False':
        dxds = ( np.concatenate([x[1:,:],x[-1:,:]], axis = ax ) - np.concatenate([x[0:1,:],x[0:-1,:]], axis = ax ) ) / ( np.concatenate([dx[1:,:],dx[-1:,:]], axis = ax ) + dx )
        meanval = np.fmax( np.concatenate([dxds[1:,:],dxds[-1:,:]], axis = ax ), np.concatenate([dxds[0:1,:],dxds[0:-1,:]], axis = ax ) )
        meanval[np.isnan(meanval)!=np.isnan(x)] = 0.0 #In notches bounded by nans, meanval in i,j will be nan as well. Set to zero derivative here. 
    elif ax == 1 and cyclic == 'True':
        dxds = ( np.concatenate([x[:,1:],x[:,0:1]], axis = ax ) - np.concatenate([x[:,-1:],x[:,0:-1]], axis = ax ) ) / ( np.concatenate([dx[:,1:],dx[:,0:1]], axis = ax ) + dx )
        meanval = np.fmax( np.concatenate([dxds[:,1:],dxds[:,0:1]], axis = ax ), np.concatenate([dxds[:,-1:],dxds[:,0:-1]], axis = ax ) )
        meanval[np.isnan(meanval)!=np.isnan(x)] = 0.0 #In notches bounded by nans, meanval in i,j will be nan as well. Set to zero derivative here. 
    elif ax == 1 and cyclic == 'False':
        dxdx = ( np.concatenate([x[:,1:],x[:,-1:]], axis = ax ) - np.concatenate([x[:,0:1],x[:,0:-1]], axis = ax ) ) / ( np.concatenate([dx[:,1:],dx[:,-1:]], axis = ax ) + dx )
        meanval = np.fmax( np.concatenate([dxds[:,1:],dxds[:,-1:]], axis = ax ), np.concatenate([dxds[:,0:1],dxds[:,0:-1]], axis = ax ) )
        meanval[np.isnan(meanval)!=np.isnan(x)] = 0.0 #In notches bounded by nans, meanval in i,j will be nan as well. Set to zero derivative here. 

    dxds[np.isnan(dxds)!=np.isnan(x)] = meanval[np.isnan(dxds)!=np.isnan(x)] #Replace ocean NaNs with nearest neighbour dxds value. 

    return dxds        
