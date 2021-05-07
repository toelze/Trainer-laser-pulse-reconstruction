def GetSASE(CentralEnergy, dE_FWHM, dt_FWHM, samples=0, onlyT=False):
    import numpy as np
    from scipy.interpolate import interp1d

    h=4.135667662 #in eV*fs
    dE=dE_FWHM/2.355 #in eV, converts to sigma
    dt=dt_FWHM/2.355 #in fs, converts to sigma
    if samples == 0:
        samples= int(600.*dt*CentralEnergy/h)
    else:
        if (samples < 400.*dt*CentralEnergy/h):
            print("Number of samples is a little small, proceeding anyway. Got", samples, "prefer more than",400.*dt*CentralEnergy/h)

    EnAxis=np.linspace(0.,20.*CentralEnergy,num=samples)
    EnInput=np.zeros(samples, dtype=np.complex64)
    #for i in range(samples):
    EnInput=np.exp(-(EnAxis-CentralEnergy)**2/2./dE**2+2*np.pi*1j*np.random.random(size=samples))
#   to speed up slow fft: resample with 32*1024 samples
    resamplingI0=interp1d(EnAxis,EnInput)
    newEaxis=np.linspace(0,140,32*1024)
    EnInput=resamplingI0(newEaxis)

    newTaxis=np.fft.fftfreq(32*1024,d=140/(32*1024))*h
    TOutput=np.exp(-newTaxis**2/2./dt**2)*np.fft.fft(EnInput)    

#     sort TOutput and newTaxis
    ind = np.argsort(newTaxis, axis=0)
    newTaxis=np.sort(newTaxis)
    TOutput=np.take_along_axis(TOutput, ind, axis=0)
    
    
#     TAxis=np.fft.fftfreq(samples,d=(20.*CentralEnergy)/samples)*h
#     En_FFT=np.exp(-TAxis**2/2./dt**2)*np.fft.fft(EnInput)
#     resamplingI=interp1d(TAxis,En_FFT)

#     newEaxis=np.linspace(0,140,32*1024)
#     newTaxis=np.fft.fftfreq(32*1024,d=140/(32*1024))*h
#     TOutput=resamplingI(newTaxis)
    if not(onlyT):
        EnOutput=np.fft.ifft(TOutput)
    if (onlyT):
        return newTaxis, TOutput
    else:
        return newEaxis, EnOutput, newTaxis, TOutput
    
def GetSASE_gpu(CentralEnergy, dE_FWHM, dt_FWHM, samples=0, onlyT=False):
    from cupy import interp
    import cupy as cp
    h=4.135667662 #in eV*fs
    dE=dE_FWHM/2.355 #in eV, converts to sigma
    dt=dt_FWHM/2.355 #in fs, converts to sigma
    if samples == 0:
        samples=int(600.*dt*CentralEnergy/h)
    else:
        if (samples < 400.*dt*CentralEnergy/h):
            print("Number of samples is a little small, proceeding anyway. Got", samples, "prefer more than",400.*dt*CentralEnergy/h)

    EnAxis=cp.linspace(0.,20.*CentralEnergy,num=samples, dtype=cp.float32)
    newEaxis=cp.linspace(0,140,32*1024)
#     EnInput=cp.zeros(samples, dtype=cp.complex64)
#     for i in range(samples):
    EnInput=cp.exp(-(EnAxis-CentralEnergy)**2/2./dE**2+2*cp.pi*1j*cp.random.random(size=samples),dtype=cp.complex64)
    EnInput=interp(newEaxis,EnAxis,EnInput)

    newTaxis=cp.fft.fftfreq(32*1024,d=140/(32*1024))*h
    TOutput=cp.exp(-newTaxis**2/2./dt**2)*cp.fft.fft(EnInput)       
 
#     sort TOutput and newTaxis
    ind = cp.argsort(newTaxis, axis=0)
    newTaxis=cp.sort(newTaxis)
    TOutput=cp.take_along_axis(TOutput, ind, axis=0)
    
#     En_FFT=cp.fft.fft(EnInput)
#     TAxis=cp.fft.fftfreq(samples,d=(20.*CentralEnergy)/samples)*h
#     TOutput=cp.exp(-TAxis**2/2./dt**2)*En_FFT
    if not(onlyT):
        EnOutput=cp.fft.ifft(TOutput)
    if (onlyT):
        return newTaxis, TOutput
    else:
        return newEaxis, EnOutput, newTaxis, TOutput