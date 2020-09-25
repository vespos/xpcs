import numpy as np
import lmfit
from scipy.special import gamma, factorial, gammaln
from lmfit import Model


def get_pk(imgs_roi, adu_1ph, nphot=8, sizeBin=1):
    """ Get the photon probabilities in a stack of imgs 
    
    Args:
        imgs_roi: stack of input images
        adu_1ph: one-photon value
        nphot: max number of photon to consider
        sizeBin: use for masked detector
    Returns:
        pk: probability array (nshots, nphot)
        kbar: k average (nshots)
        photonMaps:
        Nroi: roi size (important for MLE)
    """
    if len(imgs_roi.shape)==2:
        imgs_roi = imgs_roi[None, ...]
    
    photonMaps = np.int32(np.round(imgs_roi/adu_1ph))
    pk = np.asarray(
        [np.bincount(photonMap.ravel(), minlength=nphot)[:nphot] for photonMap in photonMaps]
        ).astype(float)
    Nroi = imgs_roi[0].size
    pk = pk/Nroi*sizeBin
    kbar = np.mean(np.mean(photonMaps, axis=1), axis=1)*sizeBin
    return pk, kbar, photonMaps, Nroi

def Pk(k,kavg,M):
    """ 
    Photon statistics according to the negative binomial distribution
        k: number of photon
        kave: average number of photon per speckle
        M: number of modes (C=1/sqrt(M))
    Returns the probability of having k photon in a pixel given kave averagae photon and M modes
    """
    y1 = gamma(k+M)/gamma(M)/factorial(k)
    y2 = (kavg/(kavg+M))**k
    y3 = (M/(kavg+M))**M
    return y1*y2*y3

def NB_dist(k,kavg,M):
    """ From Yanwen """
    temp1 = gammaln(k+M)-gammaln(k+1)-gammaln(M)
    temp2 = -k*np.log(1 + M/kavg)
    temp3 = -M*np.log(1 + kavg/M)
    return np.exp(temp1+temp2+temp3)

def fit_Pk(kavg, prob, k, M=2, weights=None, func='Pk'):
    if func=='Pk':
        Pk_model = Model(Pk, independent_vars=['kavg'])
    elif func=='NB':
        Pk_model = Model(NB_dist, independent_vars=['kavg'])
    params = Pk_model.make_params()
    params['k'].set(value=k, vary=False)
    params['M'].set(value=M)
    return Pk_model.fit(prob, params, kavg=kavg, weights=weights, nan_policy='omit')

def chi_MLE(kavg, prob, Ms, nRoi):
    kmax = prob.shape[1]
    prob = prob.transpose()
    N = np.size(kavg)
    k = np.reshape(np.arange(kmax),(kmax,1))
    k = np.tile(k,(1,N))
    
    chi_sq = np.asarray([-2*np.nansum( prob*nRoi*np.log(Pk(k,kavg,M)/prob) ) for M in Ms])
    return chi_sq


class SpeckleStatistics(object):
    def __init__(self, kavg, pk, nRoi=1, **kwargs):
        self.kavg = np.asarray(kavg)
        self.pk = np.asarray(pk)
        self.ks = self.pk.shape[1]
        assert (self.kavg.shape[0] == self.pk.shape[0]), "Sizes of kave and pk do not match"
        if nRoi==1:
            print('Nroi not given, the uncertainty estimate of the MLE will be wrong.')
        self.nRoi = nRoi
        
        if 'kavgRange' in kwargs:
            # kavgRange: [min, max, binNb]
            self._kavgMin = kwargs['kavgRange'][0]
            self._kavgMax = kwargs['kavgRange'][1]
            self._kavgBinNb = kwargs['kavgRange'][2]
        else:
            self._kavgMin = 0.01
            self._kavgMax = 0.3
            self._kavgBinNb = 10
        self._kavgBinType = 'log' # chose between linspace or logspace for the bins
        return
    
    
    def fit_pk(self, k, ax=None, bin_kavg=True):
        """ Fit the negative binomial distribution to the Pk(kave) curves.
        """
        kavg = self.kavg
        kavgfilt = (kavg>=self._kavgMin)&(kavg<=self._kavgMax)
        kavg = kavg[kavgfilt]
        pk = self.pk[kavgfilt,k]
        
        if bin_kavg:
            if self._kavgBinType=='lin':
                binedges = np.linspace(self._kavgMin, self._kavgMax, self._kavgBinNb+1)
            elif self._kavgBinType=='log':
                binedges = np.logspace(np.log10(self._kavgMin), np.log10(self._kavgMax), self._kavgBinNb+1)
            # np.digitize seems to also include values below/above the first/last bin
            # filtering on kavg is thus important
            counts, edges = np.histogram(kavg, bins=binedges)

            inds = np.digitize(kavg, edges)
            n = counts.size
            kavg_binned = np.zeros(n)
            kavg_err = np.zeros(n)
            pk_binned = np.zeros(n)
            pk_err = np.zeros(n)
            nphots = np.zeros(n)

            for ii, count in enumerate(counts):
                filt = (inds==ii+1)
    #             if np.sum(filt)==0:
    #                 continue
                kavg_binned[ii] = np.mean(kavg[filt])
                kavg_err[ii] = np.std(kavg[filt])/np.sqrt(count)
                pk_binned[ii] = np.mean(pk[filt])
                pk_err[ii] = np.std(pk[filt])/np.sqrt(count)

            fitRes = fit_Pk(kavg_binned, pk_binned, k, weights=(counts*kavg_binned**2), func='Pk')
        else:
            fitRes = fit_Pk(kavg, pk, k, func='Pk')
        
        M0 = fitRes.params['M'].value
        M0_err = fitRes.params['M'].stderr
        beta = 1/M0
        beta_err = M0_err/M0**2
        
        if ax is not None:
            kavg_fit = np.linspace(self._kavgMin, self._kavgMax, 50)
            yfit = Pk(k, kavg_fit, M0)
            ymin = Pk(k, kavg_fit, 100)
            ymax = Pk(k, kavg_fit, 1)
            if bin_kavg:
                ax.errorbar(kavg_binned, pk_binned, xerr=kavg_err, yerr=pk_err, color='orange', fmt='o')
            else:
                ax.plot(kavg, pk, '.', color='purple', markersize=1)
            ax.plot(kavg_fit, yfit, color='orange', label=r'$\beta$ = {:.3f} $\pm$ {:.3f}'.format(beta,beta_err))
            ax.plot(kavg_fit, ymax, '-.', color='gray')
            ax.plot(kavg_fit, ymin, ':', color='gray')
            ax.set_xlabel('<k> (ph/px)')
            ax.set_ylabel('P(k)')
            ax.legend()
            if self._kavgBinType=='log':
                ax.set_xscale('log')
                ax.set_yscale('log')
        return beta, beta_err, fitRes

    
    def MLE_contrast(self, M=np.arange(1,20,0.1), ax=None):
        """ See Towards ultrafast dynamics with split-pulse X-ray Photon
        Correlation Spectroscopy at Free Electron Laser Sources -
        Supplementary Information - Roseker et al.
        """
        kavg = self.kavg
        kavgfilt = (kavg>=self._kavgMin)&(kavg<=self._kavgMax)
#         kavgfilt = np.ones_like(kavg).astype(bool)
        kavg = kavg[kavgfilt]
        pk = self.pk[kavgfilt,:]
        
        chi_sq = chi_MLE(kavg, pk, M, self.nRoi)
        
        M_MLE = M[np.argmin(chi_sq)]
        dM = M[1]-M[0]
        M_MLE_err = 1./(np.diff(chi_sq,n=2)/dM)[np.argmin(chi_sq)]
        beta = 1./M_MLE
        beta_err = M_MLE_err/M_MLE**2
        
        if ax is not None:
            ax.axvline(M_MLE, ls=':', label=r'$\beta$ = {:.3f} $\pm$ {:.3f}'.format(beta,beta_err))
            ax.plot(M,chi_sq, color='orange')
            ax.set_xlabel('M')
            ax.set_ylabel(r'$\chi^2$')
            ax.legend()
        return beta, beta_err, M, chi_sq