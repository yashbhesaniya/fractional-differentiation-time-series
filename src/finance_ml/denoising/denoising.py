# Import required packages
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KernelDensity

from scipy.optimize import minimize

class Denoising(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 alpha: float = 0.5,
                 pts: int = 1000,
                 nFacts: int = 100,
                 q: float = 10.,
                 method: str = 'constant_residuals',
                 bWidth: float = 0.01,
                 ):
        
        self.__alpha = alpha
        self.__pts = pts
        self.__nFacts = nFacts
        self.__q = q
        self.__method = method
        self.__bWidth = bWidth
            
    def mpPDF(self,
              var: float  ) -> pd.Series(dtype = float):
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Creates a Marchenko-Pastur Probability Density Function
        Args:
            var (float): Variance
            q (float): T/N where T is the number of rows and N the number of columns
            pts (int): Number of points used to construct the PDF
        Returns:
            pd.Series: Marchenko-Pastur PDF
        """
        # Marchenko-Pastur pdf
        # q=T/N
        # Adjusting code to work with 1 dimension arrays
        if isinstance(var, np.ndarray):
            if var.shape == (1,):
                var = var[0]
        eMin, eMax = var * (1 - (1. / self.__q) ** .5) ** 2, var * (1 + (1. / self.__q) ** .5) ** 2
        eVal = np.linspace(eMin, eMax, self.__pts)
        pdf = self.__q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** .5
        pdf = pd.Series(pdf, index=eVal)
        return pdf
    
    def fitKDE(self, 
               obs: np.array, 
               kernel: str = 'gaussian', 
               x: np.array = None             ) -> pd.Series(dtype = float):
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Fit kernel to a series of obs, and derive the prob of obs x is the array of values
            on which the fit KDE will be evaluated. It is the empirical PDF
        Args:
            obs (np.ndarray): observations to fit. Commonly is the diagonal of Eigenvalues
            bWidth (float): The bandwidth of the kernel. Default is .25
            kernel (str): The kernel to use. Valid kernels are [‘gaussian’|’tophat’|
                ’epanechnikov’|’exponential’|’linear’|’cosine’] Default is ‘gaussian’.
            x (np.ndarray): x is the array of values on which the fit KDE will be evaluated
    
        Returns:
            pd.Series: Empirical PDF
        """
        if len(obs.shape) == 1:
            obs = obs.reshape(-1, 1)
        kde = KernelDensity(kernel=kernel, bandwidth=self.__bWidth).fit(obs)
        if x is None:
            x = np.unique(obs).reshape(-1, 1)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        logProb = kde.score_samples(x)  # log(density)
        pdf = pd.Series(np.exp(logProb), index=x.flatten())
        return pdf

    @staticmethod
    def getPCA( matrix: np.array):
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Utility method that gets the Eigenvalues and Eigenvector values from 
            a Hermitian Matrix
        Args:
            matrix pd.DataFrame: Correlation matrix
    
        Returns:
             (tuple): tuple containing:
                np.ndarray: Eigenvalues of correlation matrix
                np.ndarray: Eigenvectors of correlation matrix
        """
        # Get eVal,eVec from a Hermitian matrix
        eVal, eVec = np.linalg.eigh(matrix)
        indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
        eVal, eVec = eVal[indices], eVec[:, indices]
        eVal = np.diagflat(eVal)
        return eVal, eVec

    @staticmethod
    def cov2corr(cov: np.ndarray) -> np.ndarray([]):
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Utility method to derive the correlation matrix from a covariance 
            matrix
        Args:
            cov (np.ndarray): covariance matrix
    
        Returns:
            np.ndarray: correlation matrix
        """
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
        return corr

    @staticmethod
    def corr2cov(  corr: np.ndarray,
                   std     ) -> np.ndarray([]): 
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Utility method to derive the covariance matrix from a correlation 
            matrix
        Args:
            corr (np.ndarray): correlation matrix
    
        Returns:
            np.ndarray: covariance matrix
        """
        cov=corr*np.outer(std,std) 
        return cov

    def errPDFs(self, 
                var: float, 
                eVal: np.ndarray  ) -> float:
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Fit error of Empirical PDF (uses Marchenko-Pastur PDF)
        Args:
            var (float): Variance
            eVal (np.ndarray): Eigenvalues to fit.
            q (float): T/N where T is the number of rows and N the number of columns
            bWidth (float): The bandwidth of the kernel.
            pts (int): Number of points used to construct the PDF
    
        Returns:
            float: sum squared error
        """
        # Fit error
        pdf0 = self.mpPDF(var)  # theoretical pdf
        pdf1 = self.fitKDE(eVal, x=pdf0.index.values)  # empirical pdf
        sse = np.sum((pdf1 - pdf0) ** 2)
        return sse
    
    def findMaxEval(self, 
                    eVal: np.ndarray  ) -> (float, float):
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Find max random eVal by fitting Marchenko’s dist (i.e) everything else larger than
            this, is a signal eigenvalue
        Args:
            eVal (np.ndarray): Eigenvalues to fit on errPDFs
            q (float): T/N where T is the number of rows and N the number of columns
            bWidth (float): The bandwidth of the kernel.
    
        Returns:
            (tuple): tuple containing:
                float: Maximum random eigenvalue
                float: Variance attributed to noise (1-result) is one way to measure
                    signal-to-noise
        """
    
        out = minimize(lambda *x: self.errPDFs(*x), .5, args=(eVal),   
                       bounds=((1E-5, 1 - 1E-5),))
        if out['success']:
            var = out['x'][0]
        else:
            var = 1
        eMax = var * (1 + (1. / self.__q) ** .5) ** 2
        return eMax, var
    
    def denoisedCorr(self, 
                     eVal: np.ndarray,
                     eVec: np.ndarray,
                     nFacts: int ) -> np.ndarray([]):   
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Remove noise from corr by fixing random eigenvalues by constant residual 
            eigenvalue method
        Args:
            eVal (np.ndarray): Eigenvalues of Correlation Matrix
            eVec (np.ndarray): Eigenectors of Correlation Matrix
            nFacts (integer): The bandwidth of the kernel.
    
        Returns:
            corr (np.ndarray): denoised correlation matrix
        """
        
        eVal_ = np.diag(eVal).copy() 
        eVal_[nFacts:] = eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)  
        eVal_ = np.diag(eVal_)
        corr1 = np.dot(eVec,eVal_).dot(eVec.T) 
        corr1 = self.cov2corr(corr1)
        return corr1
 
    def denoisedCorr2(self, 
                       eVal: np.ndarray,
                       eVec: np.ndarray,
                       nFacts: int ) -> np.ndarray([]):  
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Remove noise from corr by fixing random eigenvalues by shrinkage method
        Args:
            eVal (np.ndarray): Eigenvalues of Correlation Matrix
            eVec (np.ndarray): Eigenectors of Correlation Matrix
            nFacts (integer): The bandwidth of the kernel.
    
        Returns:
            corr (np.ndarray): denoised correlation matrix
        """
        # Remove noise from corr through targeted shrinkage
        eValL,eVecL = eVal[:nFacts,:nFacts],eVec[:,:nFacts]    
        eValR,eVecR = eVal[nFacts:,nFacts:],eVec[:,nFacts:]    
        corr0 = np.dot(eVecL,eValL).dot(eVecL.T) 
        corr1 = np.dot(eVecR,eValR).dot(eVecR.T) 
        corr2 = corr0+self.__alpha*corr1+(1-self.__alpha)*np.diag(np.diag(corr1)) 
        return corr2
   
    def fit(self, 
            X: pd.DataFrame(dtype=float), 
            y = None):
               
        return self
        
    def transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None                     ) -> (np.ndarray([]),
                                                     np.ndarray([]),
                                                     np.ndarray([]),
                                                     np.ndarray([])):
        """
        Transforms the dataframe containing all variables of our financial series
            calculating the covariance matrix and processing the denoise and detoning.
            
        Args:
            self: object
                All entries in function __init__.        
    
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate the covariance matrix.

        Returns:
            (tuple): tuple containing:
                Cov1 (np.ndarray): Denoised covariance matrix.
                Corr1 (np.ndarray): Denoised correlation matrix.
                eVal1 (np.ndarray): Eigenvalues of Correlation Matrix
                eVec1 (np.ndarray): Eigenvectors of Correlation Matrix

        """
 
        cov = np.cov(X,rowvar=0)
        
        # Converting Covariancce to Correlation
        corr0 = self.cov2corr(cov)
        # Getting Eigenvalues and Eigenvectors
        eVal0, eVec0 = self.getPCA(corr0)
        
        # Getting Max Eigenvalues and calculating variance attributed to noise
        eMax0, var0 = self.findMaxEval(np.diag(eVal0))
        nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
        
        #----- Denoising The Corr Matrix - Residual Eigenvalue
        if self.__method == 'constant_residuals':
            corr1 = self.denoisedCorr(eVal0, eVec0, nFacts0)
            eVal1,eVec1 = self.getPCA(corr1)
        else:
        #----- Denoising The Corr Matrix - Targeting Shrinkage
            corr1 = self.denoisedCorr2(eVal0, eVec0, nFacts0) 
            eVal1,eVec1 = self.getPCA(corr1)

        cov1 = self.corr2cov(corr1, np.diag(cov)**.5)
            
        return cov1, corr1, eVal1, eVec1
    
    def fit_transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None                     ) -> (np.ndarray([]),
                                                     np.ndarray([]),
                                                     np.ndarray([]),
                                                     np.ndarray([])):
        """
        Fit and Transforms the dataframe containing all variables of our financial series
            calculating the covariance matrix and processing the denoise and detoning.
            (This routine is intented to maintaing sklearn Pipeline compatibility)
            
        Args:
            self: object
                All entries in function __init__.        
    
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate the covariance matrix.

        Returns:
            None

        """

        self.fit(X)
        self.transform(X)
        
    
