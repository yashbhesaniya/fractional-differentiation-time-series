# Import required packages
import pandas as pd
import numpy as np

#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KernelDensity

from scipy.optimize import minimize

#class Denoising(BaseEstimator, TransformerMixin):
class Denoising():
    
    def __init__(self,
                 alpha: float = 0.5,
                 pts: int = 1000,
                 nFacts: int = 100,
                 q: float = 10.,
                 method: str = 'constant_residuals',
                 bWidth: float = 0.01,
                 ):
        if (type(alpha) != float) | (alpha <= 0.0) | (alpha >= 1.0):
            raise ValueError('Denoising Class - Parameter alpha must be float, between 0 and 1')

        if (type(bWidth) != float) | (bWidth <= 0.0):
            raise ValueError('Denoising Class - Parameter bWidth must be float, positive')
        
        if (type(method) != str) | (method not in set(['constant_residuals', 'shrinkage'])):
            raise ValueError('Denoising Class - Parameter method must be either constant_residuals or shrinkage')

        if (type(nFacts) != int) | (nFacts <= 0):
            raise ValueError('Denoising Class - Parameter nFacts must be positive')
            
        if  (type(pts) != int) | (pts <= 1):
            raise ValueError('Denoising Class - Parameter pts must be greater than 1')
            
        if (type(q) != float) | (q <= 1.0):
            raise ValueError('Denoising Class - Parameter q=T/N must be float, greater than 1.0')
        
        self.__alpha = alpha    # alpha (float): Regulates the amount of shrinkage among the eigenvectors
                                # and eigenvalues associated with noise (default=.5)
        self.__pts = pts        # pts (int): No. of points used to construct the PDF (default=1000)
        self.__nFacts = nFacts  # nFacts (int): No. of significant factors (default=100)
        self.__q = q            # q (float): T/N where T is the no. of rows and N the no. of columns (default=10)
        self.__method = method  # method (str): Method for denoising ['constant_residuals', 'shrinkage']
                                #   (default='constant_residuals')
        self.__bWidth = bWidth  # bWidth (float): The bandwidth of the kernel (default=.01)
            
    def calc_PDF(self,
              var: float  ) -> pd.Series(dtype = float):
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Calculates a Marchenko-Pastur Probability Density Function
        Args:
            var (float): Variance
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
    
    def fit_KDE(self, 
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
    def calc_PCA( matrix: np.array) -> (np.ndarray([]),
                                      np.ndarray([])):
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Utility method that calculates the Eigenvalues and Eigenvector values from 
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
    def cov_to_corr(cov: np.ndarray) -> np.ndarray([]):
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
    def corr_to_cov(  corr: np.ndarray,
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

    @property
    def get_corr_original (self) -> np.ndarray([]): 
        """
        Args:
            self (object)
    
        Returns:
            np.ndarray: original correlation matrix 
        """
        return self.__corr0
    
    @property
    def get_cov_original (self) -> np.ndarray([]): 
        """
        Args:
            self (object)
    
        Returns:
            np.ndarray: original covariance matrix 
        """
        return self.__cov0
    
    @property
    def get_eval_original (self) -> np.ndarray([]): 
        """
        Args:
            self (object)
    
        Returns:
            np.ndarray: eigenvalues from the original covariance matrix 
        """
        return self.__eVal0

    @property
    def get_evec_original (self) -> np.ndarray([]): 
        """
        Args:
            self (object)
    
        Returns:
            np.ndarray: eigenvectors from the original covariance matrix 
        """
        return self.__eVec0

    @property
    def get_corr_denoised (self) -> np.ndarray([]): 
        """
        Args:
            self (object)
    
        Returns:
            np.ndarray: denoised correlation matrix 
        """
        return self.__corr1
    
    @property
    def get_cov_denoised (self) -> np.ndarray([]): 
        """
        Args:
            self (object)
    
        Returns:
            np.ndarray: denoised covariance matrix 
        """
        return self.__cov1
    
    @property
    def get_eval_denoised (self) -> np.ndarray([]): 
        """
        Args:
            self (object)
    
        Returns:
            np.ndarray: eigenvalues from the denoised covariance matrix 
        """
        return self.__eVal1

    @property
    def get_evec_denoised (self) -> np.ndarray([]): 
        """
        Args:
            self (object)
    
        Returns:
            np.ndarray: eigenvectors from the denoised covariance matrix 
        """
        return self.__eVec1

    def err_PDFs(self, 
                var: float, 
                eVal: np.ndarray  ) -> float:
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Fit error of Empirical PDF (uses Marchenko-Pastur PDF)
        Args:
            var (float): Variance
            eVal (np.ndarray): Eigenvalues to fit.
    
        Returns:
            float: sum squared error
        """
        # Fit error
        pdf0 = self.calc_PDF(var)  # theoretical pdf
        pdf1 = self.fit_KDE(eVal, x = pdf0.index.values)  # empirical pdf
        sse = np.sum((pdf1 - pdf0) ** 2)
        return sse
    
    def find_max_eval(self, 
                    eVal: np.ndarray  ) -> (float, float):
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Find max random eVal by fitting Marchenko’s dist (i.e) everything else larger than
            this, is a signal eigenvalue
        Args:
            eVal (np.ndarray): Eigenvalues to fit on errPDFs
    
        Returns:
            (tuple): tuple containing:
                float: Maximum random eigenvalue
                float: Variance attributed to noise (1-result) is one way to measure
                    signal-to-noise
        """
    
        out = minimize(lambda *x: self.err_PDFs(*x), .5, args = (eVal),   
                       bounds = ((1E-5, 1 - 1E-5),))
        if out['success']:
            var = out['x'][0]
        else:
            var = 1
        eMax = var * (1 + (1. / self.__q) ** .5) ** 2
        return eMax, var
    
    def denoised_corr(self, 
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
            nFacts (int): No. of significant factors
    
        Returns:
            corr (np.ndarray): denoised correlation matrix
        """
        
        eVal_ = np.diag(eVal).copy() 
        eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)  
        eVal_ = np.diag(eVal_)
        corr1 = np.dot(eVec, eVal_).dot(eVec.T) 
        corr1 = self.cov_to_corr(corr1)
        return corr1
 
    def denoised_corr_shrinkage(self, 
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
            nFacts (int): No. of significant factors
    
        Returns:
            corr (np.ndarray): denoised correlation matrix
        """
        # Remove noise from corr through targeted shrinkage
        eValL,eVecL = eVal[:nFacts,:nFacts],eVec[:,:nFacts]    
        eValR,eVecR = eVal[nFacts:,nFacts:],eVec[:,nFacts:]    
        corr0 = np.dot(eVecL,eValL).dot(eVecL.T) 
        corr1 = np.dot(eVecR,eValR).dot(eVecR.T) 
        corr2 = corr0 + self.__alpha * corr1 + (1 - self.__alpha) * np.diag(np.diag(corr1)) 
        return corr2
   
    def fit(self, 
            X: pd.DataFrame(dtype=float), 
            y = None):
               
        """
        Uses the dataframe containing all variables of our financial series
            to calculate its covariance matrix. The results are passed through
            'self' object and are acesses using getter methods of Denoising class.
            
        Args:
            self: object
                All entries in function __init__.        
    
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate covariance matrix.

        Returns:
            self (object)

        """
        self.__cov0 = np.cov(X, rowvar = 0)
        
        # Converting Covariancce to Correlation
        self.__corr0 = self.cov_to_corr(self.__cov0)
        # Getting Eigenvalues and Eigenvectors
        self.__eVal0, self.__eVec0 = self.calc_PCA(self.__corr0)
        
        # Getting Max Eigenvalues and calculating variance attributed to noise
        eMax0, var0 = self.find_max_eval(np.diag(self.__eVal0))
        self.__nFacts0 = self.__eVal0.shape[0] - np.diag(self.__eVal0)[::-1].searchsorted(eMax0)

        return self
        
    def transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None                     ):
        """
        Transforms covariance matrix calculated by the 'fit' process and performs
            the denoise and detoning of correlation matrix.
            
        Args:
            self: object
                All entries in function __init__.        
    
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate the covariance matrix. The results are passed through
                'self' object and are acesses using getter methods of Denoising class.

        Returns:
            self (object)

        """
        #----- Denoising The Corr Matrix - Residual Eigenvalue
        if self.__method == 'constant_residuals':
            self.__corr1 = self.denoised_corr(self.__eVal0, self.__eVec0, self.__nFacts0)
            self.__eVal1, self.__eVec1 = self.calc_PCA(self.__corr1)
        else:
        #----- Denoising The Corr Matrix - Targeting Shrinkage
            self.__corr1 = self.denoised_corr_shrinkage(self.__eVal0, self.__eVec0, self.__nFacts0) 
            self.__eVal1, self.__eVec1 = self.calc_PCA(self.__corr1)

        self.__cov1 = self.corr_to_cov(self.__corr1, np.diag(self.__cov0)**.5)
            
        return self
    
    def fit_transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None                     ):
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
            self (object)

        """

        self.fit(X)
        self.transform(X)
        
        return self
        
    
