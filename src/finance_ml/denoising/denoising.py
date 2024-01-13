"""
Created on Mon May 8 19:00:00 2023
@Group: 
    Franci Daniele Prochnow Gaensly
    Frederico Alexandre
    Coded by Luis Alvaro Correia
    
Updated: 10 November 2023 by
@Group:
    Priyanka Teja Ravi
    Ramiscan Yakar
    Tolga Keskinoglu

"""

# Import required packages
import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity

from scipy.optimize import minimize

class Denoising():
    
    def __init__(self,
                 alpha: float = 0.5,
                 pts: int = 1000,
                 nFacts: int = 100,
                 q: float = None,
                 method: str = 'constant_residuals',
                 bWidth: float = 0.01,
                 detoning: bool=False,
                 market_component: int=1
                 ):
        """
        Initialize data.
            
        Args:
            data (pd.DataFrame): Columns of dataframe containing the variables 
                to be scaled.
            alpha (float): Regulates the amount of shrinkage among the eigenvectors 
                and eigenvalues associated with noise
            pts (int): No. of points used to construct the PDF
            nFacts (int): No. of significant factors
            q (float): T/N where T is the no. of rows and N the no. of columns
            method (str): Method for denoising ['constant_residuals', 'shrinkage']
            bWidth (float): The bandwidth of the kernel
            detoning (bool): Boolean for appyling detoning, default False

        Returns:
            None

        """
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
            
        if (q != None) and ((type(q) != float ) | (q <= 1.0)):
            raise ValueError('Denoising Class - Parameter q=T/N must be None or float, greater than 1.0')
        
        if (type(detoning) != bool) :
            raise ValueError('Denoising Class - Parameter detoning must be bool, either True or False')
        
        if  (type(market_component) != int) | (market_component < 1) :
            raise ValueError('Denoising Class - Parameter market_component must be int and greater than 1')
        
        self.__alpha = alpha    
        self.__pts = pts
        self.__nFacts = nFacts
        self.__q = q
        self.__T = None
        self.__N = None
        self.__method = method
        self.__bWidth = bWidth
        self.__detoning=detoning
        self.__market_component=market_component
            
    def calc_PDF(self, # calc Marchenko-Pastur pdf
              var: float,
              q: float = None) -> pd.Series(dtype = float):
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

        if q is None:
            if self.get_q is None:
                raise ValueError('Denoising Class - Parameter q=T/N must be provided, either manually or by providing data matrix with run_denoising method')
            else:
                q = self.get_q
               
        # Adjusting code to work with 1 dimension arrays
        if isinstance(var, np.ndarray):
            if var.shape == (1,):
                var = var[0]

        eMin, eMax = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
        eVal = np.linspace(eMin, eMax, self.__pts)
        pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** .5
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
                      std: np.ndarray     ) -> np.ndarray([]): 
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        Utility method to derive the covariance matrix from a correlation 
            matrix
        Args:
            corr (np.ndarray): correlation matrix
            std (np.ndarray): standard deviation matrix
        Returns:
            np.ndarray: covariance matrix
        """
        cov=corr*np.outer(std,std) 
        return cov

    @property
    def get_corr_original (self) -> np.ndarray([]): 
        """
        Args:
            None.
    
        Returns:
            np.ndarray: original correlation matrix 
        """
        return self.__corr0
    
    @property
    def get_cov_original (self) -> np.ndarray([]): 
        """
        Args:
            None.
    
        Returns:
            np.ndarray: original covariance matrix 
        """
        return self.__cov0
    
    @property
    def get_eval_original (self) -> np.ndarray([]): 
        """
        Args:
            None.
    
        Returns:
            np.ndarray: eigenvalues from the original covariance matrix 
        """
        return self.__eVal0

    @property
    def get_evec_original (self) -> np.ndarray([]): 
        """
        Args:
            None.
    
        Returns:
            np.ndarray: eigenvectors from the original covariance matrix 
        """
        return self.__eVec0

    @property
    def get_corr_denoised (self) -> np.ndarray([]): 
        """
        Args:
            None.
    
        Returns:
            np.ndarray: denoised correlation matrix 
        """
        return self.__corr1
    
    @property
    def get_cov_denoised (self) -> np.ndarray([]): 
        """
        Args:
            None.
    
        Returns:
            np.ndarray: denoised covariance matrix 
        """
        return self.__cov1
    
    @property
    def get_eval_denoised (self) -> np.ndarray([]): 
        """
        Args:
            None.
    
        Returns:
            np.ndarray: eigenvalues from the denoised covariance matrix 
        """
        return self.__eVal1

    @property
    def get_evec_denoised (self) -> np.ndarray([]): 
        """
        Args:
            None.
    
        Returns:
            np.ndarray: eigenvectors from the denoised covariance matrix 
        """
        return self.__eVec1
    
    @property
    def get_q (self) -> float:
        """
        Args:
            None.
    
        Returns:
            float: q parameter
        """
        return self.__q
    
    @get_q.setter
    def set_q (self, q: float):
        """
        Args:
            q (float): q parameter
    
        Returns:
            None
        """
        self.__q = q
    
    @property
    def get_T (self) -> float:
        """
        Args:
            None.
    
        Returns:
            float: T parameter
        """
        return self.__T
    
    @get_T.setter
    def set_T (self, T: float):
        """
        Args:
            T (float): T parameter
    
        Returns:
            None
        """
        self.__T = T
    
    @property
    def get_N (self) -> float:
        """
        Args:
            None.
    
        Returns:
            float: N parameter
        """
        return self.__N
    
    @get_N.setter
    def set_N (self, N: float):
        """
        Args:
            N (float): N parameter
    
        Returns:
            None
        """
        self.__N = N

    def err_PDFs(self, 
                var: float,
                eVal: np.ndarray,
                q: float = None) -> float:
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

        pdf0 = self.calc_PDF(var, q) # theoretical pdf

        pdf1 = self.fit_KDE(eVal, x = pdf0.index.values)  # empirical pdf
        sse = np.sum((pdf1 - pdf0) ** 2)
        return sse
    
    def find_max_eval(self, 
                    eVal: np.ndarray,
                      q = None  ) -> (float, float):
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
        """
        Updated on 10 November 2023
        
        Adding q as a parameter to the optimization function.
        Not present in the "Machine Learning for Asset Managers" book.
        """

        if q is None: # q not proivded during initialization
            if self.get_q is None: # q never calculated during run_denoising method
                raise ValueError('Denoising Class - Parameter q=T/N must be provided, either manually or by providing data matrix with run_denoising method')
            
            # Optimizing condition met: q not set manually and calculated during run_denoising method
            else:
                try:
                    q = self.get_q
                except TypeError:  # If T or N are not set during class initialization
                    raise ValueError("No q value available. Either set q manually or proivde data matrix with run_denoising method.")
            
                # initial guess for var and q
                x0 = [0.5, q/2]

                # bounds for var and q
                bounds = ((1E-5, 1 - 1E-5), (1.00001, q))
        
                out = minimize(lambda x: self.err_PDFs(var=x[0], eVal=eVal, q=x[1]), x0, bounds=bounds)

                if out['success']:
                    var, q = out['x'][0], out['x'][1]
                else:
                    var, q = 1, q.item()

        # q manually proivded, no optimization needed
        else:
            out = minimize(lambda *x: self.err_PDFs(*x), .5, args = (eVal),   
            bounds = ((1E-5, 1 - 1E-5),))
            
            if out['success']:
                var = out['x'][0]
            else:
                var = 1

        eMax = var * (1 + (1. / q) ** .5) ** 2
        return eMax, var, float(q)
    
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
 
    def detoned_corr(self, 
                     corr1: np.ndarray,
                     eVal: np.ndarray,
                     eVec: np.ndarray,
                     market_component: int ) -> np.ndarray([]):   
        """
        Adapted from Chap. 2 of  "Machine Learning for Asset Managers", by
        - Marcos M. Lopez de Prado - 1st. edition
        
        De-tones the de-noised correlation matrix by removing the market component.

        Args:
            corr (np.ndarray): De-noised Correlation Matrix
            eVal (np.ndarray): Eigenvalues of Correlation Matrix
            eVec (np.ndarray): Eigenectors of Correlation Matrix
            market_component (int): Number of fist eigevectors related to a market component. (1 by default)
    
        Returns:
            corr (np.ndarray):  De-toned correlation matrix.
        """
        
        if (market_component>corr1.shape[1]):
            raise ValueError(f'Parameter market_component must less than the number of features in denoised correlation matrix: {corr1.shape[1]}')
        
        eigenvalues_market = eVal[:market_component, :market_component]
        eigenvectors_market = eVec[:, :market_component]
        corr_market = np.dot(eigenvectors_market, eigenvalues_market).dot(eigenvectors_market.T)
        corr1 = corr1 - corr_market
        corr1 = self.cov_to_corr(corr1) #Rescaling the correlation matrix to have 1s on the main diagonal
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
   
    def run_denoising(self, 
                      X: pd.DataFrame(dtype=float)):
               
        """
        Uses the dataframe containing all variables of our financial series
            to calculate its covariance matrix. The results are passed through
            'self' object and are acesses using getter methods of Denoising class.
        Transforms covariance matrix calculated by the 'fit' process and performs
            the denoise and detoning of correlation matrix.
            
        Args:
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate covariance matrix.

        Returns:
            self (object)

        """
        self.set_T = X.shape[0]
        self.set_N = X.shape[1]

        self.set_q = self.get_T / self.get_N

        self.__cov0 = np.cov(X, rowvar = 0)
        
        # Converting Covariancce to Correlation
        self.__corr0 = self.cov_to_corr(self.__cov0)
        # Getting Eigenvalues and Eigenvectors
        self.__eVal0, self.__eVec0 = self.calc_PCA(self.__corr0)
        
        # Getting Max Eigenvalues and calculating variance attributed to noise
        eMax0, var0, q0 = self.find_max_eval(np.diag(self.__eVal0))
        self.__nFacts0 = self.__eVal0.shape[0] - np.diag(self.__eVal0)[::-1].searchsorted(eMax0)

        #----- Denoising The Corr Matrix - Residual Eigenvalue
        if self.__method == 'constant_residuals':
            self.__corr1 = self.denoised_corr(self.__eVal0, self.__eVec0, self.__nFacts0)
            self.__eVal1, self.__eVec1 = self.calc_PCA(self.__corr1)
            if self.__detoning==True:
                self.__corr1=self.detoned_corr(self.__corr1,self.__eVal1,self.__eVec1,self.__market_component)

        else:
        #----- Denoising The Corr Matrix - Targeting Shrinkage
            self.__corr1 = self.denoised_corr_shrinkage(self.__eVal0, self.__eVec0, self.__nFacts0) 
            self.__eVal1, self.__eVec1 = self.calc_PCA(self.__corr1)
            if self.__detoning==True:
                self.corr1=self.detoned_corr(self.__corr1,self.__eVal1,self.__eVec1,self.__market_component)

        self.__cov1 = self.corr_to_cov(self.__corr1, np.diag(self.__cov0)**.5)
            
        return self, var0, q0
        
