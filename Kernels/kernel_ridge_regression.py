# Kernels and Kernel Machines

# Linear Kernel
def linear_kernel(X1, X2):
    """
    Computes the linear kernel between two sets of vectors.
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
    Returns:
        matrix of size n1xn2, with x1_i^T x2_j in position i,j
    """
    X1 = X1.reshape(X2.shape)
    
    return np.dot(X1,np.transpose(X2))

# RBF Kernel
def RBF_kernel(X1,X2,sigma):
    """
    Computes the RBF kernel between two sets of vectors   
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
        sigma - the bandwidth (i.e. standard deviation) for the RBF/Gaussian kernel
    Returns:
        matrix of size n1xn2, with exp(-||x1_i-x2_j||^2/(2 sigma^2)) in position i,j
    """

    X1 = X1.reshape(X2.shape)
    norm_matrix = cdist(X1, X2, 'sqeuclidean')
    rbf_kernel_matrix = np.exp(-norm_matrix/(2*(sigma**2)))
    
    return rbf_kernel_matrix

# Polynoial Kernel
def polynomial_kernel(X1, X2, offset, degree):
    """
    Computes the inhomogeneous polynomial kernel between two sets of vectors
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
        offset, degree - two parameters for the kernel
    Returns:
        matrix of size n1xn2, with (offset + <x1_i,x2_j>)^degree in position i,j
    """
    
    X1 = X1.reshape(X2.shape)
    offset_matrix = offset*np.ones((X1.shape[0], 
                                    X2.shape[0]))
    
    polynom_kernel_matrix = np.power((offset_matrix + linear_kernel(X1, X2)), 
                                     degree)
    
    return polynom_kernel_matrix

# Sine Kernel (1)
def sin_kernel1(X1, X2):
    '''Returns
    matrix of size n1xn2, with exp(-2sin^k(||x - x'||/2)/p)
    in position i,j'''
    
    X1 = X1.reshape(X2.shape)
    kernel_matrix = np.exp(-np.power(2*np.sin(1/2*cdist(X1, X2, 'sqeuclidean')),2)/1)
    
    return kernel_matrix

# Sine Kernel (3)
def sin_kernel3(X1, X2):
    '''Returns
    matrix of size n1xn2, with exp(-2sin^k(||x - x'||/2)/p) in 
    position i,j'''
    
    X1 = X1.reshape(X2.shape)
    kernel_matrix = np.exp(-np.power(2*np.sin(1/2*cdist(X1, X2, 'sqeuclidean')),3)/3)
    
    return kernel_matrix

# Kernel Machine
class Kernel_Machine(object):
    def __init__(self, kernel, prototype_points, weights):
        self.kernel = kernel
        self.prototype_points = prototype_points
        self.weights = weights
    def predict(self, X):
        """
        Evaluates the kernel machine on the points given by the rows of X
        Args:
            X - an nxd matrix with inputs x_1,...,x_n in the rows
        Returns:
            Vector of kernel machine evaluations on the n points in X.  
            Specifically, jth entry of return vector is
                Sum_{i=1}^R w_i k(x_j, mu_i)
        """

        weights = np.array(self.weights).reshape(-1,1)
        prototype_points = np.array(self.prototype_points).reshape(-1,1)

        preds = []
        if self.kernel == "RBF":
            k = functools.partial(RBF_kernel, \
                                  X1=prototype_points, sigma=1)
            w_kernel_matrix=weights*k(X2=X)

            for i in w_kernel_matrix.T:
                preds.append(np.sum(i))
                
        elif self.kernel == "polynomial":
            k = functools.partial(polynomial_kernel, \
                                  X1=prototype_points,offset=1,degree=3)
            w_kernel_matrix = weights*k(X2=X)
            
            for i in w_kernel_matrix.T:
                preds.append(np.sum(i))
                
        elif self.kernel == "linear":
            k = functools.partial(linear_kernel, \
                                  X1=prototype_points)
            w_kernel_matrix = weights*k(X2=X)
            
            for i in w_kernel_matrix.T:
                preds.append(np.sum(i))
                
        elif self.kernel == "sin_kernel1":
            
            k = functools.partial(sin_kernel1, \
                                  X1=prototype_points)
            w_kernel_matrix = weights*k(X2=X)
            
            for i in w_kernel_matrix.T:
                preds.append(np.sum(i))
                
        elif self.kernel == "sin_kernel3":
            
            k = functools.partial(sin_kernel3, \
                                  X1=prototype_points)
            w_kernel_matrix = weights*k(X2=X)
            
            for i in w_kernel_matrix.T:
                preds.append(np.sum(i))
            
        return preds
    
## Train
## Kernel Ridge Regression Solution
def train_kernel_ridge_regression(X, y, kernel, l2reg, 
                                  sigma=None, offset=None, degree=None):
    """Takes as input:
    - X: Data, 
    - y: Labels, 
    - kernel: The type of kernel to use,
    - l2reg: l2 regularization parameter,
    - sigma: param sigma of RBF kernel, None if kernel != RBF,
    - offset: param offset of Polynomial kernel, None if kernel != Polynomial,
    - degree: param degree of polynomial kernel, None if kernel != Polynomial.
    
    Evaluates kernel matrix for the passed data and kernel type, 
    
    Returns:
    - Kernel Machine output for the given kernel."""

    if (kernel == "RBF"):
        kernel_matrix = RBF_kernel(X, X, sigma=sigma)
    elif (kernel == "linear"):
        kernel_matrix = linear_kernel(X, X)
    elif (kernel == "polynomial"):
        kernel_matrix = polynomial_kernel(X, X, offset=offset, degree=degree)
    elif (kernel == "sin_kernel1"):
        kernel_matrix = sin_kernel1(X,X)
    elif (kernel == "sin_kernel3"):
        kernel_matrix == sin_kernel3(X,X)
    else:
        raise ValueError ("Please indicate a valid kernel name")
    alpha = np.dot(np.linalg.inv(l2reg*np.identity(len(kernel_matrix))+ kernel_matrix), y)
    
#     return Kernel_Machine(kernel, prototype_points=X.ravel(), weights=alpha)
    return Kernel_Machine(kernel, prototype_points=X, weights=alpha)

# Loss Function
def loss_mape(true, pred, ndays=7):
    """Takes as input:
    - true: the true array of "visitor" numbers 
            for the determined timeframe - shape: (n_days,)
    - pred: the predicted array of "visitor" numbers
            for the determined timeframe - shape: (n_days,)
    - ndays: timeframe in days to compute the loss.
    
    Returns:
    - avg_percent_error: average percentage error in terms
                        of visitors for the input (int) number
                        of days"""
    
    if type(ndays) != int:
        raise ValueError("The number of of days (ndays) should be an integer.")
    else:
        abs_error = np.abs(np.subtract(true[:ndays],pred[:ndays]))
        loss = np.divide(abs_error,true[:ndays])

        if len(loss) == 0:
            avg_percent_error = None
        else:
            avg_percent_error = np.mean(loss)
    
    return avg_percent_error

# Skeleton code taken from: https://davidrosenberg.github.io/mlcourse/Homework/hw4.zip

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class KernelRidgeRegression(BaseEstimator, RegressorMixin):
    """Class that implements Kernel Ridge Regression. 
    
    Takes as input: 
    - kernel: the kernel type to use,
    - sigma: param sigma of RBF kernel, None if kernel != RBF,
    - offset: param offset of Polynomial kernel, None if kernel != Polynomial,
    - degree: param degree of polynomial kernel, None if kernel != Polynomial,
    - l2reg: l2 regularization parameter.
    
    Functions:
    
    - fit:     Takes as input X (data) and y (labels), and fits the 
               kernel ridge regression to X and y. Returns the fitted 
               regressor (the KernelMachine).
    - predict: Takes as input X (data - seen or unseen), predicts the
               labels according to the KRR. Returns the pred label values.
    - score:   Takes as input X (data) and y (labels), predicts the 
               labels of the given data points, then evaluates the truth
               value of the labels against the true labels and the given 
               loss function. Returns the loss evaluation as float. 
            """
    def __init__(self, kernel="RBF", 
                 sigma=1, degree=2, offset=1, 
                 l2reg=1):
        
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.offset = offset
        self.l2reg = l2reg 
        
    def fit(self, X, y=None):
        
        if (self.kernel =="RBF"):
            self.kernel_machine_ = train_kernel_ridge_regression(X, y, "RBF", l2reg=self.l2reg,
                                                                 sigma=self.sigma,degree=None, 
                                                                 offset=None)
        elif (self.kernel =="linear"):
            self.kernel_machine_ = train_kernel_ridge_regression(X, y, "linear", l2reg=self.l2reg, 
                                                                 sigma=None,degree=None, 
                                                                 offset=None)
        elif (self.kernel =="polynomial"):
            self.kernel_machine_ = train_kernel_ridge_regression(X, y, "polynomial", l2reg=self.l2reg,
                                                                 sigma=None,degree=self.degree, 
                                                                 offset=self.offset)
        elif (self.kernel == "sin_kernel1"):
            self.kernel_machine_ = train_kernel_ridge_regression(X, y, "sin_kernel1", l2reg=self.l2reg, 
                                                                 sigma=None,degree=None, 
                                                                 offset=None)
        elif (self.kernel == "sin_kernel3"):
            self.kernel_machine_ = train_kernel_ridge_regression(X, y, "sin_kernel3", l2reg=self.l2reg, 
                                                                 sigma=None,degree=None, 
                                                                 offset=None)
        else:
            raise ValueError('Unrecognized kernel type requested.')
        return self
    
    def predict(self, X, y=None):
        
        try:
            getattr(self, "kernel_machine_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return(self.kernel_machine_.predict(X))
    
    def score(self, X, y=None):
        # get the average square error
        preds = self.predict(X)
        true = y
        return loss_mape(true, preds)