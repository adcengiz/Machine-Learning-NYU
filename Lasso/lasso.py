### Coordinate Descent ###
def shooting(X, y, method='randomized', theta_init='zeros',\
             max_iter=1000, conv_threshold=1e-8, lambda_reg=0.01):
    
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((max_iter, num_features, num_features))
    loss_hist = np.zeros((max_iter, num_features))
    
    ###### Zero ###### Initialize theta,
    ##### Theta ###### with a zero vector
            
    if theta_init == 'zeros':
        theta = np.zeros(num_features) ## initialize theta
        lambda_reg = lambda_reg
        print ('lambda = ' + str(lambda_reg))
        ## you don't have to initialize shooting parameters as arrays,
        ## it is less efficient this way, but I wanted to see the value history 
        a = np.zeros(num_features)
        c = np.zeros(num_features)
        m = np.zeros(num_features)
        delta = np.zeros(num_features)
           
    ################## Initialize theta with the ridge 
    ###### Warm ###### regression solution,Not with zeros - 
    ##### Start ###### calling the ridge function inside 
        
    elif theta_init == 'warm_start':
        
        ### Using the RidgeRegression provided
        ### after doing the grid search again, the below 
        ### ridge fit might change 
        
        ridge_warmstart = RidgeRegression(l2reg=0.01) 
        ridge_warmstart.fit(X, y)
        ## initialize theta with the solution of Ridge
        theta = np.array(ridge_warmstart.w_) 
        lambda_reg = lambda_reg
        a = np.zeros(num_features)
        c = np.zeros(num_features)
        m = np.zeros(num_features)
        delta = np.zeros(num_features)
        
    ################### we pass through the coordinates 
    #### Randomized ### by randomly choosing indexes
        
    if method == 'randomized':
        i = 0 ## initialize iterations (max = max_iter)
        j = 0 ## initialize j (the randomly chosen theta vector index)
            
        while i in range(max_iter-1): 
            theta = theta ## theta update before the next epoch
            loss = compute_regularized_square_loss(X, y, theta, lambda_reg) 
            js = [] ## empty js to append js below 
            while len(js) < len(theta):  ## max number of updates 
                                         ## in one epoch = len(theta)

                a[j] = 2*(np.dot(np.transpose(X[:,j]), X[:,j]))

        ## if a == 0, the corresponding theta is automatically set 
        ## to 0 - it's not going to have an effect anyways
                if a[j] == 0:
                    theta[j] = 0
                    theta_hist[i, j] = theta
                    loss = compute_regularized_square_loss\
                    (X, y, theta, lambda_reg)
                    loss_hist[i,len(js)] = loss 
                else:
                    a[j] = a[j]
                    c[j] = 2*((np.dot(X[:,j], y)) -\
                              (np.dot(np.dot(X, theta), X[:,j])) + \
                              (np.dot(np.dot(theta[j], X[:,j]), X[:,j])))
                    m[j] = c[j]/a[j]
                    delta[j] = (lambda_reg/a[j])

                    theta[j] = soft_thresholding(m[j], delta[j]) 
                    theta_hist[i, len(js)] = theta
                    loss = compute_regularized_square_loss\
                    (X, y, theta, lambda_reg)
                    loss_hist[i,len(js)] = loss 
                    
                ## randomly select an index from theta vector                           
                j = np.random.choice(range(len(theta)))
                
                ## to control the number of selections in an epoch
                js.append(j)
                loss_hist[i,len(js)-1] = loss 
                while len(js) >= len(theta):
                    break
                
            i = i + 1    
        while i not in range(max_iter-1) \
        or (loss_hist[i-1][-1] - loss_hist[i-2][-1]) <= conv_threshold:
            print \
            ('The difference between the last losses of last two iterations =' + \
             str((loss_hist[i-1][-1] - loss_hist[i-2][-1])))
            return theta_hist, loss_hist

    ############### we pass through the coordinates one by one,
    #### Cyclic ### in the predetermined order
            
    elif method == 'cyclic':
        ### we do not randomly select the index, we just cycle through the features
        theta = theta
        i = 0
        
        while i in range(max_iter-1):
            for j in range(len(theta)):
                theta = theta + theta_hist[i, j]
                loss = compute_regularized_square_loss\
                (X, y, theta, lambda_reg)
                loss_hist[i,j] = loss_hist[i, j-1] + \
                loss_hist[i, j]
                a[j] = 2*(np.dot(np.transpose(X[:,j]), X[:,j]))
                if a[j] == 0:
                    theta[j] = 0
                    theta_hist[i, j] = theta 
                else:
                    a[j] = a[j]
                    c[j] = 2*((np.dot(X[:,j], y)) -\
                              (np.dot(np.dot(X, theta), X[:,j]))\
                              + (np.dot(np.dot(theta[j], X[:,j]), X[:,j])))
                    m[j] = c[j]/a[j]
                    delta[j] = (lambda_reg/a[j])

                    theta[j] = soft_thresholding(m[j], delta[j]) 
                    theta_hist[i, j] = theta
                        
                    loss = compute_regularized_square_loss\
                    (X, y, theta, lambda_reg)
                    loss_hist[i, j] = loss 
            i = i + 1
        while i not in range(max_iter-1) or \
        (loss_hist[i-1][-1] - loss_hist[i-2][-1]) <= conv_threshold:
            print ('The difference between the last losses \
            of last two iterations =' + str((loss_hist[i-1][-1] - \
                                             loss_hist[i-2][-1])))
            return theta_hist, loss_hist