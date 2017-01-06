import numpy as np

# utility function; return True if the argument is a positive integer
def is_pos_int( n ) : 
    import numbers
    if not isinstance( n , numbers.Integral ) : return False
    elif n <= 0 : return False
    else : return True

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class CatTree : 
    
    """ 
    A binary (decision) tree class
    
    The tree itself is implemented as a linked list construct... Each split is of the form
    
        (i) [ p , k , y , t , f ]
        i : internal "list" index
        p : parent split in the list (note: not sure if this is needed)
        k : feature this split splits over from {0,...,K-1}, or -1 if none
        y : (majority) prediction for this level
        t : "i" index in the list to move to for True features k
        f : "i" index in the list to move to for False features k
        
    The "i" indices are implicit, literally being the positional indices. The list is initialized as 
    
        [ -1 , k0 , 1 , 2 ]
        
    standing for root (no parent) and k0 in {0,1,...,K-1} being the first split (if any). 
    The list can be traversed with the predict function, which basically does 
    
        i = 0
        while T[i].t >= 0 and T[i].f >= 0 : 
            i = T[i].t if x[k] else T[i].f
        y = T[i].y
        
    
    """
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    class CatTreeNode :
        
        """ A node class to build trees from """
        
        def __init__( self , p=-1 , k=-1 , S=None , y=-1 , L=None ) :
            self._p = p # Parent node in the tree
            self._k = k # Feature to split over here
            self._S = S # Split indices for this node... i.e. goto S[x[k]]
            self._y = y # Prediction at this node (used for leafs)
            self._L = L # List of indices this node concerns (used for leafs only?)
            self._c = None # counts on items
            self._C = 0 # Total counts, or node coverage size
            self._e = None # Error on a set of data used to fit
            
        def print( self , h=0 ) : 
            s = ''
            for i in range(0,h) : s = '%s  ' % s
            
            print( '%sparent , feature , prediction , error :' % s , self._p , self._k , self._y , self._e )
            print( '%s  goto list: ' % s , self._S )
            print( '%s  indx list: ' % s , self._L )
            
        def MajPred( self , y ) : 
            
            """ set self._y as a majority prediction over y(self._L), updating error as well """
            
            import numpy as np
            
            if y is None or self._L is None : return
            # check for indexing mismatch? 
            try : y[self._L] 
            except Exception as e :
                raise ValueError( 'passed y cannot be indexed by CatTreeNode._L (%s)' % e )
            
            # search through unique items in y[L], getting counts and majority element. 
            # implementation differs for lists and for numpy.ndarrays
            
            self._c = {} # empty dictionary for counts
            self._C = 0 
            if isinstance( y , list ) : 
                u = set( y[self._L] )
                for i in u :
                    self._c[i] = y[self._L].count(i)
                    if self._c[i] > self._C : 
                        self._y = i
                        self._C = self._c[i]
            elif isinstance( y , np.ndarray ) : 
                u = np.unique( y[self._L] )
                for i in u :
                    self._c[i] = len( np.where( y[self._L] == i ) )
                    if self._c[i] > self._C : 
                        self._y = i
                        self._C = self._c[i]
            else : 
                raise ValueError( 'y is not a comprehensible object here (list, numpy.ndarray)' )
                
            # now, self._y is set as a majority predictor, unique item counts are set in self._c, 
            # and we can (re)set self._C as the total coverage
            self._C = len( self._L )
            
            # set error for this majority prediction... note using np.nditer
            self._e = sum( 1 if y[i] != self._y else 0 for i in self._L ) # np.nditer(self._L) )
            
            # return error count
            return self._e
            
        def Split( self , k ) : 
            """ Split this node (wiping out some data) on feature k """
            return
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def __init__( self , K , L=None ) :
        
        """
        CatTree init function. Requires a feature "spec" : 
        
            K: number of features
            L: a K-list of numbers of 'levels' per feature, defaults to binary if None
        
        """
        
        import numpy as np
        
        if not is_pos_int( K ) :
            raise ValueError( 'CatTree requires a positive integer number of features' )
            
        self._K = K
        self._L = 2 * np.ones((K,)) # initialize as binary
        self._T = [] # empty list initialization of the tree
        
        if L != None : 
            if len(L) != K : 
                raise ValueError( 'if feature levels are provided, you must provide for ALL features' )
            else : 
                for k in range(0,K) :
                    if not is_pos_int( L[k] ) :
                        raise ValueError( 'CatTree requires a positive integer number of feature levels when provided' )
                    elif L[k] == 1 :
                        raise ValueError( 'CatTree requires features with at least two levels (binary features)' )
                    else :
                        self._L[k] = L[k]
                    
        else : # nothing to do, as we've already done a binary feature initialization
            pass
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def split( self , n , X , y ) : 
        
        """
        CatTree "split" function used in (recursive) fit routine
        """
        
        import numpy as np
        
        # should we split the tree node n? look at error (which we hope is defined)
        e = self._T[n]._e
        
        # if error at current node is zero, don't split
        if e == 0 : return
        
        # other conditions? 
        
        # otherwise... build trial splits, and compare to current node error
        Tp = None
        Ep = len( self._T[n]._L ) # we know e must be less than this
        ip = None
        kp = -1
        for k in range(0,self._K) :
            # Is there any variation in X[T[n]._L,k]? If so, we could split over it
            u = np.unique( X[self._T[n]._L,k] ) # u holds unique elements in the kth feature
            if len( u ) > 1 : # if there is variation
                t = [] # initialize empty node list for this trial
                j = [] # initialize empty index list for this trial 
                E = 0 # initialize error for this trial, accumulated below
                for i in u : # evaluate each possible value
                    # get a LIST of indices such that X[T[n]._L,k] == i
                    l = np.where(  X[self._T[n]._L,k] == i )[0].flatten()
                    # add a tree node to the trial list for that list, with node n as parent
                    t.append( self.CatTreeNode( p=n , L=l ) )
                    j.append( i )
                    # set predictor and error and accumulate error counts
                    E += t[i].MajPred( y )
                if E < Ep : # find feature minimizing trial error
                    del Tp # delete the last version of update to tree
                    Tp = list( t ) # NOTE: need to make sure this is a "deep copy" operator
                    ip = j
                    Ep = E # reset minimum
                    kp = k # set feature index
                del t # delete the temporary list (necessary?)
                    
        # if we are splitting, append the trial nodes to the Tree (node list) self._T 
        # ... and recurse into them in turn (depth-first "search")
        
        if Ep < e : # we found a split that could lower error, and have the best one
            
            # NOTE: at some point, take this naive splitting and attempt to merge it
            # to a smaller ruleset when predictions are the same 
            
            # define the "goto" list in T[n]._S for this "branching"
            self._T[n]._S = {}
            for i in range(0,len(Tp)) : 
                self._T[n]._S[ ip[i] ] = n+i+1
            # define the feature over which we are branching
            self._T[n]._k = kp
            # append (extend) the new elements to the tree (holding n for now)
            self._T.extend( Tp[:] )
            # recursively split on each of these new elements in turn, accumulating error
            Ep = 0
            for m in self._T[n]._S.values() : 
                self.split( m , X , y )
                Ep += self._T[m]._e
            # over-write error with this new value
            self._T[n]._e = Ep 
            
        return
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
    def fit( self , N , X , y ) : 
        
        """
        CatTree "fit" function: fit a binary tree construct to feature data X and outcomes
        y. Expects an (integral, positive) number of observations N, an N x K matrix X of
        features, and an N vector y of outcomes. 
        """
        
        if not is_pos_int( N ) : 
            raise ValueError( 'CatTree requires a positive integer number of observations' )
        
        if X is None or y is None : 
            raise ValueError( 'CatTree requires a feature matrix X and a observation vector y' )
            
        try : S = X.shape 
        except AttributeError : 
            raise ValueError( 'CatTree requires feature matrices with a shape attribute' )
        else : # don't catch other exceptions (which could be... what?)
            if len(S) > 2 or len(S) == 0 : 
                raise ValueError( 'CatTree requires feature matrices; that is, dim-2 arrays' )
            if S[0] != N : 
                raise ValueError( 'CatTree expects an N x K feature matrix' )
            if len(S) == 1 and self._K > 1 : 
                raise ValueError( 'CatTree expects an N x K feature matrix' )
            if len(S) == 2 : 
                if S[1] != self._K : 
                    raise ValueError( 'CatTree expects an N x K feature matrix' )
                    
        # assert fit over categorical coded values in the data matrix
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        
        # blank initialization of tree as a list of CatTreeNode classes
        self._T = [ self.CatTreeNode( L=list(range(0,N)) ) ]
        self._T[0].MajPred( y )
        
        print( 'starting error: ' , self._T[0]._e )
        
        # set current node at the root
        n = 0
        
        # start (recursive) iteration
        self.split( 0 , X , y )
            
        # when split returns, we have built out self._T...
        print( 'fit error: ' , self._T[0]._e )
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def print( self , n=0 , h=0 ) :
        """ print method, starting from a certain index """
        self._T[n].print( h )
        if self._T[n]._S is not None : 
            H = h+1
            for i in self._T[n]._S.values() : 
                self.print( n=i , h=H )
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def predict_nocheck( self , x ) : 
        """ Method to do actual predictions via tree search, but with no argument checks """
        n = 0
        while True :
            if self._T[n]._k < 0 : return self._T[n]._y
            else : n = self._T[n]._S[ x[self._T[n]._k] ]
    
    def predict( self , X=None ) : 
        """ Method to execute predictions via tree search, but with argument checks """
        
        if X is None : return
        if len(self._T) == 0 : 
            raise ValueError( 'CatTree has not yet been fit, so cannot predict' )
        
        try : S = X.shape
        except AttributeError as e :
            raise ValueError( 'predict expects feature data X that has a shape attribute (%s)' % e )
        
        print( S )
        if len(S) > 2 or len(S) == 0 : 
            raise ValueError( 'predict expects feature data X that is N x K or K x 1' )
        if len(S) == 1 : 
            if S[0] != self._K : 
                raise ValueError( 'predict expects feature data X that is N x K or K x 1' )
            else : # use current tree to predict 
                y = self.predict_nocheck( X )
        else : # len(S) == 2
            if S[1] != self._K : 
                raise ValueError( 'predict expects feature data X that is N x K or K x 1' )
            else : # use current tree to predict each element
                if S[0] == 1 : y = self.predict_nocheck( X[0] )
                else : 
                    y = []
                    for i in range(0,S[0]) :
                        y.append( self.predict_nocheck( X[i] ) )
        
        return y
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    