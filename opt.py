#連続最適化
#与えられた関数の最小値または最大値、最適解を、
#最急降下法、ニュートン法、準ニュートン法、二次補間法を用いて計算

"""
python opt.py <問題> <手法> <直線探索>のように使う。
手法は１）最急降下法、２）ニュートン法、３）準ニュートン法
直線探索は１）バックトラッキング法、２）二次補間法

ex)
python opt.py 3 1 2

"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *
from scipy import *
from scipy import linalg
import string
import sys
from matplotlib import *
from matplotlib import pyplot

def main():
    args = sys.argv[1:]
    problem = int(args[0])
    method = int(args[1])
    search_method = int(args[2])

    xi = math.pow(10,-4) #Step 0におけるバックトラック法のパラメータ（アルミホ条件で使う）
    rho = 0.5 #Step 0におけるバックトラック法のパラメータ(t = rho * tとtを小さくする時に使う）
    d = 0
##################################################################    
    if problem == 1: #kadai2,3
        """
        minimize f(x):= x0**2+ exp(x0) + x1**4 + x1**2 -2*x0*x1 + 3
        subject to x:= (x0, x1)⊤ ∈ R**2
        """

        #define functions
        def evalf(x):
            #compute a function f
            f = x[0]**2 + math.pow(math.e,x[0]) + x[1]**4 + x[1]**2 - 2*x[0]*x[1] + 3
            return f

        def evalg(x):
            #compute the gradient of f
            g = array([2*x[0] + math.pow(math.e,x[0]) - 2*x[1],4*(x[1]**3) + 2*x[1] - 2*x[0]])
            return g

        def evalh(x):
            #compute a Hessian matrix of f
            H = array([[2 + math.pow(math.e,x[0]), -2], [-2, 12*(x[1]**2) + 2]])
            return H
        
        B = eye(2)  #initial B used if method is quasi-Newton method
        X = array([0.0,0.0]) #initial points
        k = 0 #number of iterations
        epsilon = 2 * math.pow(10,-6) #終了条件のパラメータ
        
        #find the optimal solution
        while linalg.norm(evalg(X)) > epsilon:
            
            #compute a search direction
            if method == 1: #Steepest descent
                d_old = d
                d = -evalg(X)           
            elif method == 2: #Newton method
                T = 0
                m = 0
                while True:
                    try:
                        d_old = d
                        L = linalg.cho_factor(evalh(X) + T*eye(2)) #Cholesky factorization
                        d = linalg.cho_solve(L, -evalg(X)) #search direction
                        break
                    except linalg.LinAlgError:
                        if m == 0:
                            T = 2
                        else:
                            T = T*2
                        m = m + 1                   
            elif method == 3:  #quasi-Newton method
                d_old = d
                L = linalg.cho_factor(B) #Cholesky factorization
                d = linalg.cho_solve(L, -evalg(X)) #search direction
            else:
                sys.exit()
                    
            f = evalf(X) #objective function
            e = inner(d,evalg(X)) #dot product search direction and the gradient
            t = 1.0 #initial step size

            #find t(step size)
            while evalf(X+t*d) > f +t*xi*e:
                if search_method == 1: #backtracking method
                    t = rho*t
                elif search_method == 2: #quadratic interpolation method
                    t_old = t #previous t
                    t = -e*math.pow(t,2)/2*(evalf(X+t*d)- f -t*e)
                    if t < 0.1*t_old or t >  0.9*t_old:
                        t = t_old/2.0
                else:
                    sys.exit()

            X_old = X #previous x
            X = X_old + t*d
            if method == 3: #if the method is quasi-Newton method
                #recaculate B
                s = X - X_old
                y = evalg(X) - evalg(X_old)
                B = B - outer(B.dot(s),B.dot(s))/inner(s,B.dot(s)) + outer(y,y)/inner(s,y)
            k = k + 1
            print ("反復回数k = {0}, ||▽f(x^k)|| = {1:9.2e}".format(k, linalg.norm(evalg(X)) ))
        #output
        print ('反復回数', k)
        print ('最適値 = {0:9.2e}'.format(f))
        print ('求まった解 = ({0:9.2e},{1:9.2e})'.format(X[0],X[1]))
###############################################################
    elif problem == 2: #kadai4
        """
        minimize f(x) := f0(x)**2 + f1(x)**2
        subject to x := (x0, x1)⊤ ∈ R**2
        """

        #define functions
        def evalf(x):
            #compute a function f
            f = 0
            y = array([1.5,2.25,2.625])
            for i in range(3):
                f_i = y[i] - x[0]*(1 - math.pow(x[1],i+1))
                f = f + math.pow (f_i,2)
            return f

        def evalg(x):
            #compute the gradient of f
            g = 0
            y = array([1.5,2.25,2.625])
            for i in range(3):
                f_i = y[i]- x[0]*(1 - math.pow(x[1],i+1))
                gradf_i = array([-(1 - math.pow(x[1],i+1)),(i+1)*x[0]*math.pow(x[1],i) ])
                g = g + 2*f_i*gradf_i
            return g

        def evalh(x):
            #compute a Hessian matrix of f
            H = 0
            y = array([1.5,2.25,2.625])
            for i in range(3):
                f_i = y[i]- x[0]*(1 - math.pow(x[1],i+1))
                gradf_i = array([-(1 - math.pow(x[1],i+1)),(i+1)*x[0]*math.pow(x[1],i) ]) #the gradient of f(i)
                #a Hassian matrix of f(i)
                hessef_i = array([[0,(i+1)*math.pow(x[1],i)],[(i+1)*math.pow(x[1],i),i*(i+1)*x[0]*math.pow(x[1],i-1)]])
                H = H + 2*(f_i*hessef_i + outer(gradf_i,gradf_i))
            return H

        B = eye(2)  #initial B used if method is quasi-Newton method
        X = array([1.0,1.0]) #initial points
        k = 0 #number of iterations
        epsilon = 2 * math.pow(10,-6) #終了条件のパラメータ
        
        #find the optimal solution
        while linalg.norm(evalg(X)) > epsilon:
            
            #compute a search direction
            if method == 1: #Steepest descent
                d = -evalg(X)           
            elif method == 2: #Newton method
                T = 0
                m = 0
                while True:
                    try:
                        L = linalg.cho_factor(evalh(X) + T*eye(2)) #Cholesky factorization
                        d = linalg.cho_solve(L, -evalg(X)) #search direction
                        break
                    except linalg.LinAlgError:
                        if m == 0:
                            T = 2
                        else:
                            T = T*2
                        m = m + 1                   
            elif method == 3:  #quasi-Newton method
                L = linalg.cho_factor(B) #Cholesky factorization
                d = linalg.cho_solve(L, -evalg(X)) #search direction
            else:
                sys.exit()

            f = evalf(X) #objective function
            e = inner(d,evalg(X)) #dot product search direction and the gradient
            t = 1.0 #initial step size

            #find t(step size)
            while evalf(X+t*d) > f + t*xi*e:
                if search_method == 1: #backtracking method
                    t = rho*t
                elif search_method == 2: #quadratic interpolation method
                    t_old = t #previous t
                    #print ("t_old = {}".format(t))
                    t = -e*math.pow(t,2)/2*(evalf(X+t*d)-f-t*e)
                    #print ("t = {}".format(t))
                    #print ("0.1*t_old = {},  0.9*t_old = {}".format(0.1*t_old,0.9*t_old))
                    if t < 0.1*t_old or t >  0.9*t_old:
                        t = t_old/2.0
                else:
                    sys.exit()

            X_old = X #previous x
            X = X_old + t*d
            if method == 3: #if the method is quasi-Newton method
                #recaculate B
                s = X - X_old
                y = evalg(X) - evalg(X_old)
                B = B - outer(B.dot(s),B.dot(s))/inner(s,B.dot(s)) + outer(y,y)/inner(s,y)
            k = k + 1
            print ("反復回数k = {0}, ||▽f(x^k)|| = {1:9.2e}".format(k, linalg.norm(evalg(X)) ))
        #output
        print ('反復回数 = ', k)
        print ('最適値 = {0:9.2e}'.format(f))
        print ('求まった解 = ({0:9.2e},{1:9.2e})'.format(X[0],X[1]))
        
###############################################################
    elif problem == 3: #kadai5
        """
            minimize f(x) := 1/2 * <Ax,x>
            subject to x ∈ R^n
            A ∈ R^(n*n)は正定値対称行列
            要素が全て[0,1]の乱数であるZ ∈ R^(n*n)を用いてA =　Z⊤Zと定義。
        """

        #define functions
        def evalf(A,x):
            #compute a function f
            f = 0.5*dot(x,dot(A,x))
            return f

        def evalg(A,x):
            #compute the gradient of f
            g = A.dot(x)
            return g

        def evalh(A,x):
            #compute a Hessian matrix of f
            H = A
            return H

        n = array([2,5,10,20])
        result_f = zeros([4,5]) #stores the results of min f(x)
        result_x_0 = zeros([4,5]) #stores the results of x[0] 
        result_x_1 = zeros([4,5]) #stores the results of x[1]
        means_number_of_iterations = zeros(4) #stores the means of results of number of iterations
        k = 0 #used when calculating number of iterations
        
        #find the optimal solution
        for i in range(4):
            for l in range(5):
                
                Z = rand(n[i],n[i]) #Random matrix
                A = Z.T.dot(Z) #positive-definite symmetrix matrix
                X = ones(n[i]) #initial points
                B = eye(n[i]) #initial B used if the method is quasi-Newton method 
                k = 0
                epsilon = n[i] * math.pow(10,-6) #終了条件のパラメータ
                
                while linalg.norm(evalg(A,X)) > epsilon:
                    
                    #compute a search direction
                    if method == 1: #Steepest descent
                        d = -evalg(A,X)
                        
                    elif method == 2: #Newton method
                        T = 0
                        m = 0
                        while True:
                            try:
                                L = linalg.cho_factor(evalh(A,X) + T*eye(n[i])) #Cholesky factorization
                                d = linalg.cho_solve(L, -evalg(A,X)) #search direction
                                break
                            except linalg.LinAlgError:
                                if m == 0:
                                    T = 2
                                else:
                                    T = T*2
                                m = m + 1
                            
                    elif method == 3:  #quasi-Newton method
                        L = linalg.cho_factor(B) #Cholesky factorization
                        d = linalg.cho_solve(L, -evalg(A,X)) #search direction
                    else:
                        sys.exit()

                    f = evalf(A,X) #objective function
                    e = inner(d,evalg(A,X)) #dot product search direction and the gradient
                    t = 1.0 #initial step size
                    
                    #find t(step size)
                    while evalf(A,X+t*d) > f +t*xi*e:
                        if search_method == 1: #backtracking method
                            t = rho*t
                        elif search_method == 2: #quadratic interpolation method
                            t_old = t #previous t
                            t = -e*math.pow(t,2)/2*(evalf(A,X+t*d)-f-t*e)
                            if t < 0.1*t_old or t >  0.9*t_old:
                                t = t_old/2.0
                        else:
                            sys.exit()

                    X_old = X #previous x
                    X = X_old + t*d
                    if method == 3: #if the method is quasi-Newton method
                        #recaculate B
                        s = X - X_old
                        y = evalg(A,X) - evalg(A,X_old)
                        B = B - outer(B.dot(s),B.dot(s))/inner(s,B.dot(s)) + outer(y,y)/inner(s,y)
                    k = k + 1
                result_f[i,l] = f
                result_x_0[i,l] = X[0]
                result_x_1[i,l] = X[1]
                means_number_of_iterations[i] = means_number_of_iterations[i] + k/5.0

        for i in range(4):
            print ('n = {0}, 反復回数kの平均 = {1:9.2e}'.format(n[i],means_number_of_iterations[i]))

        
###############################################################
    elif problem == 4: #kadai6
        """
        あらかじめ用意された訓練データA,zを用いて、２乗ヒンジ損失を用いるソフトマージン最大化の問題を解く。
        x,w ∈ R^q, b ∈ Rを用いて、SVMの線形分類境界Φ:R^q → Rを Φ(x) = w⊤x + bと表す。
        また、２乗ヒンジ損失関数 l(y) = (max{0,1-y})**2、ηを正則化パラメータとする。
        
        minimize 1/2 * ||w||**2 + η ∑ l(ziΦ(xi))
        subject to w ∈ R^q, b ∈ R
        """

        #definite functions
        n = 1 #regularization parameter
        def evalb(y,x):
            #compute border
            w = array([y[0],y[1]])
            return inner(w,x) + y[2]
            
        def evall(x):
            #compute a 0-1 loss function
            return math.pow(max(0,1-x),2)
            
        def evalf(y,z,B):
            #compute a function f
            w = array([y[0],y[1]])
            f = 0.5*math.pow(linalg.norm(w),2)
            for i in range(len(B)):
                f = f + n*evall(z[i]*evalb(y,B[i]))
            return f

        def evalg(y,B):
            #compute the gradient of f
            g = array([y[0],y[1],0])
            for i in range(len(B)):
                g = g - 2*n*z[i]*max(0,1-z[i]*evalb(y,B[i]))*array([B[i][0],B[i][1],1])
            return g
                                     
        #Training Data and labels
        A = array([[0,-2.1],[1,3.7],[5,2],[6.2,5],[3,4.8],[3.1,-9.5],[8,2],[0.7,6],[6,-5],[4.5,10]])
        z = array([-1,-1,1,1,-1,1,1,-1,1,-1])

        #initial points
        w = array([1,1])
        b = 1
        y = array([w[0],w[1],b])
        
        B = eye(3)  #initial B used if method is quasi-Newton method                                               
        k = 0 #number of iterations
        epsilon = 3 * math.pow(10,-6) #終了条件のパラメータ
        
        #find the optimal solution
        while linalg.norm(evalg(y,A)) > epsilon:
            
            #compute a search direction
            if method == 1: #Steepest descent
                d = -evalg(y,A)                               
            elif method == 3:  #quasi-Newton method
                L = linalg.cho_factor(B) #Cholesky factorization
                d = linalg.cho_solve(L, -evalg(y,A)) #search direction
            else:
                sys.exit()
                
            f = evalf(y,z,A) #objective function
            e = inner(d,evalg(y,A)) #dot product search direction and the gradient
            t = 1.0 #initial step size
                
            #find t(step size)
            while evalf(y+t*d,z,A) > f +t*xi*e:
                if search_method == 2: #quadratic interpolation method
                    t_old = t #previous t
                    t = -e*math.pow(t,2)/2*(evalf(y+t*d,z,A)-f-t*e)
                    if t < 0.1*t_old or t >  0.9*t_old:
                        t = t_old/2.0
                else:
                    sys.exit()
            
            y_old = y #previous y
            y = y_old + t*d
            if method == 3: #if the method is quasi-Newton method
                #recalculate B
                s = y - y_old
                Y = evalg(y,A) - evalg(y_old,A)
                B = B - outer(B.dot(s),B.dot(s))/inner(s,B.dot(s)) + outer(Y,Y)/inner(s,Y)
            k = k + 1
            #print ("反復回数k = {0}, ||▽f(x^k)|| = {1:9.2e}".format(k, linalg.norm(evalg(y,A)) ))

        #output
        print ('反復回数', k)
        print ('最適値 = {0:9.2e}'.format(f))
        print ('求まった解 (w,b) = ({0:9.2e},{1:9.2e},{2:9.2e})'.format(y[0],y[1],y[2]))

        # plot the points which A contains
        if method == 1:
            pyplot.title('Steepest Descent, Quadratic Interpolation Method')
        else: #method == 3
            pyplot.title('Quasi-Newton Method, Quadratic Interpolation Method')
        pyplot.xlabel('$x_0$')
        pyplot.ylabel('$x_1$')
        for i in range(len(A)):
            if z[i] == 1:
                pyplot.plot(A[i][0],A[i][1],color='r',marker='o')
            else: # z[i] == -1
                pyplot.plot(A[i][0],A[i][1],color='b',marker='o')

        #plot the graph
        if y[1] != 0:
            X = linspace(-15,15,100)
            Y = - (y[0]/y[1])*X - y[2]/y[1]
            pyplot.plot(X,Y)
        else:
            pyplot.axvline(x = -y[2]/y[0])
        pyplot.show()
            
###############################################################
    elif problem == 5: #kadai7
        """
        ランダムに生成された訓練データA, zを用いて、２乗ヒンジ損失を用いるソフトマージン最大化の問題を解く。
        それ以外はkadai6と同じ。
        """

        #define functions
        n = 1 #regularization parameter
        def evalb(y,x):
            #compute border
            w = array([y[0],y[1]])
            return inner(w,x) + y[2]
            
        def evall(x):
            #compute a 0-1 loss function
            return math.pow(max(0,1-x),2)
            
        def evalf(y,z,B):
            #compute a function f
            w = array([y[0],y[1]])
            f = 0.5*math.pow(linalg.norm(w),2)
            for i in range(len(B)):
                f = f + n*evall(z[i]*evalb(y,B[i]))
            return f

        def evalg(y,B):
            #compute the gradient of f
            g = array([y[0],y[1],0])
            for i in range(len(B)):
                g = g - 2*n*z[i]*max(0,1-z[i]*evalb(y,B[i]))*array([B[i][0],B[i][1],1])
            return g

        mu1 = [-2,-2] #mean 1
        mu2 = [2,2] #mean 2
        cov1 = [[2,1],[1,2]] #covariance matrix
        cov2 = [[1,0],[0,1]] #covariance matrix
        N = [30,60,90] # number of the training data
        
        #find the optimal solution
        for i in range(3):
            
            #training data and labels
            A1 = random.multivariate_normal(mu1,cov1,N[i]//2)
            A2 = random.multivariate_normal(mu2,cov2,N[i]//2)
            z1 = ones(N[i]//2)
            z2 = -ones(N[i]//2)
            A = concatenate((A1, A2), axis=0)
            z = concatenate((z1, z2))
            
            #initial points
            w = array([1,1])
            b = 1
            y = array([w[0],w[1],b])
        
            B = eye(3)  #initial B used if method is quasi-Newton method                                               
            k = 0 #number of iterations
            epsilon = 3 * math.pow(10,-6) #終了条件のパラメータ
            
            while linalg.norm(evalg(y,A)) > epsilon:
            
                #compute a search direction
                if method == 1: #Steepest descent
                    d = -evalg(y,A)                               
                elif method == 3:  #quasi-Newton method
                    L = linalg.cho_factor(B) #Cholesky factorization
                    d = linalg.cho_solve(L, -evalg(y,A)) #search direction
                else:
                    sys.exit()
                    
                f = evalf(y,z,A) #objective function
                e = inner(d,evalg(y,A)) #dot product search direction and the gradient
                t = 1.0 #initial step size
                    
                #find t(step size)
                while evalf(y+t*d,z,A) > f + t*xi*e:
                    if search_method == 2: #quadratic interpolation method
                        t_old = t #previous t
                        t = -e*math.pow(t,2)/2*(evalf(y+t*d,z,A)-f-t*e)
                        if t < 0.1*t_old or t >  0.9*t_old:
                            t = t_old/2.0
                    else:
                        sys.exit()
                
                y_old = y #previous y
                y = y_old + t*d
                if method == 3: #if the method is quasi-Newton method
                    #recalculate B
                    s = y - y_old
                    Y = evalg(y,A) - evalg(y_old,A)
                    B = B - outer(B.dot(s),B.dot(s))/inner(s,B.dot(s)) + outer(Y,Y)/inner(s,Y)
                k = k + 1
            print('訓練事例の数N = {0}, 反復回数k = {1}'.format(N[i],k))
    
            # plot the points which A contains
            pyplot.figure(figsize = (9,6))
            if method == 1:
                pyplot.title('The number of the training data N = {0}, Steepest Descent, Quadratic Interpolation Method'.format(N[i]))
            else: #method == 3
                pyplot.title('The number of the training data N = {0}, Quasi-Newton Method, Quadratic Interpolation Method'.format(N[i]))
            pyplot.xlabel('$x_0$')
            pyplot.ylabel('$x_1$')
            for i in range(N[i]):
                if z[i] == 1:
                    pyplot.plot(A[i][0],A[i][1],color='r',marker='o')
                else: # z[i] == -1
                    pyplot.plot(A[i][0],A[i][1],color='b',marker='o')
                        
            #plot the graph
            if y[1] != 0:
                X = linspace(-15,15,100)
                Y = - (y[0]/y[1])*X - y[2]/y[1]
                pyplot.plot(X,Y)
            else:
                pyplot.axvline(x = -y[2]/y[0])
        pyplot.show()
    else:
        sys.exit()
if __name__ == '__main__':
        main()
