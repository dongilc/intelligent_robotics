# Modern Robotics Python Symbolic Calculation Library, BSD License.
# Written by Dongil Choi, MyongJi University, South Korea. dongilc@mju.ac.kr
# Theoretical Background (Textbook) : Modern Robotics, Kevin M. Lynch and Frank C. Park
#
# 2021 / 06 / 02
# How to use : 
#       import symbolic_modern_robotics as smr
#       dir(smr)

import sympy as s
from sympy.physics.vector import dynamicsymbols
from sympy.physics.vector import time_derivative
from sympy.physics.vector import ReferenceFrame
N = ReferenceFrame('N')

###### Rotation Matrix
# Calculate so3 from 3-vector
def VecToso3(w):
    o1 = w[0]
    o2 = w[1]
    o3 = w[2]
    skew_w = s.Matrix([[0,-o3,o2],
                       [o3,0,-o1],
                       [-o2,o1,0]])
    return skew_w

###### Rotation Matrix
# Calculate 3-vector from so3
def so3ToVec(so3mat):
    w = s.Matrix([[0],[0],[0]]);
    w[0] = -so3mat[1,2]
    w[1] = so3mat[0,2]
    w[2] = -so3mat[0,1]
    return w

###### Rotation Matrix
# Calculate unit axis of rotation(omega_hat) and theta from 3-vector
def AxisAng3(w):
    norm = w.norm()
    w_hat = w/norm
    theta = norm
    
    return w_hat, theta

###### Rotation Matrix
# Calculate SO3 from so3 by Matrix exponential
def MatrixExp3(so3mat):
    w = so3ToVec(so3mat)
    w_hat,theta = AxisAng3(w)
    skew_w = VecToso3(w_hat)
    I = s.Matrix([[1,0,0],[0,1,0],[0,0,1]])
    R = s.simplify(I + s.sin(theta)*skew_w + (1-s.cos(theta))*skew_w*skew_w)
    return R
        
###### Rotation Matrix
# Calculate so3 from SO3 by Matrix logarithm
def MatrixLog3(SO3):
    R = SO3
    R_T = R.T
    tr_R = s.simplify(s.Trace(R))
    theta = s.simplify(s.acos((tr_R-1)/2))
    w_hat_skew = s.simplify(1/(2*s.sin(theta))*(R-R_T))
    so3 = s.simplify(w_hat_skew*theta)
    return so3

###### Homogeneous Transform Matrix
# Calculate se3 from 6-vector twist
def VecTose3(V):
    w = s.Matrix(V[0:3])
    v = s.Matrix(V[3:6])
    skew_w = VecToso3(w)
    se3 = s.Matrix([[skew_w,v],
                    [0,0,0,0]])
    return se3

###### Homogeneous Transform Matrix
# Calculate screw-axis and angle from 6-vector
def AxisAng6(V):
    w = s.Matrix(V[0:3])
    v = s.Matrix(V[3:6])
    w_norm = w.norm()
    
    if w_norm == 0:
        norm = v.norm()
    else:
        norm = w.norm()
        
    S = s.simplify(V/norm)
    theta_dot = norm
    return S,theta_dot 

###### Homogeneous Transform Matrix
# Calculate 6-vector twist from se3
def se3ToVec(se3mat):
    V = s.Matrix([[0],[0],[0],[0],[0],[0]]);
    V[0] = -se3mat[1,2]
    V[1] = se3mat[0,2]
    V[2] = -se3mat[0,1]
    V[3] = se3mat[0,3]
    V[4] = se3mat[1,3]
    V[5] = se3mat[2,3]
    return V

###### Homogeneous Transform Matrix
# Calculate SE3 from se3 by Matrix exponential
def MatrixExp6(se3mat):
    V = se3ToVec(se3mat)
    w = s.Matrix(V[0:3])
    
    if w.norm() == 0:
        v = s.Matrix(V[3:6])
        I = s.Matrix([[1,0,0],[0,1,0],[0,0,1]])
        SE3 = s.Matrix([[I,v],
                        [0,0,0,1]])
    else:
        w_hat, theta = AxisAng3(w)
        v = s.Matrix(V[3:6])/theta
        so3_hat = VecToso3(w_hat)
        R = MatrixExp3(so3_hat*theta)
        I = s.Matrix([[1,0,0],[0,1,0],[0,0,1]])
        G_theta = I*theta + (1-s.cos(theta))*so3_hat + (theta-s.sin(theta))*so3_hat*so3_hat
        p = s.simplify(G_theta*v)
        SE3 = s.Matrix([[R,p],
                        [0,0,0,1]])
    #return V,w,w_hat,theta,v,so3_hat,R,p,SE3
    return SE3

###### Homogeneous Transform Matrix
# Calculate se3 from SE3 by Matrix logarithm
def MatrixLog6(T):
    R = T[0:3,0:3]
    p = T[0:3,3]
    
    I = s.Matrix([[1,0,0],[0,1,0],[0,0,1]])
    
    if R == I:
        w_zero = s.Matrix([[0,0,0],[0,0,0],[0,0,0]])
        se3 = s.Matrix([[w_zero,p],
                        [0,0,0,0]]) 
    else:
        so3 = MatrixLog3(R)
        w = so3ToVec(so3)
        w_hat,theta = AxisAng3(w)
        so3_hat = VecToso3(w_hat)
        I = s.Matrix([[1,0,0],[0,1,0],[0,0,1]])
        G_inv_theta = s.simplify(1/theta*I - 1/2*so3_hat + (1/theta - 1/2*s.cot(theta/2))*so3_hat*so3_hat)
        v = s.simplify(G_inv_theta*p)
        se3 = s.Matrix([[so3,v],
                        [0,0,0,0]])
    #return R, p, so3, w, w_hat, theta, so3_hat, G_inv_theta, v, se3 
    return se3

###### Space Form PoE
# Computes forward kinematics in the space frame
def FKinSpace(M, Slist, thetalist):
    T = M
    for i in range(len(thetalist) - 1, -1, -1):
       T = s.simplify(MatrixExp6(VecTose3(Slist[:, i]*thetalist[i]))@T)
    return T

###### Body Form PoE
# Computes forward kinematics in the body frame
def FKinBody(M, Blist, thetalist):
    T = M
    for i in range(len(thetalist)):
       T = s.simplify(T@MatrixExp6(VecTose3(Blist[:, i]*thetalist[i])))
    return T

##### Adjoint of T
# Computes Adjoint of T
def Adjoint(T):
    R = T[0:3,0:3]
    p = T[0:3,3]
    
    Ad_T = s.Matrix([[R, s.Matrix.zeros(3)],
                     [VecToso3(p)@R ,R]])
    return Ad_T

##### adjoint of V (Lie Bracket of V)
# Computes adjoint of Twist, V
def ad(V):
    w = s.Matrix(V[0:3])
    v = s.Matrix(V[3:6])
        
    skew_w = VecToso3(w)
    skew_v = VecToso3(v)

    adV = s.Matrix([[skew_w, s.Matrix.zeros(3)],
                    [skew_v, skew_w]])
    return adV

##### Inverse of Homogeneos Transformation Matrix
# Computes inverse of T
def TransInv(T):
    R = T[0:3,0:3]
    p = T[0:3,3]

    T_inv = s.Matrix([[R.T, -R.T@p],
                      [0,0,0,1]])
    return T_inv

###### Body Jacobian
# Computes Body Jacobian
def JacobianBody(Blist, thetalist):
    Jb = Blist.copy()
    T = s.Matrix.eye(4)
    for i in range(len(thetalist) - 2, -1, -1):
        T = T @ MatrixExp6(VecTose3(Blist[:, i+1]*-thetalist[i+1]))
        Jb[:, i] = s.simplify(Adjoint(T)@Blist[:, i])
    return Jb

###### Space Jacobian
# Computes Space Jacobian
def JacobianSpace(Slist, thetalist):
    Js = Slist.copy()
    T = s.Matrix.eye(4)
    for i in range(1, len(thetalist)):
        T = T @ MatrixExp6(VecTose3(Slist[:, i-1]*thetalist[i-1]))
        Js[:, i] = s.simplify(Adjoint(T)@Slist[:, i])
        #print(i, Js[:, i])
    return Js

##### Inverse Dynamics
# Newton-Euler Inverse Dynamics Calculation
# 1. Forward Iteration : Calculate twist, twist_dot from base to tip
# 2. Backward Iteration : Calculate F, Tau from tip to base
def InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist):
    n = len(thetalist)  # Degree of freedom
    
    Mi = s.Matrix.eye(4)    # M_0,i - M1, M2, M3 ...
    Ai = s.Matrix.zeros(6,n) # Body Screw - A1, A2, A3 ...
    
    Vi = s.Matrix.zeros(6, n + 1) # twist - V0, V1, V2 ...
    Vi_dot = s.Matrix.zeros(6, n + 1) # twist_dot - V_dot0, V_dot1, ...
    Vi_dot[:, 0] = s.Matrix([0, 0, 0, -g[0], -g[1], -g[2]])

    AdTi_im1 = [[None]] * (n + 1)
    
    Fi = Ftip.copy()
    taulist = s.Matrix.zeros(n, 1)
    
    # Forward Iteration
    for k in range(n):
        i = k+1
        Mi = Mi@Mlist[k]
        Ai[:,k] = Adjoint(TransInv(Mi))@Slist[:,k]
        Tim1_i = s.simplify(Mlist[k]@MatrixExp6(VecTose3(Ai[:,k]*thetalist[k])))
        Ti_im1 = s.simplify(TransInv(Tim1_i))
        AdTi_im1[k] = Adjoint(Ti_im1)
        Vi[:,i] = s.simplify(Ai[:,k]*dthetalist[k] + AdTi_im1[k]@Vi[:,k])
        Vi_dot[:,i] = s.simplify(Ai[:,k]*ddthetalist[k] + AdTi_im1[k]@Vi_dot[:,k] + ad(Vi[:,k])@Ai[:,k]*dthetalist[k])
        #print(i)
        #print(AdTi_im1[k])
        #print(Ai[:,k])
    
    #print(Vi)
    #print(Vi_dot)
    
    AdTi_im1[n] = Adjoint(TransInv(Mi))
    #print(AdTi_im1)
    
    # Backward Iteration
    for k in range (n - 1, -1, -1):
        i = k+1
        Fi = AdTi_im1[i].T@Fi + Glist[k]@Vi_dot[:,i] - ad(Vi[:,i]).T@(Glist[k]@Vi[:,i])
        taulist[k] = s.simplify( Fi.T@Ai[:,k] )
    return taulist, Vi, Vi_dot

###### 머니퓰레이터 운동방정식 정리해주는 함수
def get_EoM_from_T(tau,qdd,g):
    # Inertia Matrix, M(q)를 구해주는 부분
    M = s.zeros(len(tau));
    i = 0;
    for tau_i in tau:
        M_i = [];
        M_i.append(s.simplify(s.diff(tau_i,qdd)));
        M[:,i] = s.Matrix(M_i);
        i+=1;

    # Gravity Matrix, G(q) 를 구해주는 부분
    G = s.zeros(len(tau),1);
    i = 0;
    for tau_i in tau:
        G_i = [];
        G_i.append(s.simplify(s.diff(tau_i,g)));
        G[i] = s.Matrix(G_i);
        i+=1;
        
    # 원심력 & 코리올리스 행렬, C(q,qd) 를 구해주는 부분
    C = s.simplify(tau - M@qdd - G*g);
    
    return s.simplify(M), s.simplify(C), s.simplify(G*g)