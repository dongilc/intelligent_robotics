# Intelligent Robotics Python Library, BSD License.
# Written by Dongil Choi, MyongJi University, South Korea. dongilc@mju.ac.kr
# Theoretical Background (Textbook) : 지능형 로봇공학 / 문승빈, 고경철, 곽관웅, 강병훈, 이순걸, 김종형 지음 / 사이텍미디어
# Symbolic Calculation Functions : DH Parameter, Jacobian, Statics, Dynamics (Newton-Euler & Lagrangian)
# 2020 / 06 / 08
# How to use : 
#       import intelligent_robotics as ir
#       dir(ir)
#       help(ir.DH)

import sympy
from sympy.physics.vector import dynamicsymbols
from sympy.physics.vector import time_derivative
from sympy.physics.vector import ReferenceFrame
N = ReferenceFrame('N')

###### 동차변환
# DH 파라미터를 이용해 Homogeneous Transformation Matrix 만드는 함수
def DH(a, alpha, d, theta):
    T_rot_x = sympy.Matrix([[1,0,0,0],
                            [0,sympy.cos(alpha),-sympy.sin(alpha),0],
                            [0,sympy.sin(alpha),sympy.cos(alpha),0],
                            [0,0,0,1]]);
    T_trans_x = sympy.Matrix([[1,0,0,a],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1]]);
    T_trans_z = sympy.Matrix([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,d],
                              [0,0,0,1]]);
    T_rot_z = sympy.Matrix([[sympy.cos(theta),-sympy.sin(theta),0,0],
                            [sympy.sin(theta),sympy.cos(theta),0,0],
                            [0,0,1,0],
                            [0,0,0,1]]);
    T = T_rot_x*T_trans_x*T_trans_z*T_rot_z;
    return T

###### 동차변환
# 동차변환에서 회전행렬만 뽑아내는 함수
def get_R_from_T(T):
    R = T[0:3,0:3];
    return R

###### 동차변환
# 동차변환에서 위치백터만 뽑아내는 함수
def get_P_from_T(T):
    P = T[0:3,3];
    return P

###### 자코비안
# 선속도와 각속도를 입력으로 받아 자코비안을 구하는 함수
def get_Jacobian_from_vel(w_0_n,v_0_n,qd):
    J = sympy.zeros(6,len(qd));
    i = 0;
    for qd_i in qd:
        j_i = [];
        j_i.append(sympy.simplify(sympy.diff(v_0_n,qd_i)));
        j_i.append(sympy.simplify(sympy.diff(w_0_n,qd_i)));
        J[:,i] = sympy.Matrix(j_i);
        i+=1;
    return J

###### 자코비안
# 힘과 토크를 입력으로 받아 자코비안 T 을 구하는 함수
def get_Jacobian_from_ft(f,n):
    Jt = sympy.zeros(len(n),len(f));
    i = 0;
    for f_i in f:
        jt_i = [];
        jt_i.append(sympy.simplify(sympy.diff(n,f_i)));
        Jt[:,i] = sympy.Matrix(jt_i);
        i+=1;
    return Jt

###### 정역학
# 정역학 - 힘 구하는 공식
def get_statics_force_i(T_i_ip1, f_ip1_ip1):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    f_i_i = R_i_ip1@f_ip1_ip1;
    return sympy.simplify(f_i_i)

###### 정역학
# 정역학 - 토크 구하는 공식
def get_statics_torque_i(T_i_ip1, n_ip1_ip1, f_i_i):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    r_i_ip1 = get_P_from_T(T_i_ip1);
    n_i_i = R_i_ip1@n_ip1_ip1 + r_i_ip1.cross(f_i_i);
    return sympy.simplify(n_i_i)

###### 동역학 공식 - Newton-Euler
###### 1. Forward Iteration
### 회전관절
# 속도전파식을 이용하여 각속도를 구하는 함수
def get_angular_vel_R(T_i_ip1,w_i_i,thd_ip1):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    R_ip1_i = sympy.transpose(R_i_ip1);
    Z_ip1_ip1 = sympy.Matrix([[0],[0],[1]]);
    w_ip1_ip1 = R_ip1_i@w_i_i + thd_ip1*Z_ip1_ip1;
    return w_ip1_ip1    

###### 동역학 공식 - Newton-Euler
### 회전관절
# 속도전파식을 이용하여 각가속도를 구하는 함수
def get_angular_acc_R(T_i_ip1,w_i_i,wd_i_i,thd_ip1,thdd_ip1):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    R_ip1_i = sympy.transpose(R_i_ip1);
    Z_ip1_ip1 = sympy.Matrix([[0],[0],[1]]);
    w_ip1_i = R_ip1_i@w_i_i; # 이부분이 수정됨.
    wd_ip1_ip1 = R_ip1_i@wd_i_i + thdd_ip1*Z_ip1_ip1 + w_ip1_i.cross(thd_ip1*Z_ip1_ip1);
    return wd_ip1_ip1 

###### 동역학 공식 - Newton-Euler
### 회전관절
# 속도전파식을 이용하여 선속도를 구하는 함수
def get_linear_vel_R(T_i_ip1,w_i,v_i):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    R_ip1_i = sympy.transpose(R_i_ip1);
    r_i_ip1 = T_i_ip1[0:3,3:4];
    v_ip1_ip1 = R_ip1_i@(v_i + w_i.cross(r_i_ip1));
    return v_ip1_ip1

###### 동역학 공식 - Newton-Euler
### 회전관절
# 속도전파식을 이용하여 선가속도를 구하는 함수
def get_linear_acc_R(T_i_ip1,w_i_i,wd_i_i,vd_i):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    R_ip1_i = sympy.transpose(R_i_ip1);
    r_i_ip1 = T_i_ip1[0:3,3:4];
    w_ip1_i = R_ip1_i@w_i_i; # 이부분이 수정됨.
    vd_ip1_ip1 = R_ip1_i@vd_i + R_ip1_i@(wd_i_i.cross(r_i_ip1)) + w_ip1_i.cross(R_ip1_i@(w_i_i.cross(r_i_ip1)));
    return vd_ip1_ip1

###### 동역학 공식 - Newton-Euler
### 직동관절
# 속도전파식을 이용하여 각속도를 구하는 함수
def get_angular_vel_P(T_i_ip1,w_i_i):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    R_ip1_i = sympy.transpose(R_i_ip1);
    w_ip1_ip1 = R_ip1_i@w_i_i;
    return w_ip1_ip1

###### 동역학 공식 - Newton-Euler
### 직동관절
# 속도전파식을 이용하여 각가속도를 구하는 함수
def get_angular_acc_P(T_i_ip1,wd_i_i):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    R_ip1_i = sympy.transpose(R_i_ip1);
    wd_ip1_ip1 = R_ip1_i@wd_i_i;
    return wd_ip1_ip1 

###### 동역학 공식 - Newton-Euler
### 직동관절
# 속도전파식을 이용하여 선속도를 구하는 함수
def get_linear_vel_P(T_i_ip1,w_i_i,v_i_i,dd_ip1):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    R_ip1_i = sympy.transpose(R_i_ip1);
    r_i_ip1 = T_i_ip1[0:3,3:4];
    Z_ip1_ip1 = sympy.Matrix([[0],[0],[1]]);
    v_ip1_ip1 = R_ip1_i@(v_i_i + w_i_i.cross(r_i_ip1)) + dd_ip1*Z_ip1_ip1;
    return v_ip1_ip1

###### 동역학 공식 - Newton-Euler
### 직동관절
# 속도전파식을 이용하여 선가속도를 구하는 함수
def get_linear_acc_P(T_i_ip1,w_i_i,wd_i_i,w_ip1_ip1,v_i_i,vd_i_i,dd_ip1,ddd_ip1):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    R_ip1_i = sympy.transpose(R_i_ip1);
    r_i_ip1 = T_i_ip1[0:3,3:4];
    Z_ip1_ip1 = sympy.Matrix([[0],[0],[1]]);
    vd_ip1_ip1 = R_ip1_i@vd_i_i + ddd_ip1*Z_ip1_ip1 + 2*w_ip1_ip1.cross(dd_ip1*Z_ip1_ip1) + R_ip1_i@(wd_i_i.cross(r_i_ip1) + w_i_i.cross(w_i_i.cross(r_i_ip1)));
    return vd_ip1_ip1

###### 동역학 공식 - Newton-Euler
### 직동관절
# 질량중심의 속도 구하는 함수
def get_com_vel(v_i_i,w_i_i,r_i_Gi):
    v_i_Gi = v_i_i +  w_i_i.cross(r_i_Gi);
    return v_i_Gi

###### 동역학 공식 - Newton-Euler
### 직동관절
# 질량중심의 가속도 구하는 함수
def get_com_acc(vd_i_i,w_i_i,wd_i_i,r_i_Gi):
    vd_i_Gi = vd_i_i + wd_i_i.cross(r_i_Gi) + w_i_i.cross(w_i_i.cross(r_i_Gi));
    return vd_i_Gi

###### 동역학 공식 - Newton-Euler
###### 2. Inverse Iteration
# 동역학 - 힘 구하는 공식
def get_dynamics_force_i(T_i_ip1,T_0_i,m_i,g_0,vd_i_Gi,f_ip1_ip1):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    R_0_i = get_R_from_T(T_0_i);
    R_i_0 = sympy.transpose(R_0_i);
    f_i_i = m_i*vd_i_Gi + R_i_ip1@f_ip1_ip1 - m_i*R_i_0*g_0;
    return sympy.simplify(f_i_i)

###### 동역학 공식 - Newton-Euler
# 동역학 - 토크 구하는 공식
def get_dynamics_torque_i(T_i_ip1,n_ip1_ip1,f_i_i,f_ip1_ip1,w_i_i,wd_i_i,r_i_Gi,I_i_Gi):
    R_i_ip1 = get_R_from_T(T_i_ip1);
    R_ip1_i = sympy.transpose(R_i_ip1);
    r_i_ip1 = T_i_ip1[0:3,3:4];
    r_ip1_Gi = r_i_Gi - r_i_ip1;   
    n_i_i = I_i_Gi@wd_i_i + w_i_i.cross(I_i_Gi@w_i_i) + R_i_ip1@n_ip1_ip1 + r_i_Gi.cross(f_i_i) - r_ip1_Gi.cross(R_i_ip1@f_ip1_ip1);
    return sympy.simplify(n_i_i)

###### 동역학 공식 - Lagrangian
###### 운동에너지 T 구하는 부분 
# T 구하는 함수
def get_kinectic_energy(m_i,v_i_Gi,I_i_Gi,w_i_i):
    T = 1/2*m_i*v_i_Gi.T@v_i_Gi + 1/2*w_i_i.T*I_i_Gi*w_i_i;
    return T

###### 동역학 공식 - Lagrangian
###### 위치에너지 V를 구하는 함수
def get_potential_energy(T_0_i,m_i,g_0,r_i_Gi):
    r_0_i = T_0_i[0:3,3:4];
    R_0_i = get_R_from_T(T_0_i);
    r_0_Gi = r_0_i + R_0_i@r_i_Gi;
    V = -m_i*g_0.T@r_0_Gi;
    return V

###### 동역학 공식 - Lagrangian
###### 라그랑지안 L 구하는 함수
def get_lagrangian(T,V):
    L = T - V;
    return L

###### 동역학 공식 - Lagrangian
###### 라그랑지안을 이용해 토크 계산
def get_torque_from_L(L,q,qd):
    round_L_round_q = sympy.zeros(len(q),1);
    i = 0;
    for q_i in q:
        round_L_round_q_i = [];
        round_L_round_q_i = sympy.simplify(sympy.diff(L,q_i));
        round_L_round_q[i] = sympy.Matrix(round_L_round_q_i);
        i+=1;
     
    d_dt_round_L_round_qd = sympy.zeros(len(qd),1);
    i = 0;
    for qd_i in qd:
        round_L_round_qd_i = [];
        d_dt_round_L_round_qd_i = [];
        round_L_round_qd_i = sympy.diff(L,qd_i);
        d_dt_round_L_round_qd_i = sympy.simplify(time_derivative(round_L_round_qd_i,N));
        d_dt_round_L_round_qd[i] = sympy.Matrix(d_dt_round_L_round_qd_i);
        i+=1;
        
    tau = d_dt_round_L_round_qd - round_L_round_q 
    return tau

###### 동역학 공식 - Lagrangian
###### 머니퓰레이터 운동방적식
def get_EoM_from_T(tau,qdd,g):
    # Inertia Matrix, M(q)를 구해주는 부분
    M = sympy.zeros(len(tau));
    i = 0;
    for tau_i in tau:
        M_i = [];
        M_i.append(sympy.simplify(sympy.diff(tau_i,qdd)));
        M[:,i] = sympy.Matrix(M_i);
        i+=1;

    # Gravity Matrix, G(q) 를 구해주는 부분
    G = sympy.zeros(len(tau),1);
    i = 0;
    for tau_i in tau:
        G_i = [];
        G_i.append(sympy.simplify(sympy.diff(tau_i,g)));
        G[i] = sympy.Matrix(G_i);
        i+=1;
        
    # 원심력 & 코리올리스 행렬, C(q,qd) 를 구해주는 부분
    C = sympy.simplify(tau - M@qdd - G*g);
    
    return sympy.simplify(M), sympy.simplify(C), sympy.simplify(G*g)

###### 동역학 : C term을 M term으로 부터 구하는 방법
# Christoffel Symbol (Gamma)
# C(q,qd) = qd^T * Gamma(q) * qd 
# Gammga_ijk(q) = 1/2 * (round_m_ij/round_q_k + round_m_ik/round_q_j - round_m_jk/round_q_i)
def get_Christoffel_term(M,q,qd):
    n = len(qd);
    C = sympy.zeros(n,1);
    Gamma = [];
    for i in range(n):
        Gamma_i = sympy.zeros(n,n);
        for j in range(n):
            for k in range(n):
                Gamma_i[j,k] = 1/2*(sympy.diff(M[i,j],q[k]) + sympy.diff(M[i,k],q[j]) - sympy.diff(M[j,k],q[i]));
        Gamma.append(Gamma_i);
        
    for i in range(n):
        Gamma_i = Gamma[i];
        C[i] = sympy.simplify(qd.T@Gamma_i@qd);
        
    return Gamma, C