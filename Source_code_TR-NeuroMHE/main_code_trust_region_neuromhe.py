"""
This is the main function that trains 2nd-order TR-NeuroMHE and compares it with NeuroBEM
The training is done in a supervised learning fashion on a real flight dataset[1]
[1] Bauersfeld, L., Kaufmann, E., Foehn, P., Sun, S. and Scaramuzza, D., 2021. 
    NeuroBEM: Hybrid Aerodynamic Quadrotor Model. ROBOTICS: SCIENCE AND SYSTEM XVII.
----------------------------------------------------------------------------
WANG, Bingheng, 3 Sept. 2023, at Control & Simulation Lab, ECE Dept. NUS
Should you have any questions, please feel free to contact the author via: wangbingheng@u.nus.edu

"""
import Uav_Env
import Uav_mhe_SL_Hessian_trust_region_neural
from casadi import *
import time as TM
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

"""---------------------------------Load environment---------------------------------------"""
# System parameters used in the paper of 'NeuroBEM'
# Sys_para = np.array([0.752, 0.00252, 0.00214, 0.00436])
Sys_para  = np.array([0.772,0.0025,0.0021,0.0043]) # updated by the author of NeuroBEM

# Sampling time-step, 400Hz
dt_sample = 0.0025
uav       = Uav_Env.quadrotor(Sys_para, dt_sample)
uav.model()
horizon   = 10
key3      = '10'
# Learning rate
lr_nn     = 1e-4

"""---------------------------------Define neural network model-----------------------------"""
D_in, D_h, D_out = 6, 8, 26
uavNet    = Uav_mhe_SL_Hessian_trust_region_neural.Net(D_in, D_h, D_out)
"""---------------------------------Quaternion to Rotation Matrix---------------------------"""
def Quaternion2Rotation(q): 
    q = q/LA.norm(q) # normalization, which is very important to guarantee that the resulting R is a rotation matrix (Lie group: SO3)
    q0, q1, q2, q3 = q[0,0], q[1,0], q[2,0], q[3,0]
    R = np.array([
        [2 * (q0 ** 2 + q1 ** 2) - 1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q0 * q2 + 2 * q1 * q3],
        [2 * q0 * q3 + 2 * q1 * q2, 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2, 2 * q0 * q1 + 2 * q2 * q3, 2 * (q0 ** 2 + q3 ** 2) - 1]
    ]) # from body frame to inertial frame
    return R

"""---------------------------------Compute Ground Truth------------------------------------"""
def GroundTruth(w_B, acc, mass, J_B):
    acc_p = np.array([[acc[0, 0], acc[1, 0], acc[2, 0]]]).T # measured in body frame, already including the gravity
    acc_w = np.array([[acc[3, 0], acc[4, 0], acc[5, 0]]]).T # measured in body frame
    
    df    = mass*acc_p
    dt    = np.matmul(J_B, acc_w) + \
            np.cross(w_B.T, np.transpose(np.matmul(J_B, w_B))).T 
    return df, dt

"""---------------------------------Define MHE----------------------------------------------"""
uavMHE = Uav_mhe_SL_Hessian_trust_region_neural.MHE(horizon, dt_sample)
uavMHE.SetStateVariable(uav.xa)
uavMHE.SetOutputVariable(uav.y)
uavMHE.SetNoiseVariable(uav.w)
uavMHE.SetModelDyn(uav.dyn)
uavMHE.SetCostDyn()

"""---------------------------------Define Trust-region System------------------------------"""
uavTRS = Uav_mhe_SL_Hessian_trust_region_neural.Trust_region(uavNet.nn_para)

"""---------------------------------Define NeuroMHE-----------------------------------------"""
uavNMHE = Uav_mhe_SL_Hessian_trust_region_neural.KF_gradient_solver(uav.xa, uavMHE.weight_para, uav.w, uavNet.nn_para)

"""---------------------------------Training process----------------------------------------"""
# Parameterization of the MHE tuning parameters (for guaranteeing positive definiteness)
epsilon, gmin = 1e-4, 1e-4
def SetPara(epsilon, gmin, tunable_para):
    p_diag = np.zeros((1, 12))
    for i in range(12):
        p_diag[0, i] = epsilon + tunable_para[0, i]**2
    P0 = np.diag(p_diag[0])

    gamma_r = gmin + (1 - gmin) * 1/(1+np.exp(-tunable_para[0,12]))
    gamma_q = gmin + (1 - gmin) * 1/(1+np.exp(-tunable_para[0,19]))

    # r_diag = np.zeros((1, 5))
    r_diag = np.zeros((1, 6))
    for i in range(6):
        r_diag[0, i] = epsilon + tunable_para[0, i+13]**2
    R_N    = np.diag(r_diag[0])

    q_diag = np.zeros((1, 6))
    for i in range(6):
        q_diag[0, i] = epsilon + tunable_para[0, i+20]**2
    Q_N1   = np.diag(q_diag[0])
    weight_para = np.hstack((p_diag, np.reshape(gamma_r, (1,1)), r_diag, np.reshape(gamma_q,(1,1)), q_diag))
    return P0, gamma_r, gamma_q, R_N, Q_N1, weight_para

# Gradient of the parameterized MHE tuning parameters w.r.t the outputs of the neural network
def chainRule_gradient(epsilon, gmin, tunable_para):
    tunable = SX.sym('tunable', 1, 26)
    P = SX.sym('P', 1, 12)
    for i in range(12):
        P[0, i] = epsilon + tunable[0, i]**2

    gamma_r = gmin + (1 - gmin) * 1/(1+exp(-tunable[0,12]))
    R = SX.sym('R', 1, 6)
    for i in range(6):
        R[0, i] = epsilon + tunable[0, i+13]**2
    
    gamma_q = gmin + (1 - gmin) * 1/(1+exp(-tunable[0,19]))
    Q = SX.sym('Q', 1, 6)
    for i in range(6):
        Q[0, i] = epsilon + tunable[0, i+20]**2
    weight = horzcat(P, gamma_r, R, gamma_q, Q)
    w_jaco = jacobian(weight, tunable)
    w_jaco_fn = Function('W_fn',[tunable],[w_jaco],['tp'],['W_fnf'])
    weight_grad = w_jaco_fn(tp=tunable_para)['W_fnf'].full()
    return weight_grad

def chainRule_hessian(epsilon, gmin, tunable_para):
    tunable = SX.sym('tunable', 1, 26)
    P = SX.sym('P', 1, 12)
    for i in range(12):
        P[0, i] = epsilon + tunable[0, i]**2

    gamma_r = gmin + (1 - gmin) * 1/(1+exp(-tunable[0,12]))
    R = SX.sym('R', 1, 6)
    for i in range(6):
        R[0, i] = epsilon + tunable[0, i+13]**2
    
    gamma_q = gmin + (1 - gmin) * 1/(1+exp(-tunable[0,19]))
    Q = SX.sym('Q', 1, 6)
    for i in range(6):
        Q[0, i] = epsilon + tunable[0, i+20]**2
    weight = horzcat(P, gamma_r, R, gamma_q, Q)
    w_hess = jacobian(jacobian(weight, tunable),tunable) # automatically result in a 625-by-25 matrix
    w_hess_fn = Function('W_fn',[tunable],[w_hess],['tp'],['W_fnf'])
    weight_hess = w_hess_fn(tp=tunable_para)['W_fnf'].full()
    return weight_hess

def Train():
    # Initialization of network learnable parameters (Kaiming and LeCun method)
    #---Initialization of bias using LeCun method---#
    b_1          = np.reshape(np.random.uniform(-1/np.sqrt(D_in),1/np.sqrt(D_in),D_h),(1,D_h))
    b_2          = np.reshape(np.random.uniform(-1/np.sqrt(D_h),1/np.sqrt(D_h),D_h),(1,D_h))
    b_o          = np.reshape(np.random.uniform(-1/np.sqrt(D_h),1/np.sqrt(D_h),D_out),(1,D_out))
    #---Initialization of weights using Kaiming method---#
    w_1          = np.reshape(np.random.uniform(-1/np.sqrt(D_in),1/np.sqrt(D_in),D_in*D_h),(1,D_in*D_h))
    w_2          = np.reshape(np.random.uniform(-1/np.sqrt(D_h),1/np.sqrt(D_h),D_h**2),(1,D_h**2))
    w_o          = np.reshape(np.random.uniform(-1/np.sqrt(D_h),1/np.sqrt(D_h),D_h*D_out),(1,D_h*D_out))
    # nn_para0     = np.hstack((b_1,b_2,b_o,w_1,w_2,w_o))
    # You can also use the saved initial network model (generated using the above initialization method)
    key2 = input("enter '1', or '2',... no more than '10' without the quotation mark:")
    nn_para0 = np.load('initial_TR_NeuroMHE_model/nn_para0_'+str(key2)+'.npy')
    # np.save('trained_data/nn_para0_4',nn_para0)
   
    # Sampling index (sample over every n data points from the real dataset)
    n = 1
    # Value of loss function
    Mean_Loss = []
    # Dictionary of the dataset for training
    train_set = {'a': "processed_data/merged_2021-02-23-14-41-07_seg_3.csv"} # an agile figure-8 trajectory that covers a wide range of velocity and multiple force spikes
    # train_set = {'b': "processed_data/merged_2021-02-03-13-44-49_seg_3.csv"} # slow trajectory
    # Number of trained episode
    n_ep = 0
    Trained_episode = []
    # Initial difference of loss and training stop criterion 
    delta_cost = 10000
    eps  = 1
    # Initial mean loss
    mean_loss0 = 0
    # Initial network parameters
    nn_para = nn_para0
    # Thresholds of the trust-region method
    t1,t2,t3   = 0, 0.01, 0.9
    chi1, chi2 = 0.99, 2
    upperb     = 1e3
    # Initial trust-region radius
    radius     = 1e1

    while delta_cost >= eps: #flag ==0
        if n_ep==1:
            start_time0 = TM.time()
        # flag = 1
        # Sum of loss
        sum_loss = 0.0
        # Sum of cpu runtime
        sum_time = 0.0
        # estimated process noise
        Grad_time = []
        n_start = 1200
        for key in train_set:
            it = 0
            Trained_episode += [n_ep]
            # Obtain the size of the data
            dataset    = pd.read_csv(train_set[key])
            dataframe  = pd.DataFrame(dataset)
            # Obtain the sequences of the state, and acceleration from the dataframe
            angaccx_seq, angaccy_seq, angaccz_seq = dataframe['ang acc x'], dataframe['ang acc y'], dataframe['ang acc z']
            angvelx_seq, angvely_seq, angvelz_seq = dataframe['ang vel x'], dataframe['ang vel y'], dataframe['ang vel z']
            qx_seq, qy_seq, qz_seq, qw_seq = dataframe['quat x'], dataframe['quat y'], dataframe['quat z'], dataframe['quat w']
            accx_seq, accy_seq, accz_seq = dataframe['acc x'], dataframe['acc y'], dataframe['acc z']
            velx_seq, vely_seq, velz_seq = dataframe['vel x'], dataframe['vel y'], dataframe['vel z']
            # Initial states
            v_B0       = np.array([[velx_seq[n_start], vely_seq[n_start], velz_seq[n_start]]]).T
            q0         = np.array([[qw_seq[n_start], qx_seq[n_start], qy_seq[n_start], qz_seq[n_start]]]).T
            w_B0       = np.array([[angvelx_seq[n_start], angvely_seq[n_start], angvelz_seq[n_start]]]).T
            R_B0       = Quaternion2Rotation(q0)  
            v_I0       = np.matmul(R_B0, v_B0)
            acc        = np.array([[accx_seq[n_start], accy_seq[n_start], accz_seq[n_start], angaccx_seq[n_start], angaccy_seq[n_start], angaccz_seq[n_start]]]).T
            df_t, dt_t = GroundTruth(w_B0, acc, uav.m, uav.J)
            df_t       = np.matmul(R_B0, df_t)
            df_B0      = np.reshape(df_t, (3, 1))
            dt_B0      = np.reshape(dt_t, (3, 1))
            x_hat0     = np.vstack((v_I0,df_B0, w_B0, dt_B0)) # in training, we set the inital guess to be the ground truth
            x_hat      = x_hat0 # initial guess in the MHE arrival cost
            xmhe_traj  = x_hat0.T
            noise_traj = np.zeros((1,6))
            # Measurement list
            Y = []
            # Quaternion list
            q_seq = []
            # Ground_truth list
            ground_truth = []  

            for j in range(n_start,n_start+100): # these data points covers a wide range of velocity from 0.05 to 16.38 m/s 
                # Take the measurements from the dataset
                v_B = np.array([[velx_seq[n*j], vely_seq[n*j], velz_seq[n*j]]]).T
                q = np.array([[qw_seq[n*j], qx_seq[n*j], qy_seq[n*j], qz_seq[n*j]]]).T
                w_B = np.array([[angvelx_seq[n*j], angvely_seq[n*j], angvelz_seq[n*j]]]).T
                R_B = Quaternion2Rotation(q)  
                v_I = np.matmul(R_B, v_B) 
                measurement = np.vstack((v_I, w_B)) 
                Y += [measurement]
                q_seq += [q]
                # Take the output of the neural network model
                z5, z5_sym = uavNet.Forward(measurement,nn_para)
                tunable_para = np.reshape(z5,(1,D_out))
                P0, gamma_r, gamma_q, R_N, Q_N1, weight_para = SetPara(epsilon, gmin, tunable_para)
                print('Trained=', n_ep, 'sample=', it, 'p1=', P0[0, 0],'p2=', P0[1, 1], 'gamma1=', gamma_r,'gamma2=', gamma_q,'r1=', R_N[0, 0], 'r2=', R_N[1, 1],'r3=', R_N[2, 2])
                print('Trained=', n_ep, 'sample=', it, 'q1=', Q_N1[0, 0], 'q2=', Q_N1[1, 1], 'q3=', Q_N1[2, 2])
                # Ground truth list
                acc = np.array([[accx_seq[n*j], accy_seq[n*j], accz_seq[n*j], angaccx_seq[n*j], angaccy_seq[n*j], angaccz_seq[n*j]]]).T
                df_t, dt_t    = GroundTruth(w_B, acc, uav.m, uav.J)
                df_t          = np.matmul(R_B, df_t) # transformed to the world frame
                # Compute the gradient of loss
                state_t       = np.vstack((df_t, dt_t))
                ground_truth += [state_t]

                time_index   = j-n_start
                # Solve an MHE in the forward path
                if time_index <=horizon:
                    opt_sol      = uavMHE.MHEsolver(Y, x_hat, xmhe_traj, noise_traj, weight_para, time_index)
                    xmhe_traj    = opt_sol['state_traj_opt']
                    noise_traj   = opt_sol['noise_traj_opt']
                    costate_ipopt= opt_sol['costate_ipopt']
                    # Estimated disturbance for printing out
                    df_Imh = np.transpose(xmhe_traj[-1, 3:6])
                    df_Imh = np.reshape(df_Imh, (3, 1))
                    dt_Bmh = np.transpose(xmhe_traj[-1, 9:12])
                    dt_Bmh = np.reshape(dt_Bmh, (3, 1))
                    
                    print('Trained=', n_ep, 'sample=', it, 'Dis_x=', df_t[0, 0], 'df_Imh_x=', df_Imh[0, 0], 'Dis_y=', df_t[1, 0], 'df_Imh_y=', df_Imh[1, 0], 'Dis_z=', df_t[2, 0], 'df_Imh_z=', df_Imh[2, 0])
                    # Establish the auxiliary MHE system 
                    auxSys        = uavMHE.GetAuxSys_general(xmhe_traj, costate_ipopt, noise_traj, weight_para, Y)
                    M = np.zeros((len(x_hat),D_out)) # MHE estimate at t-N which is the second term in the estimate trajectory made at t-1 {from t-N-1 to t-1}
                    # Solve the auxiliary MHE system to obtain the gradient for back-propagation 
                    start_time = TM.time()
                    gra_opt = uavNMHE.GradientSolver_general(M, auxSys, weight_para)
                    gradtime = (TM.time() - start_time)*1000
                    print("--- %s ms ---" % format(gradtime,'.2f'))
                    Grad_time += [gradtime]
                    
                    # Update the gradient sequence
                    X_opt    = gra_opt['state_gra_traj']
                    LAMBDA   = gra_opt['costate_gra_traj']
                    W_opt    = gra_opt['noise_gra_traj']
                    # Solve the 2nd-order auxiliary MHE system to obtain the hessian for back-propagation 
                    auxSys_2nd = uavMHE.GetAuxSys_second_general(M, xmhe_traj, costate_ipopt, noise_traj, weight_para, Y, X_opt, LAMBDA, W_opt)    
                    HXhat = np.zeros((len(x_hat)*D_out,D_out))
                    start_time = TM.time()
                    HpX_opt = uavNMHE.GradientSolver_2nd_general(HXhat,auxSys,auxSys_2nd,weight_para)
                    hesstime = (TM.time() - start_time)*1000
                    # sum_time  += hesstime
                    print("hesstime=--- %s ms ---" % format(hesstime,'.2f'))
                    loss        = uavNMHE.loss_horizon(xmhe_traj,ground_truth,horizon,time_index)
                
                else:
                    for ix in range(len(x_hat)):
                        x_hat[ix,0] = xmhe_traj[1, ix] # update x^hat_{t-N} using the MHE estimate x^hat_{t-N|t-1} from the previous xmhe_traj
                    opt_sol      = uavMHE.MHEsolver(Y, x_hat, xmhe_traj, noise_traj, weight_para, time_index)
                    xmhe_traj_curr    = opt_sol['state_traj_opt']
                    noise_traj_curr   = opt_sol['noise_traj_opt']
                    costate_ipopt_curr= opt_sol['costate_ipopt']
                    # Estimated disturbance for printing out
                    df_Imh = np.transpose(xmhe_traj_curr[-1, 3:6])
                    df_Imh = np.reshape(df_Imh, (3, 1))
                    dt_Bmh = np.transpose(xmhe_traj_curr[-1, 9:12])
                    dt_Bmh = np.reshape(dt_Bmh, (3, 1))
                    print('Trained=', n_ep, 'sample=', it, 'Dis_x=', df_t[0, 0], 'df_Imh_x=', df_Imh[0, 0], 'Dis_y=', df_t[1, 0], 'df_Imh_y=', df_Imh[1, 0], 'Dis_z=', df_t[2, 0], 'df_Imh_z=', df_Imh[2, 0])
                    # Establish the auxiliary MHE system 
                    auxSys        = uavMHE.GetAuxSys_general(xmhe_traj_curr, costate_ipopt_curr, noise_traj_curr, weight_para, Y)
                    M = X_opt[1] # MHE estimate at t-N which is the second term in the estimate trajectory made at t-1 {from t-N-1 to t-1}
                    # Solve the auxiliary MHE system to obtain the gradient for back-propagation 
                    start_time = TM.time()
                    gra_opt = uavNMHE.GradientSolver_general(M, auxSys, weight_para)
                    gradtime = (TM.time() - start_time)*1000
                    print("--- %s ms ---" % format(gradtime,'.2f'))
                    Grad_time += [gradtime]
                    
                    # Update the gradient sequence
                    X_opt    = gra_opt['state_gra_traj']
                    LAMBDA   = gra_opt['costate_gra_traj']
                    W_opt    = gra_opt['noise_gra_traj']
                    # Solve the 2nd-order auxiliary MHE system to obtain the hessian for back-propagation 
                    auxSys_2nd = uavMHE.GetAuxSys_second_general(M, xmhe_traj_curr, costate_ipopt_curr, noise_traj_curr, weight_para, Y, X_opt, LAMBDA, W_opt)    
                    HXhat = HpX_opt[1]
                    start_time = TM.time()
                    HpX_opt = uavNMHE.GradientSolver_2nd_general(HXhat,auxSys,auxSys_2nd,weight_para)
                    hesstime = (TM.time() - start_time)*1000
                    sum_time  += hesstime
                    print("hesstime=--- %s ms ---" % format(hesstime,'.2f'))
                    # Obtain the gradient, Hessian trajectories and the loss value
                    weight_grad = chainRule_gradient(epsilon, gmin, tunable_para)
                    weight_hess = chainRule_hessian(epsilon, gmin, tunable_para)
                    print('start to calculate backpropagation')
                    start_time = TM.time()
                    neural_grad, neural_hess = uavNet.Backward(z5_sym,nn_para)
                    backproptime = (TM.time() - start_time)
                    print("backproptime=--- %s s ---" % format(backproptime,'.2f'))
                    start_time = TM.time()
                    Grad_dldn, Hess_dldn, loss = uavNMHE.ChainRule_2nd(ground_truth, xmhe_traj_curr, X_opt, HpX_opt, weight_grad, weight_hess, neural_grad, neural_hess)
                    chainruletime = (TM.time() - start_time)
                    print("chainruletime=--- %s s ---" % format(chainruletime,'.2f'))
                    # Solve the trust-region system (TRS) to obtain the candidate weighting change (wc)
                    print('start to solve the large TR subproblem')
                    start_time = TM.time()
                    wc_opt, TRS_opt = uavTRS.TRS_solver_Eigen(Grad_dldn, Hess_dldn, radius)
                    TRStime = (TM.time() - start_time)
                    print("TRStime=--- %s s ---" % format(TRStime,'.2f'))
                    # Update the network weightings
                    nn_para_updated = nn_para + wc_opt
                    z5_updated, z5_sym_updated = uavNet.Forward(measurement,nn_para_updated)
                    tunable_para_updated = np.reshape(z5_updated,(1,D_out))
                    P0, gamma_r, gamma_q, R_N, Q_N1, weight_para_updated = SetPara(epsilon, gmin, tunable_para_updated)
                    opt_sol_updated     = uavMHE.MHEsolver(Y, x_hat, xmhe_traj, noise_traj, weight_para_updated, time_index)
                    xmhe_traj_updated   = opt_sol_updated['state_traj_opt']
                    loss_updated        = uavNMHE.loss_horizon(xmhe_traj_updated,ground_truth,horizon,time_index)
                    # Obtain the ratio between the actual and predicted loss changes
                    ratio  = uavTRS.Get_ratio(loss,loss_updated,TRS_opt)
                    if n_ep>0:
                        # Update the trust-region radius
                        radius = uavTRS.TRS_radius(wc_opt,ratio,radius,t2,t3,chi1,chi2,upperb)
                    print('Trained=', n_ep, 'sample=', time_index, 'TRS_radius=',radius,'TRS_ratio=',ratio)
                    if ratio>t1 and n_ep >0:
                        # Update the weightings using the projected TRS solution
                        nn_para = np.reshape(nn_para_updated,(1,D_h*2+D_out+D_in*D_h+D_h**2+D_h*D_out))
                    # Update the MHE estimate trajectory
                    xmhe_traj  = xmhe_traj_curr
                    noise_traj = noise_traj_curr
                np.save('trained_data/nn_para_TR_NeuroMHE_'+str(key2),nn_para)
                # Sum the loss
                loss = np.reshape(loss, (1))
                sum_loss += loss
               
                it += 1
            print('it=',it)
        
        mean_loss = sum_loss / it
        mean_time = format(sum_time / (it-horizon-1),'.2f')
        print('Hess_cputime--- %s ms ---=',mean_time)
        # np.save('Hessian_cputime_horizon/hess_runtime_N='+str(key3),mean_time)
        if n_ep == 0:
            eps = mean_loss/10
        Mean_Loss += [mean_loss]
        if n_ep > 1: # train at least 1 episodes
            delta_cost = abs(mean_loss - mean_loss0)
            
        mean_loss0 = mean_loss
        print('learned', n_ep, 'mean_loss=', mean_loss)
        n_ep += 1
        if not os.path.exists("trained_data"):
            os.makedirs("trained_data")
        
        np.save('trained_data/Mean_loss_TR_NeuroMHE_'+str(key2), Mean_Loss)
        np.save('trained_data/Trained episodes_TR_NeuroMHE_'+str(key2), Trained_episode)
     
    # Loss function
    if n_ep>1:
        trainingtime = (TM.time() - start_time0)
        np.save('trained_data/trainingtime_tr_neuromhe_'+str(key2),trainingtime)
        print("trainingtime=--- %s s ---" % format(trainingtime,'.2f'))
    Dim_mean_loss = np.size(Mean_Loss)
    Trained = []
    for i in range(Dim_mean_loss):
        Trained += [i]
    
    plt.figure(1)
    plt.plot(Trained, Mean_Loss, linewidth=1.5,marker='o')
    plt.xlabel('Number of episodes')
    plt.ylabel('Mean loss')
    plt.grid()
    plt.savefig('trained_data/mean_loss_train_TR_NeuroMHE_'+str(key2)+'.png',dpi=600)
    plt.show()



"""---------------------------------Evaluation process-----------------------------"""
def Evaluate(key):
    
    nn_para = np.load('trained_data/nn_para_TR_NeuroMHE_1.npy')
    # Sampling index (sample over every n data points from the real dataset)
    n =1
    # Time sequence list
    Time = []
    # Ground truth sequence list
    Gt_fx, Gt_fy, Gt_fz, Gt_fxy, Gt_f, Bemnn_fx, Bemnn_fy, Bemnn_fxy, Bemnn_fz, Bemnn_f = [], [], [], [], [], [], [], [], [], []
    Gt_tx, Gt_ty, Gt_tz, Gt_txy, Gt_t, Bemnn_tx, Bemnn_ty, Bemnn_txy, Bemnn_tz, Bemnn_t = [], [], [], [], [], [], [], [], [], []
    E_BEM_fx, E_BEM_fy, E_BEM_fz, E_BEM_tx, E_BEM_ty, E_BEM_tz = [], [], [], [], [], []
    E_BEM_fxy, E_BEM_f, E_BEM_txy, E_BEM_t = [], [], [], []
    vx, vy, vz, wx, wy, wz, = [], [], [], [], [], [] 
    # Estimation sequence list
    vx_mhe, vy_mhe, vz_mhe, wx_mhe, wy_mhe, wz_mhe = [], [], [], [], [], []
    fx_mhe, fy_mhe, fz_mhe, fxy_mhe, f_mhe = [], [], [], [], []
    tx_mhe, ty_mhe, tz_mhe, txy_mhe, t_mhe = [], [], [], [], []
    E_MHE_fx, E_MHE_fy, E_MHE_fz, E_MHE_tx, E_MHE_ty, E_MHE_tz = [], [], [], [], [], []
    E_MHE_fxy, E_MHE_f, E_MHE_txy, E_MHE_t = [], [], [], []
    E_des_fx, E_des_fy, E_des_fz, E_des_tx, E_des_ty, E_des_tz = [], [], [], [], [], []
    E_des_fxy, E_des_f, E_des_txy, E_des_t = [], [], [], []
    # Estimation set
    evaluate_set = {'a':"processed_data/merged_2021-02-18-13-44-23_seg_2.csv",
                    'b':"processed_data/merged_2021-02-18-16-53-35_seg_2.csv",
                    'c':"processed_data/merged_2021-02-18-17-03-20_seg_2.csv",
                    'd':"processed_data/merged_2021-02-18-17-19-08_seg_2.csv",
                    'e':"processed_data/merged_2021-02-18-17-26-00_seg_1.csv",
                    'f':"processed_data/merged_2021-02-18-18-08-45_seg_1.csv",
                    'g':"processed_data/merged_2021-02-23-10-48-03_seg_2.csv",
                    'h':"processed_data/merged_2021-02-23-11-41-38_seg_3.csv",
                    'i':"processed_data/merged_2021-02-23-14-21-48_seg_3.csv",
                    'j':"processed_data/merged_2021-02-23-17-27-24_seg_2.csv",
                    'k':"processed_data/merged_2021-02-23-19-45-06_seg_2.csv",
                    'l':"processed_data/merged_2021-02-23-22-26-25_seg_2.csv",
                    'm':"processed_data/merged_2021-02-23-22-54-17_seg_1.csv"
    } 
    # Bem+nn set for comparison (force and torque prediction results of NeuroBEM)
    bem_nn_set   = {'a':"bem+nn/bem+nn_2021-02-18-13-44-23_seg_2.csv",
                    'b':"bem+nn/bem+nn_2021-02-18-16-53-35_seg_2.csv",
                    'c':"bem+nn/bem+nn_2021-02-18-17-03-20_seg_2.csv",
                    'd':"bem+nn/bem+nn_2021-02-18-17-19-08_seg_2.csv",
                    'e':"bem+nn/bem+nn_2021-02-18-17-26-00_seg_1.csv",
                    'f':"bem+nn/bem+nn_2021-02-18-18-08-45_seg_1.csv",
                    'g':"bem+nn/bem+nn_2021-02-23-10-48-03_seg_2.csv",
                    'h':"bem+nn/bem+nn_2021-02-23-11-41-38_seg_3.csv",
                    'i':"bem+nn/bem+nn_2021-02-23-14-21-48_seg_3.csv",
                    'j':"bem+nn/bem+nn_2021-02-23-17-27-24_seg_2.csv",
                    'k':"bem+nn/bem+nn_2021-02-23-19-45-06_seg_2.csv",
                    'l':"bem+nn/bem+nn_2021-02-23-22-26-25_seg_2.csv",
                    'm':"bem+nn/bem+nn_2021-02-23-22-54-17_seg_1.csv"
    }
    # Sum of loss
    sum_loss = 0.0
    # Measurement list
    Y = []
    # Quaternion list
    q_seq = []
    # Ground_truth list 
    ground_truth = []
    # Tunable parameter sequence list
    gamma1       = []
    gamma2       = []
    Tunable_para = []
    # MHE solver runtime
    MHE_runtime = []
    # noise list
    noise_traj = np.zeros((1,6))
    # print("==========================================================================================================")
    # print("Please choose which trajectory to evaluate")
    # print("'a': 3d circle 1,        Vmean=4.6169, Vmax=9.6930, Tmean=0.0070, Tmax=0.2281, Fmean=10.3181, Fmax=18.5641")
    # print("'b': linear oscillation, Vmean=5.1243, Vmax=15.0917,Tmean=0.0236, Tmax=0.4653, Fmean=12.0337, Fmax=34.5543")
    # print("'c': lemniscate 1,       Vmean=2.5747, Vmax=6.5158, Tmean=0.0059, Tmax=0.0357, Fmean=8.1929,  Fmax=10.8099")
    # print("'d': race track 1,       Vmean=5.8450, Vmax=11.2642,Tmean=0.0217, Tmax=0.4807, Fmean=11.0304, Fmax=16.4014")
    # print("'e': race track 2,       Vmean=6.9813, Vmax=14.2610,Tmean=0.0365, Tmax=0.4669, Fmean=14.6680, Fmax=30.0297")
    # print("'f': 3d circle 2,        Vmean=5.9991, Vmax=11.8102,Tmean=0.0087, Tmax=0.1109, Fmean=13.7672, Fmax=28.6176")
    # print("'g': lemniscate 2,       Vmean=1.6838, Vmax=3.4546, Tmean=0.0043, Tmax=0.0138, Fmean=7.6071,  Fmax=7.8332")
    # print("'h': melon 1,            Vmean=3.4488, Vmax=6.8019, Tmean=0.0074, Tmax=0.0883, Fmean=8.8032,  Fmax=14.4970")
    # print("'i': lemniscate 3,       Vmean=6.7462, Vmax=13.6089,Tmean=0.0290, Tmax=0.4830, Fmean=13.9946, Fmax=24.2318")
    # print("'j': lemniscate 4,       Vmean=9.1848, Vmax=17.7242,Tmean=0.0483, Tmax=0.5224, Fmean=18.5576, Fmax=36.2017")
    # print("'k': melon 2,            Vmean=7.0817, Vmax=12.1497,Tmean=0.0124, Tmax=0.1967, Fmean=17.2943, Fmax=34.4185")
    # print("'l': random point,       Vmean=2.5488, Vmax=8.8238, Tmean=0.0292, Tmax=0.6211, Fmean=9.1584,  Fmax=29.0121")
    # print("'m': ellipse,            Vmean=9.4713, Vmax=16.5371,Tmean=0.0117, Tmax=0.0962, Fmean=19.4827, Fmax=35.0123")

    # key = input("enter 'a', or 'b',... without the quotation mark:")
    # print("==========================================================================================================")
    dataEva = pd.read_csv(evaluate_set[key])
    dataBem = pd.read_csv(bem_nn_set[key],header=None, names=['t', 'ang acc x', 'ang acc y', 'ang acc z', \
                  'ang vel x', 'ang vel y', 'ang vel z', \
                  'quat x', 'quat y', 'quat z', 'quat w', \
                  'acc x', 'acc y', 'acc z', \
                  'vel x', 'vel y', 'vel z', \
                  'pos x', 'pos y', 'pos z', \
                  'mot 1', 'mot 2', 'mot 3', 'mot 4', \
                  'dmot 1', 'dmot 2', 'dmot 3', 'dmot 4', 'vbat', \
                  'model_fx', 'model_fy', 'model_fz', \
                  'model_tx', 'model_ty', 'model_tz', \
                  'residual_fx', 'residual_fy', 'residual_fz', \
                  'residual_tx', 'residual_ty', 'residual_tz'])
    dataframe = pd.DataFrame(dataEva)
    dataframe_bem = pd.DataFrame(dataBem)
    time_seq = dataframe['t']
    N_ev = int(time_seq.size/n)
    # Obtain the sequences of the state, acc, and motor speed from the dataframe
    angaccx_seq, angaccy_seq, angaccz_seq = dataframe['ang acc x'], dataframe['ang acc y'], dataframe['ang acc z']
    angvelx_seq, angvely_seq, angvelz_seq = dataframe['ang vel x'], dataframe['ang vel y'], dataframe['ang vel z']
    qx_seq, qy_seq, qz_seq, qw_seq = dataframe['quat x'], dataframe['quat y'], dataframe['quat z'], dataframe['quat w']
    
    accx_seq, accy_seq, accz_seq = dataframe['acc x'], dataframe['acc y'], dataframe['acc z']
    velx_seq, vely_seq, velz_seq = dataframe['vel x'], dataframe['vel y'], dataframe['vel z']
    fx_bemnn, fy_bemnn, fz_bemnn = dataframe_bem['model_fx'], dataframe_bem['model_fy'], dataframe_bem['model_fz']
    tx_bemnn, ty_bemnn, tz_bemnn = dataframe_bem['model_tx'], dataframe_bem['model_ty'], dataframe_bem['model_tz']
    res_fx, res_fy, res_fz       = dataframe_bem['residual_fx'], dataframe_bem['residual_fy'], dataframe_bem['residual_fz']
    # Initial guess 
    n_start = 0
    q0      = np.array([[qw_seq[n_start], qx_seq[n_start], qy_seq[n_start], qz_seq[n_start]]]).T
    v_B0    = np.array([[velx_seq[n_start], vely_seq[n_start], velz_seq[n_start]]]).T
    w_B0    = np.array([[angvelx_seq[n_start], angvely_seq[n_start], angvelz_seq[n_start]]]).T # in evaluation, the initial guess of the angular rate is set to be the measurement
    R_B0    = Quaternion2Rotation(q0)  
    v_I0    = np.matmul(R_B0, v_B0) # in evaluation, the initial guess of the linear velocity is set to be the measurement
    df_B0   = np.array([[0,0, Sys_para[0]*9.81]]).T # in evaluation, we set the initial guess of the force to be the gravity vector as the quadrotor maneuver starts from a near hovering state
    dt_B0   = np.zeros((3,1)) # in evaluation, we set the initial guess of the torque to be zero as the quadrotor maneuver starts from a near hovering state
    x_hat0  = np.vstack((v_I0,df_B0, w_B0, dt_B0))
    x_hat = x_hat0
    xmhe_traj = x_hat.T
    
    for j in range(n_start,N_ev):
        # Take the measurements from the dataset
        Time   += [time_seq[n*j]]
        v_B     = np.array([[velx_seq[n*j], vely_seq[n*j], velz_seq[n*j]]]).T
        q       = np.array([[qw_seq[n*j], qx_seq[n*j], qy_seq[n*j], qz_seq[n*j]]]).T
        w_B     = np.array([[angvelx_seq[n*j], angvely_seq[n*j], angvelz_seq[n*j]]]).T
        R_B     = Quaternion2Rotation(q)
        v_I     = np.matmul(R_B, v_B)
        measurement = np.vstack((v_I, w_B)) 
        Y      += [measurement]
        q_seq  += [q]

        # Take the output of the neural network model
        z5, z5_sym = uavNet.Forward(measurement,nn_para)
        tunable_para = np.reshape(z5,(1,D_out))
        P0, gamma_r, gamma_q, R_N, Q_N1, weight_para = SetPara(epsilon, gmin, tunable_para)
        gamma1 += [gamma_r]
        gamma2 += [gamma_q]
        print('sample=', j, 'p1=', P0[0, 0], 'gamma1=', gamma_r,'gamma2=', gamma_q,'r1=', R_N[0, 0], 'r2=', R_N[1, 1],'r3=', R_N[2, 2])
        print('sample=', j, 'q1=', Q_N1[0, 0], 'q2=', Q_N1[1, 1], 'q3=', Q_N1[2, 2])

        # Solve an MHE in the forward path
        time_index   = j-n_start
        start_time   = TM.time()
        opt_sol      = uavMHE.MHEsolver(Y, x_hat, xmhe_traj, noise_traj, weight_para, time_index)
        runtime      = (TM.time() - start_time)*1000
        print("--- %s ms ---" % runtime)
        MHE_runtime += [runtime]
        xmhe_traj    = opt_sol['state_traj_opt']
        noise_traj   = opt_sol['noise_traj_opt']
            
        if time_index>(horizon):
            for ix in range(len(x_hat)):
                x_hat[ix,0] = xmhe_traj[1, ix]

        # MHE estimates
        df_Imh   = np.transpose(xmhe_traj[-1, 3:6])
        df_Imh   = np.reshape(df_Imh, (3, 1))
        df_Bmh   = np.reshape(np.matmul(R_B.T,df_Imh),(3,1))
        dt_Bmh   = np.transpose(xmhe_traj[-1, 9:12])
        dt_Bmh   = np.reshape(dt_Bmh, (3, 1))
        wrench_mhe = np.vstack((df_Bmh, dt_Bmh))
        fx_mhe  += [wrench_mhe[0,0]]
        fy_mhe  += [wrench_mhe[1,0]]
        fz_mhe  += [wrench_mhe[2,0]]
        fxy_mhe += [np.sqrt(wrench_mhe[0,0]**2+wrench_mhe[1,0]**2)]
        f_mhe   += [np.sqrt(wrench_mhe[0,0]**2+wrench_mhe[1,0]**2+wrench_mhe[2,0]**2)]
        tx_mhe  += [wrench_mhe[3,0]]
        ty_mhe  += [wrench_mhe[4,0]]
        tz_mhe  += [wrench_mhe[5,0]]
        txy_mhe += [np.sqrt(wrench_mhe[3,0]**2+wrench_mhe[4,0]**2)]
        t_mhe   += [np.sqrt(wrench_mhe[3,0]**2+wrench_mhe[4,0]**2+wrench_mhe[5,0]**2)]
        vx_mhe  += [xmhe_traj[-1,0]]
        vy_mhe  += [xmhe_traj[-1,1]]
        vz_mhe  += [xmhe_traj[-1,2]]
        wx_mhe  += [xmhe_traj[-1,6]]
        wy_mhe  += [xmhe_traj[-1,7]]
        wz_mhe  += [xmhe_traj[-1,8]]

        # NeuroBEM prediction
        Bemnn_Fxb, Bemnn_Fyb, Bemnn_Fzb = fx_bemnn[n*j], fy_bemnn[n*j], fz_bemnn[n*j]
        Bemnn_Tx, Bemnn_Ty, Bemnn_Tz = tx_bemnn[n*j], ty_bemnn[n*j], tz_bemnn[n*j]
        Bemnn_Fb   = np.vstack((Bemnn_Fxb,Bemnn_Fyb,Bemnn_Fzb))
        # Bemnn_FI   = np.reshape(np.matmul(R_B,Bemnn_Fb),(3,1)) # transformed to the world frame
        Bemnn_Fx, Bemnn_Fy, Bemnn_Fz = Bemnn_Fb[0,0], Bemnn_Fb[1,0], Bemnn_Fb[2,0]
        Bemnn_fx  += [Bemnn_Fx]
        Bemnn_fy  += [Bemnn_Fy]
        Bemnn_fxy += [np.sqrt(Bemnn_Fx**2+Bemnn_Fy**2)]
        Bemnn_fz  += [Bemnn_Fz]
        Bemnn_f   += [np.sqrt(Bemnn_Fx**2+Bemnn_Fy**2+Bemnn_Fz**2)]
        Bemnn_tx  += [Bemnn_Tx]
        Bemnn_ty  += [Bemnn_Ty]
        Bemnn_txy += [np.sqrt(Bemnn_Tx**2+Bemnn_Ty**2)]
        Bemnn_tz  += [Bemnn_Tz]
        Bemnn_t   += [np.sqrt(Bemnn_Tx**2+Bemnn_Ty**2+Bemnn_Tz**2)] 
        
        # Ground truth list
        acc = np.array([[accx_seq[n*j], accy_seq[n*j], accz_seq[n*j], angaccx_seq[n*j], angaccy_seq[n*j], angaccz_seq[n*j]]]).T
        df_t, dt_t = GroundTruth(w_B, acc, uav.m, uav.J)
        dfI_t      = np.reshape(np.matmul(R_B,df_t),(3,1)) # transformed to the world frame
        state_t       = np.vstack((dfI_t, dt_t))
        ground_truth += [state_t]
        loss_track = uavNMHE.loss_horizon(xmhe_traj, ground_truth, horizon,time_index)
        print('sample=', j, 'loss=', loss_track)
        Gt_fx  += [df_t[0,0]]
        Gt_fy  += [df_t[1,0]]
        Gt_fz  += [df_t[2,0]]
        Gt_fxy += [np.sqrt(df_t[0,0]**2+df_t[1,0]**2)]
        Gt_f   += [np.sqrt(df_t[0,0]**2+df_t[1,0]**2+df_t[2,0]**2)]
        Gt_tx  += [dt_t[0,0]]
        Gt_ty  += [dt_t[1,0]]
        Gt_tz  += [dt_t[2,0]]
        Gt_txy += [np.sqrt(dt_t[0,0]**2+dt_t[1,0]**2)]
        Gt_t   += [np.sqrt(dt_t[0,0]**2+dt_t[1,0]**2+dt_t[2,0]**2)]
        vx     += [v_I[0, 0]]
        vy     += [v_I[1, 0]]
        vz     += [v_I[2, 0]]
        wx     += [w_B[0, 0]]
        wy     += [w_B[1, 0]]
        wz     += [w_B[2, 0]]
        print('sample=', j, 'Dist_x=', dt_t[0, 0], 'dt_Bmh_x=', dt_Bmh[0, 0], 'Dist_y=', dt_t[1, 0], 'dt_Bmh_y=', dt_Bmh[1, 0], 'Dist_z=', dt_t[2, 0], 'dt_Bmh_z=', dt_Bmh[2, 0])
        print('sample=', j, 'Dis_x=', df_t[0, 0], 'df_Bmh_x=', df_Bmh[0, 0], 'Dis_y=', df_t[1, 0], 'df_Bmh_y=', df_Bmh[1, 0], 'Dis_z=', df_t[2, 0], 'df_Bmh_z=', df_Bmh[2, 0])
        # Prediction/estimation error
        E_BEM_fx += [df_t[0,0] - Bemnn_Fx]
        E_BEM_fy += [df_t[1,0] - Bemnn_Fy]
        E_BEM_fz += [df_t[2,0] - Bemnn_Fz]
        E_BEM_tx += [dt_t[0,0] - Bemnn_Tx]
        E_BEM_ty += [dt_t[1,0] - Bemnn_Ty]
        E_BEM_tz += [dt_t[2,0] - Bemnn_Tz]
        E_MHE_fx += [df_t[0,0] - wrench_mhe[0,0]]
        E_MHE_fy += [df_t[1,0] - wrench_mhe[1,0]]
        E_MHE_fz += [df_t[2,0] - wrench_mhe[2,0]]
        E_MHE_tx += [dt_t[0,0] - wrench_mhe[3,0]]
        E_MHE_ty += [dt_t[1,0] - wrench_mhe[4,0]]
        E_MHE_tz += [dt_t[2,0] - wrench_mhe[5,0]]
        E_des_fx += [0]
        E_des_fy += [0]
        E_des_fz += [0]
        E_des_fxy += [0]
        E_des_f  += [0]
        E_des_tx += [0]
        E_des_ty += [0]
        E_des_tz += [0]
        E_des_txy += [0]
        E_des_t  += [0]
        # vector-based error
        E_BEM_fxy += [np.sqrt(E_BEM_fx[j-n_start]**2+E_BEM_fy[j-n_start]**2)]
        E_BEM_f  += [np.sqrt(E_BEM_fx[j-n_start]**2+E_BEM_fy[j-n_start]**2+E_BEM_fz[j-n_start]**2)]
        E_BEM_txy += [np.sqrt(E_BEM_tx[j-n_start]**2+E_BEM_ty[j-n_start]**2)]
        E_BEM_t  += [np.sqrt(E_BEM_tx[j-n_start]**2+E_BEM_ty[j-n_start]**2+E_BEM_tz[j-n_start]**2)]
        E_MHE_fxy += [np.sqrt(E_MHE_fx[j-n_start]**2+E_MHE_fy[j-n_start]**2)]
        E_MHE_f  += [np.sqrt(E_MHE_fx[j-n_start]**2+E_MHE_fy[j-n_start]**2+E_MHE_fz[j-n_start]**2)]
        E_MHE_txy += [np.sqrt(E_MHE_tx[j-n_start]**2+E_MHE_ty[j-n_start]**2)]
        E_MHE_t  += [np.sqrt(E_MHE_tx[j-n_start]**2+E_MHE_ty[j-n_start]**2+E_MHE_tz[j-n_start]**2)]

        # Comparison with the residual force/torque in the dataset
        print('sample=', j,'E_BEM_fx=',format(E_BEM_fx[j-n_start],'.3f'),'res_fx=',format(res_fx[n*j],'.3f'),'error_x=',format(E_BEM_fx[j-n_start]-res_fx[n*j],'.3f'))
        print('sample=', j,'E_BEM_fy=',format(E_BEM_fy[j-n_start],'.3f'),'res_fy=',format(res_fy[n*j],'.3f'),'error_y=',format(E_BEM_fy[j-n_start]-res_fy[n*j],'.3f'))
        print('sample=', j,'E_BEM_fz=',format(E_BEM_fz[j-n_start],'.3f'),'res_fz=',format(res_fz[n*j],'.3f'),'error_z=',format(E_BEM_fz[j-n_start]-res_fz[n*j],'.3f'))
        # Sum the loss
        loss_track = np.reshape(loss_track, (1))
        sum_loss += loss_track
    # cputime = np.zeros((len(MHE_runtime),1))
    # for k in range(len(MHE_runtime)):
    #     cputime[k,0] = MHE_runtime[k]
    # print('cputime=',statistics.median(cputime))
    # np.save('trained_data/cpu_MHE_mediantime_N=10',statistics.median(cputime))
    mean_loss_ev = sum_loss/ N_ev
    print('mean_loss_ev=', mean_loss_ev)

    if not os.path.exists("trained_data"):
        os.makedirs("trained_data")
    
    np.save('trained_data/Time_'+str(key), Time)
    np.save('trained_data/Tunable_para_'+str(key), Tunable_para)
    np.save('trained_data/Gamma1_'+str(key), gamma1)
    np.save('trained_data/Gamma2_'+str(key), gamma2)
    np.save('trained_data/Gt_fx_'+str(key), Gt_fx)
    np.save('trained_data/Gt_fy_'+str(key), Gt_fy)
    np.save('trained_data/Gt_fz_'+str(key), Gt_fz)
    np.save('trained_data/Gt_fxy_'+str(key), Gt_fxy)
    np.save('trained_data/Gt_f_'+str(key), Gt_f)
    np.save('trained_data/Bemnn_fx_'+str(key), Bemnn_fx)
    np.save('trained_data/Bemnn_fy_'+str(key), Bemnn_fy)
    np.save('trained_data/Bemnn_fxy_'+str(key), Bemnn_fxy)
    np.save('trained_data/Bemnn_fz_'+str(key), Bemnn_fz)
    np.save('trained_data/Bemnn_f_'+str(key), Bemnn_f)
    np.save('trained_data/Bemnn_tx_'+str(key), Bemnn_tx)
    np.save('trained_data/Bemnn_ty_'+str(key), Bemnn_ty)
    np.save('trained_data/Bemnn_txy_'+str(key), Bemnn_txy)
    np.save('trained_data/Bemnn_tz_'+str(key), Bemnn_tz)
    np.save('trained_data/Bemnn_t_'+str(key), Bemnn_t)
    np.save('trained_data/Gt_tx_'+str(key), Gt_tx)
    np.save('trained_data/Gt_ty_'+str(key), Gt_ty)
    np.save('trained_data/Gt_txy_'+str(key), Gt_txy)
    np.save('trained_data/Gt_t_'+str(key), Gt_t)
    np.save('trained_data/Gt_tz_'+str(key), Gt_tz)
    np.save('trained_data/fx_mhe_'+str(key), fx_mhe)
    np.save('trained_data/fy_mhe_'+str(key), fy_mhe)
    np.save('trained_data/fxy_mhe_'+str(key), fxy_mhe)
    np.save('trained_data/f_mhe_'+str(key), f_mhe)
    np.save('trained_data/fz_mhe_'+str(key), fz_mhe)
    np.save('trained_data/tx_mhe_'+str(key), tx_mhe)
    np.save('trained_data/ty_mhe_'+str(key), ty_mhe)
    np.save('trained_data/txy_mhe_'+str(key),txy_mhe)
    np.save('trained_data/t_mhe_'+str(key), t_mhe)
    np.save('trained_data/tz_mhe_'+str(key), tz_mhe)
    np.save('trained_data/error_fxy_mhe_v_'+str(key), E_MHE_fxy)
    np.save('trained_data/error_f_mhe_v_'+str(key), E_MHE_f)
    np.save('trained_data/error_txy_mhe_v_'+str(key), E_MHE_txy)
    np.save('trained_data/error_t_mhe_v_'+str(key), E_MHE_t)
    np.save('trained_data/error_fxy_bem_v_'+str(key), E_BEM_fxy)
    np.save('trained_data/error_f_bem_v_'+str(key), E_BEM_f)
    np.save('trained_data/error_txy_bem_v_'+str(key), E_BEM_txy)
    np.save('trained_data/error_t_bem_v_'+str(key), E_BEM_t)

    """
    Plot figures
    """
    # compute RMSE
    # body frame
    rmse_fx = format(mean_squared_error(fx_mhe, Gt_fx, squared=False),'.3f')
    rmse_fy = format(mean_squared_error(fy_mhe, Gt_fy, squared=False),'.3f')
    rmse_fz = format(mean_squared_error(fz_mhe, Gt_fz, squared=False),'.3f')
    rmse_tx = format(mean_squared_error(tx_mhe, Gt_tx, squared=False),'.3f')
    rmse_ty = format(mean_squared_error(ty_mhe, Gt_ty, squared=False),'.3f')
    rmse_tz = format(mean_squared_error(tz_mhe, Gt_tz, squared=False),'.3f')
    rmse_fx_bemnn  = format(mean_squared_error(Bemnn_fx, Gt_fx, squared=False),'.3f')
    rmse_fy_bemnn  = format(mean_squared_error(Bemnn_fy, Gt_fy, squared=False),'.3f')
    rmse_fz_bemnn  = format(mean_squared_error(Bemnn_fz, Gt_fz, squared=False),'.3f')
    rmse_tx_bemnn  = format(mean_squared_error(Bemnn_tx, Gt_tx, squared=False),'.3f')
    rmse_ty_bemnn  = format(mean_squared_error(Bemnn_ty, Gt_ty, squared=False),'.3f')
    rmse_tz_bemnn  = format(mean_squared_error(Bemnn_tz, Gt_tz, squared=False),'.3f')
    # scalar-based error
    rmse_fxy       = format(mean_squared_error(fxy_mhe, Gt_fxy, squared=False),'.3f')
    rmse_txy       = format(mean_squared_error(txy_mhe, Gt_txy, squared=False),'.3f')
    rmse_f         = format(mean_squared_error(f_mhe, Gt_f, squared=False),'.3f')
    rmse_t         = format(mean_squared_error(t_mhe, Gt_t, squared=False),'.3f')
    rmse_fxy_bemnn = format(mean_squared_error(Bemnn_fxy, Gt_fxy, squared=False),'.3f')
    rmse_txy_bemnn = format(mean_squared_error(Bemnn_txy, Gt_txy, squared=False),'.3f')
    rmse_f_bemnn   = format(mean_squared_error(Bemnn_f, Gt_f, squared=False),'.3f')
    rmse_t_bemnn   = format(mean_squared_error(Bemnn_t, Gt_t, squared=False),'.3f')
    # vector-based error
    rmse_fxy_v     = format(mean_squared_error(E_MHE_fxy, E_des_fxy, squared=False),'.3f')
    rmse_f_v       = format(mean_squared_error(E_MHE_f, E_des_f, squared=False),'.3f')
    rmse_txy_v     = format(mean_squared_error(E_MHE_txy, E_des_txy, squared=False),'.3f')
    rmse_t_v       = format(mean_squared_error(E_MHE_t, E_des_t, squared=False),'.3f')
    rmse_fxy_v_bem = format(mean_squared_error(E_BEM_fxy, E_des_fxy, squared=False),'.3f')
    rmse_f_v_bem   = format(mean_squared_error(E_BEM_f, E_des_f, squared=False),'.3f')
    rmse_txy_v_bem = format(mean_squared_error(E_BEM_txy, E_des_txy, squared=False),'.3f')
    rmse_t_v_bem   = format(mean_squared_error(E_BEM_t, E_des_t, squared=False),'.3f')

    print('rmse_fx_bemnn=',rmse_fx_bemnn,'rmse_fy_bemnn=',rmse_fy_bemnn,'rmse_fz_bemnn=',rmse_fz_bemnn)
    print('rmse_fx=',rmse_fx,'rmse_fy=',rmse_fy,'rmse_fz=',rmse_fz)
    print('rmse_tx_bemnn=',rmse_tx_bemnn,'rmse_ty_bemnn=',rmse_ty_bemnn,'rmse_tz_bemnn=',rmse_tz_bemnn)
    print('rmse_tx=',rmse_tx,'rmse_ty=',rmse_ty,'rmse_tz=',rmse_tz)
    print("=====Scalar-based Error=====")
    print('rmse_fxy_bemnn=',rmse_fxy_bemnn,'rmse_txy_bemnn=',rmse_txy_bemnn)
    print('rmse_fxy=',rmse_fxy,'rmse_txy=',rmse_txy)
    print('rmse_f_bemnn=',rmse_f_bemnn,'rmse_t_bemnn=',rmse_t_bemnn)
    print('rmse_f=',rmse_f,'rmse_t=',rmse_t)
    print("=====Vector-based Error=====")
    print('rmse_fxy_v_bem=',rmse_fxy_v_bem,'rmse_txy_v_bem=',rmse_txy_v_bem)
    print('rmse_fxy_v=',rmse_fxy_v,'rmse_txy_v=',rmse_txy_v)
    print('rmse_f_v_bem=',rmse_f_v_bem,'rmse_t_v_bem=',rmse_t_v_bem)
    print('rmse_f_v=',rmse_f_v,'rmse_t_v=',rmse_t_v)


    # np.save('trained_data/rmse_fx_'+str(key), rmse_fx)
    # np.save('trained_data/rmse_fy_'+str(key), rmse_fy)
    # np.save('trained_data/rmse_fz_'+str(key), rmse_fz)
    # np.save('trained_data/rmse_tx_'+str(key), rmse_tx)
    # np.save('trained_data/rmse_ty_'+str(key), rmse_ty)
    # np.save('trained_data/rmse_tz_'+str(key), rmse_tz)
    # np.save('trained_data/rmse_fxy_'+str(key), rmse_fxy)
    # np.save('trained_data/rmse_txy_'+str(key), rmse_txy)
    # np.save('trained_data/rmse_f_'+str(key), rmse_f)
    # np.save('trained_data/rmse_t_'+str(key), rmse_t)
    # np.save('trained_data/rmse_fx_bemnn_'+str(key), rmse_fx_bemnn)
    # np.save('trained_data/rmse_fy_bemnn_'+str(key), rmse_fy_bemnn)
    # np.save('trained_data/rmse_fxy_bemnn_'+str(key), rmse_fxy_bemnn)
    # np.save('trained_data/rmse_fz_bemnn_'+str(key), rmse_fz_bemnn)
    # np.save('trained_data/rmse_tx_bemnn_'+str(key), rmse_tx_bemnn)
    # np.save('trained_data/rmse_ty_bemnn_'+str(key), rmse_ty_bemnn)
    # np.save('trained_data/rmse_txy_bemnn_'+str(key), rmse_txy_bemnn)
    # np.save('trained_data/rmse_tz_bemnn_'+str(key), rmse_tz_bemnn)
    # np.save('trained_data/rmse_f_bemnn_'+str(key), rmse_f_bemnn)
    # np.save('trained_data/rmse_t_bemnn_'+str(key), rmse_t_bemnn)
    # np.save('trained_data/rmse_fxy_v_'+str(key), rmse_fxy_v)
    # np.save('trained_data/rmse_f_v_'+str(key), rmse_f_v)
    # np.save('trained_data/rmse_txy_v_'+str(key), rmse_txy_v)
    # np.save('trained_data/rmse_t_v_'+str(key), rmse_t_v)
    # np.save('trained_data/rmse_fxy_v_bem_'+str(key), rmse_fxy_v_bem)
    # np.save('trained_data/rmse_f_v_bem_'+str(key), rmse_f_v_bem)
    # np.save('trained_data/rmse_txy_v_bem_'+str(key), rmse_txy_v_bem)
    # np.save('trained_data/rmse_t_v_bem_'+str(key), rmse_t_v_bem)

    if not os.path.exists("plots_testset"):
        os.makedirs("plots_testset")

    
    # Disturbance force
    plt.figure(2)
    plt.plot(Time, Gt_fxy, linewidth=1.5, linestyle='--')
    plt.plot(Time, fxy_mhe, linewidth=1)
    plt.plot(Time, Bemnn_fxy, linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Force xy in world frame [N]')
    plt.legend(['Ground truth', 'NeuroMHE', 'NeuroBEM'])
    plt.grid()
    plt.savefig('plots_testset/fxy_'+str(key)+'.png',dpi=800)
    plt.clf()
    # plt.show()
    plt.figure(3)
    plt.plot(Time, Gt_fz, linewidth=1.5, linestyle='--')
    plt.plot(Time, fz_mhe, linewidth=1)
    plt.plot(Time, Bemnn_fz, linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Force z in world frame [N]')
    plt.legend(['Ground truth', 'NeuroMHE', 'NeuroBEM'])
    plt.grid()
    plt.savefig('plots_testset/fz_'+str(key)+'.png',dpi=800)
    plt.clf()
    # plt.show()
    # Disturbance torque
    plt.figure(4)
    plt.plot(Time, Gt_txy, linewidth=1, linestyle='--')
    plt.plot(Time, txy_mhe, linewidth=0.5)
    plt.plot(Time, Bemnn_txy, linewidth=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Torque xy in body frame [Nm]')
    plt.legend(['Ground truth', 'NeuroMHE', 'NeuroBEM'])
    plt.grid()
    plt.savefig('plots_testset/torque_xy_'+str(key)+'.png',dpi=800)
    plt.clf()
    # plt.show()
    plt.figure(5)
    plt.plot(Time, Gt_tz, linewidth=1, linestyle='--')
    plt.plot(Time, tz_mhe, linewidth=0.5)
    plt.plot(Time, Bemnn_tz, linewidth=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Torque z in body frame [Nm]')
    plt.legend(['Ground truth', 'NeuroMHE', 'NeuroBEM'])
    plt.grid()
    plt.savefig('plots_testset/torque_z_'+str(key)+'.png',dpi=800)
    # plt.show()
    plt.figure(6)
    plt.plot(Time, gamma1, linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('gamma 1')
    plt.grid()
    plt.savefig('plots_testset/gamma1_'+str(key)+'.png',dpi=600)
    plt.clf()
    # plt.show()
    plt.figure(7)
    plt.plot(Time, gamma2, linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('gamma 2')
    plt.grid()
    plt.savefig('plots_testset/gamma2_'+str(key)+'.png',dpi=600)
    plt.clf()
    # plt.show()
    


"""---------------------------------Main function-----------------------------"""
print("=============================================")
print("Main code for training or evaluating NeuroMHE")
print("Please choose mode")
mode = input("enter 'train' or 'evaluate' without the quotation mark:")
print("=============================================")
if mode =="train":
    Train()
else:
    evaluation_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m']
    # evaluation_list = ['m']
    for key in evaluation_list:
        Evaluate(key)
