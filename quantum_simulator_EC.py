import numpy as np
from scipy.integrate import solve_ivp, complex_ode
import matplotlib.pyplot as plt
import sys
from termcolor import colored
from copy import copy, deepcopy
import time


pi = np.pi
ide = np.identity(2)
sig_x = np.array([[0.,1.],[1.,0.]])
sig_y = np.array([[0.,-1j],[1j,0.]])
sig_z = np.array([[1.,0.],[0.,-1.]])


class system(object):

    def __init__( self, T_dec=1200, T_single=1200, N=4, Delta_t=1, P=0.1 ):
        
        self.t = 0.
        self.Delta_t = Delta_t
        self.T_dec = T_dec
        self.T_single = T_single
        self.P = P
        self.N_qubits = N
        self.n = np.array([0,0,1],dtype='complex')
        self.hsdim = 2**self.N_qubits
        self.last_action = 0
        
        self.rho0_t0 = np.zeros( (self.hsdim, self.hsdim), dtype='complex' )
        self.drhox_t0 = np.zeros( (self.hsdim, self.hsdim), dtype='float' )
        self.drhoy_t0 = np.zeros( (self.hsdim, self.hsdim), dtype='complex' )
        self.drhoz_t0 = np.zeros( (self.hsdim, self.hsdim), dtype='float' )
        self.rho0 = np.zeros( (self.hsdim, self.hsdim), dtype='complex' )
        self.drhox = np.zeros( (self.hsdim, self.hsdim), dtype='float' )
        self.drhoy = np.zeros( (self.hsdim, self.hsdim), dtype='complex' )
        self.drhoz = np.zeros( (self.hsdim, self.hsdim), dtype='float' )
        
        """ 
        IMPORTANT: rhon only evaluated when needed
        """
        self.rhon = np.zeros( (self.hsdim, self.hsdim), dtype='complex' )
        
        self.initialise_rho_t0_mats()
        self.initialise_rho_mats()
        
        self.Sig_x = np.zeros( (self.hsdim, self.hsdim, self.N_qubits), dtype='float' )
        self.Sig_y = np.zeros( (self.hsdim, self.hsdim, self.N_qubits), dtype='complex' )
        self.Sig_z = np.zeros( (self.hsdim, self.hsdim, self.N_qubits), dtype='float' )
        self.initialise_sigmas()
        
        
        self.RQ_old = self.RQ()
        self.exp_RQ_old = self.RQ_old
        
        self.state_map = {}
        for i in range( self.hsdim ):
            
            st = np.zeros(self.hsdim)
            st[i] = 1
            
            st_small = []
            for j in range(self.N_qubits):
                z = st.dot(self.Sig_z[:,:,j]).dot(st.T)
                st_small.append(np.int( z ))
                
            self.state_map[i] =  st_small 
        
        #-- [None,0,1,2,3,4,5,6,7,8,9,10,11,0,1,2,3,0,1,2,3]
        self.actions = [None]
        for i in range(N*(N-1)):
            self.actions.append(i)
        for i in range(N):
            self.actions.append(i)
        for i in range(N):
            self.actions.append(i)
            
        self.cnot_pairs = []
        for i in range(N):
            for j in range(i+1,N):
                self.cnot_pairs.append([i,j])
                self.cnot_pairs.append([j,i])
            
        self.cnots = np.zeros( (self.hsdim, self.hsdim, N*(N-1)), dtype='float' )
        for k, qubits in enumerate( self.cnot_pairs ):
            self.cnots[:,:,k] = np.identity(self.hsdim)
            qb1 = qubits[0]
            qb2 = qubits[1]
            qubit2_pairs = []
            for i in range( self.hsdim ):
                
                if self.state_map[i][qb1] == 1:
                    state_i_qb2_flipped = self.state_map[i].copy()
                    state_i_qb2_flipped[qb2] *= -1
                    
                    for j in range( i+1, self.hsdim ):    
                        if self.state_map[j] == state_i_qb2_flipped:
                            qubit2_pairs.append([i,j])
                            
            for i,pair in enumerate(qubit2_pairs):
                self.cnots[pair[1],pair[1],k] = 0
                self.cnots[pair[0],pair[0],k] = 0
                self.cnots[pair[0],pair[1],k] = 1
                self.cnots[pair[1],pair[0],k] = 1 
                
        
        #== a dictionary to keep track of the meaning of each state 
        #== i.e. : [.,.,.,.,.,.,.,.] <-> [.,.,.] for a 3 qubit system
        
        self.Pz_up = np.zeros( (self.hsdim, self.hsdim, self.N_qubits), dtype='float' )
        self.Pz_down = np.zeros( (self.hsdim, self.hsdim, self.N_qubits), dtype='float' )
        for j in range(self.N_qubits):
            for i in range(self.hsdim):
                if self.state_map[i][j] == 1:
                    self.Pz_up[i,i,j] = 1
                elif self.state_map[i][j] == -1:
                    self.Pz_down[i,i,j] = 1
                else:
                    print("-- EROOR IN STATE MAP --")
        
        
    def initialise_sigmas(self):
        
        for i in range( self.N_qubits ):
            
            if i==0:
                mq1x = sig_x
                mq1y = sig_y
                mq1z = sig_z
            else:
                mq1x = ide
                mq1y = ide
                mq1z = ide
            
            for j in range( 1, self.N_qubits ):
                if j==i:
                    mqjx = sig_x
                    mqjy = sig_y
                    mqjz = sig_z
                else:
                    mqjx = ide
                    mqjy = ide
                    mqjz = ide
                    
                mq1x = np.kron(mq1x,mqjx)
                mq1y = np.kron(mq1y,mqjy)
                mq1z = np.kron(mq1z,mqjz)
            
            self.Sig_x[:,:,i] = mq1x
            self.Sig_y[:,:,i] = mq1y
            self.Sig_z[:,:,i] = mq1z
        
        
    def initialise_rho_t0_mats( self ):
        
        dm_other = self.make_dm([0,0,-1])
        rho0 = 0.5*( self.make_dm([0,0,1]) + self.make_dm([0,0,-1]) )
        drhox = 0.5*( self.make_dm([1,0,0]) - self.make_dm([-1,0,0]) )
        drhoy = 0.5*( self.make_dm([0,1,0]) - self.make_dm([0,-1,0]) )
        drhoz = 0.5*( self.make_dm([0,0,1]) - self.make_dm([0,0,-1]) )
                                   
        for i in range( 1, self.N_qubits ):
        
            rho0 = np.kron( rho0, dm_other )
            drhox = np.kron( drhox, dm_other )
            drhoy = np.kron( drhoy, dm_other )
            drhoz = np.kron( drhoz, dm_other )
        
        self.rho0_t0 = rho0 
        self.drhox_t0 = drhox 
        self.drhoy_t0 = drhoy 
        self.drhoz_t0 = drhoz 
        
    def eval_rhon( self ):
        
        n =self.n
        self.rhon = (self.rho0 + n[0]*self.drhox+ n[1]*self.drhoy+ n[2]*self.drhoz) \
            /(1+np.trace(n[0]*self.drhox+ n[1]*self.drhoy+ n[2]*self.drhoz))
        
    def set_initial_state( self, n ):
        
        n_normed = n / np.sqrt( np.dot(n,n) )
        self.n = n_normed
        self.eval_rhon()
        
    def initialise_rho_mats( self ):
        
        self.rho0 =  copy(self.rho0_t0)
        self.drhox = copy(self.drhox_t0)
        self.drhoy = copy(self.drhoy_t0)
        self.drhoz = copy(self.drhoz_t0)
        
        
    def initialise_all( self ):
        
        self.initialise_rho_mats()
        self.last_action = 0
        self.RQ_old = self.RQ()
        self.exp_RQ_old = self.RQ_old
        self.t = 0
        
        
    def generate_net_input_state( self, nb_vecs ):
        
        #-- density matrix input
        rho0_input = np.zeros( ( self.hsdim, nb_vecs) )
        rho1_input = np.zeros( ( self.hsdim, nb_vecs) )
        rho2_input = np.zeros( ( self.hsdim, nb_vecs) )
        rho3_input = np.zeros( ( self.hsdim, nb_vecs) )
        
        rho1 = copy(self.rho0) + copy(self.drhox)
        rho2 = copy(self.rho0) + copy(self.drhoy)
        rho3 = copy(self.rho0) + copy(self.drhoz)
        
        eig0, vecs0 = np.linalg.eig( self.rho0 )
        eig1, vecs1 = np.linalg.eig( rho1 )
        eig2, vecs2 = np.linalg.eig( rho2 )
        eig3, vecs3 = np.linalg.eig( rho3 )
        
        argsort0 = np.flip( np.argsort( eig0 ) )
        argsort1 = np.flip( np.argsort( eig1 ) )
        argsort2 = np.flip( np.argsort( eig2 ) )
        argsort3 = np.flip( np.argsort( eig3 ) )
        
        sq_eig_times_vecs0 = np.zeros_like(vecs0)
        sq_eig_times_vecs1 = np.zeros_like(vecs1)
        sq_eig_times_vecs2 = np.zeros_like(vecs2)
        sq_eig_times_vecs3 = np.zeros_like(vecs3)
        for i in range( self.hsdim ):
            sq_eig_times_vecs0[i] = np.sqrt(eig0[i]) * vecs0[:,i]
            sq_eig_times_vecs1[i] = np.sqrt(eig1[i]) * vecs1[:,i]
            sq_eig_times_vecs2[i] = np.sqrt(eig2[i]) * vecs2[:,i]
            sq_eig_times_vecs3[i] = np.sqrt(eig3[i]) * vecs3[:,i]
        
        N_1D = np.prod( rho0_input.shape )
        rho0_input = sq_eig_times_vecs0[:,argsort0[np.arange(0,nb_vecs)]].reshape( N_1D )
        rho1_input = sq_eig_times_vecs1[:,argsort1[np.arange(0,nb_vecs)]].reshape( N_1D )
        rho2_input = sq_eig_times_vecs2[:,argsort2[np.arange(0,nb_vecs)]].reshape( N_1D )
        rho3_input = sq_eig_times_vecs3[:,argsort3[np.arange(0,nb_vecs)]].reshape( N_1D )
        
        nn_input = np.append( np.real(rho0_input), np.imag(rho0_input) )
        nn_input = np.append( nn_input, np.real(rho1_input) )
        nn_input = np.append( nn_input, np.imag(rho1_input) )
        nn_input = np.append( nn_input, np.real(rho2_input) )
        nn_input = np.append( nn_input, np.imag(rho2_input) )
        nn_input = np.append( nn_input, np.real(rho3_input) )
        nn_input = np.append( nn_input, np.imag(rho3_input) )
        
        bools = []
        for i in range( self.N_qubits ):
            meas_outcome = np.real( np.trace( self.Pz_up[:,:,i].dot(self.drhox) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            meas_outcome = np.real( np.trace( self.Pz_down[:,:,i].dot(self.drhox) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            meas_outcome = np.real( np.trace( self.Pz_up[:,:,i].dot(self.drhoy) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            meas_outcome = np.real( np.trace( self.Pz_down[:,:,i].dot(self.drhoy) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            meas_outcome = np.real( np.trace( self.Pz_up[:,:,i].dot(self.drhoz) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            meas_outcome = np.real( np.trace( self.Pz_down[:,:,i].dot(self.drhoz) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            
        
        nn_input = np.append( nn_input, bools )
        nn_input = np.append( nn_input, self.last_action )
        
        return nn_input
        
    def generate_net_input_FULL( self, nb_vecs ):
        
        rho1 = copy(self.rho0) + copy(self.drhox)
        rho2 = copy(self.rho0) + copy(self.drhoy)
        rho3 = copy(self.rho0) + copy(self.drhoz)
        
        #-- density matrix input
        N_1D = np.prod( self.rho0.shape )
        rho0_input = self.rho0.reshape( N_1D )
        rho1_input = rho1.reshape( N_1D )
        rho2_input = rho2.reshape( N_1D )
        rho3_input = rho3.reshape( N_1D )
        
        nn_input = np.append( np.real(rho0_input), np.imag(rho0_input) )
        nn_input = np.append( nn_input, np.real(rho1_input) )
        nn_input = np.append( nn_input, np.imag(rho1_input) )
        nn_input = np.append( nn_input, np.real(rho2_input) )
        nn_input = np.append( nn_input, np.imag(rho2_input) )
        nn_input = np.append( nn_input, np.real(rho3_input) )
        nn_input = np.append( nn_input, np.imag(rho3_input) )
        
        bools = []
        for i in range( self.N_qubits ):
            meas_outcome = np.real( np.trace( self.Pz_up[:,:,i].dot(self.drhox) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            meas_outcome = np.real( np.trace( self.Pz_down[:,:,i].dot(self.drhox) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            meas_outcome = np.real( np.trace( self.Pz_up[:,:,i].dot(self.drhoy) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            meas_outcome = np.real( np.trace( self.Pz_down[:,:,i].dot(self.drhoy) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            meas_outcome = np.real( np.trace( self.Pz_up[:,:,i].dot(self.drhoz) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            meas_outcome = np.real( np.trace( self.Pz_down[:,:,i].dot(self.drhoz) ) )
            if np.abs(meas_outcome) < 1e-8: bools.append(0)
            else: bools.append(1)
            
        
        nn_input = np.append( nn_input, bools )
        nn_input = np.append( nn_input, self.last_action )
        
        return nn_input
        
    
    def make_dm( self, n ):
        
        den_mat = 0.5*( ide + n[0]*sig_x+n[1]*sig_y+n[2]*sig_z )
    
        return den_mat


    def bit_flip_RHS( self, t, dm_1D ):
        
        dm = dm_1D.reshape( self.hsdim, self.hsdim )
        dm_bf = np.zeros_like(dm)
        for i in range( self.N_qubits ):
            dm_bf += self.Sig_x[:,:,i].dot(dm).dot(self.Sig_x[:,:,i]) - dm
        
        dm_bf_1D = (1/self.T_dec)*dm_bf.reshape( np.prod(dm_bf.shape[:]) )
            
        return dm_bf_1D
    
    def time_evolve( self ):
    
        rho0_1D = self.rho0.reshape( self.hsdim**2 )
        drhox_1D = self.drhox.reshape( self.hsdim**2 )
        drhoy_1D = self.drhoy.reshape( self.hsdim**2 )
        drhoz_1D = self.drhoz.reshape( self.hsdim**2 )
        
        integrator = complex_ode( self.bit_flip_RHS )
        
        integrator.set_initial_value(rho0_1D, self.t)
        sol_rho0 = integrator.integrate( self.t+self.Delta_t )
        integrator.set_initial_value(drhox_1D, self.t)
        sol_drhox = integrator.integrate( self.t+self.Delta_t )
        integrator.set_initial_value(drhoy_1D, self.t)
        sol_drhoy = integrator.integrate( self.t+self.Delta_t )
        integrator.set_initial_value(drhoz_1D, self.t)
        sol_drhoz = integrator.integrate( self.t+self.Delta_t )
        
        self.rho0 = sol_rho0.reshape( self.hsdim, self.hsdim )
        self.drhox = sol_drhox.reshape( self.hsdim, self.hsdim )
        self.drhoy = sol_drhoy.reshape( self.hsdim, self.hsdim )
        self.drhoz = sol_drhoz.reshape( self.hsdim, self.hsdim )
        self.t = self.t + self.Delta_t
        
        #-- USING SOLVE_IVP
        
        #ts = [self.t, self.t+self.Delta_t]
        #sol_rho0_2 = solve_ivp(  self.bit_flip_RHS, ts, rho0_1D, t_eval=ts, vectorized=True )
        #sol_drhox_2 = solve_ivp(  self.bit_flip_RHS, ts, drhox_1D, t_eval=ts, vectorized=True )
        #sol_drhoy_2 = solve_ivp(  self.bit_flip_RHS, ts, drhoy_1D, t_eval=ts, vectorized=True )
        #sol_drhoz_2 = solve_ivp(  self.bit_flip_RHS, ts, drhoz_1D, t_eval=ts, vectorized=True )
        #
        #self.rho0 = sol_rho0_2.y[:,-1].reshape( self.hsdim, self.hsdim )
        #self.drhox = sol_drhox_2.y[:,-1].reshape( self.hsdim, self.hsdim )
        #self.drhoy = sol_drhoy_2.y[:,-1].reshape( self.hsdim, self.hsdim )
        #self.drhoz = sol_drhoz_2.y[:,-1].reshape( self.hsdim, self.hsdim )
        
    def apply_action( self, a ):
        
        #-- actions for 4 qubits
        #-- [None,0,1,2,3,4,5,6,7,8,9,10,11,0,1,2,3,0,1,2,3]
        
        N_cnots = self.N_qubits * (self.N_qubits-1)
        
        exp_RQ = None
        if a == 0:
            #-- stay idle
            pass
            exp_RQ = self.RQ()
            
        elif a > 0 and a <= N_cnots:
            #-- apply CNOT
            cnot_idx = self.actions[a]
            #print("-- CNOT_applied on qubits: with idx", cnot_idx )
            self.apply_CNOT( cnot_idx ) 
            exp_RQ = self.RQ()
            
        elif a > N_cnots and a <= N_cnots + self.N_qubits:
            #-- apply bit-flip
            qubit_to_flip = self.actions[a]
            self.apply_bit_flip( qubit_to_flip )
            exp_RQ = self.RQ()
            
        elif a > N_cnots + self.N_qubits and a <= N_cnots + 2*self.N_qubits:
            #-- apply z-measurement
            qubit_to_measure = self.actions[a]
            obtained_outcome_prob, other_outcome_prob, s_other_outcome \
                = self.apply_measurement_z( qubit_to_measure )
            # the rewards 
            exp_RQ = obtained_outcome_prob*self.RQ() \
                        + other_outcome_prob*s_other_outcome.RQ()
            
        else:
            print("====================")
            print("ERROR -- WRONG ACTION NUMBER")
            print("a = ", a)
            print("====================")
            sys.exit()
            
        reward1_, reward2_ = self.reward( exp_RQ )
        self.RQ_old = self.RQ()
        self.exp_RQ_old = exp_RQ
        
        return reward1_, reward2_
    
    def reward( self, exp_RQ  ):
        
        reward1 = 0
        reward2 = 0
        
        if exp_RQ > 1e-8:
            reward1 = 1 + (exp_RQ-self.RQ_old)/(2*self.Delta_t/self.T_single)
        if ( np.abs(self.RQ_old) > 1e-8 and np.abs(self.RQ()) < 1e-8 ):
            reward2 = - self.P

        return reward1, reward2
    
    def RQ( self ):
        
        drhox_eig, v = np.linalg.eig( self.drhox )
        drhoy_eig, v = np.linalg.eig( self.drhoy )
        drhoz_eig, v = np.linalg.eig( self.drhoz )
        
        trace_norm_drhox = np.sum( np.abs(drhox_eig) )
        trace_norm_drhoy = np.sum( np.abs(drhoy_eig) )
        trace_norm_drhoz = np.sum( np.abs(drhoz_eig) )
        
        RQ = np.min( [trace_norm_drhox, trace_norm_drhoy, trace_norm_drhoz] )
        
        return np.real(RQ)
            
    
    def get_action_type(self, a):
        
        N_cnots = self.N_qubits * (self.N_qubits-1)
        if a == 0:
            action_type = 'IDLE'
        elif a > 0 and a <= N_cnots:
            action_type = 'CNOT'
        elif a > N_cnots and a <= N_cnots + self.N_qubits:
            action_type = 'BITFLIP'
        elif a > N_cnots + self.N_qubits and a <= N_cnots + 2*self.N_qubits:
            action_type = 'ZMEASUREMENT'
        else:
            action_type = '---'
            print("ERROR -- WRONG ACTION NUMBER")
            
        return action_type
                    
        
    def apply_CNOT( self, i ):
        
        """ 
        Flip qbit 2 provided qubit 1  is in the excited state
        """
                            
        self.rho0 = self.cnots[:,:,i].dot(self.rho0).dot(self.cnots[:,:,i])
        self.drhox = self.cnots[:,:,i].dot(self.drhox).dot(self.cnots[:,:,i])
        self.drhoy = self.cnots[:,:,i].dot(self.drhoy).dot(self.cnots[:,:,i])
        self.drhoz = self.cnots[:,:,i].dot(self.drhoz).dot(self.cnots[:,:,i])
        
        
    def apply_bit_flip( self, qb1 ):
        
        self.eval_rhon()
        self.rho0 = self.Sig_x[:,:,qb1].dot( self.rho0 ).dot( self.Sig_x[:,:,qb1] )
        self.drhox = self.Sig_x[:,:,qb1].dot( self.drhox ).dot( self.Sig_x[:,:,qb1] )
        self.drhoy = self.Sig_x[:,:,qb1].dot( self.drhoy ).dot( self.Sig_x[:,:,qb1] )
        self.drhoz = self.Sig_x[:,:,qb1].dot( self.drhoz ).dot( self.Sig_x[:,:,qb1] )
        self.eval_rhon()
        
        
    def apply_measurement_z( self, qb1 ):
        
        """ 
        IMPORTANT: rhon only evaluated when needed
        """
        
        #-- eval rhon only when needed!!!
        self.eval_rhon()
        qb1_dm = self.partial_trace(self.rhon, keep=[qb1])
        r = np.random.rand()
        
        if (np.trace(qb1_dm) > 1 + 1e-6) or (np.trace(qb1_dm) < 1 - 1e-6):
            print("=== TRACE IS NOT UNITY ===")
            sys.exit()
        
        rho0_up = self.Pz_up[:,:,qb1].dot(self.rho0).dot(self.Pz_up[:,:,qb1])
        drhox_up = self.Pz_up[:,:,qb1].dot(self.drhox).dot(self.Pz_up[:,:,qb1])
        drhoy_up = self.Pz_up[:,:,qb1].dot(self.drhoy).dot(self.Pz_up[:,:,qb1])
        drhoz_up = self.Pz_up[:,:,qb1].dot(self.drhoz).dot(self.Pz_up[:,:,qb1])
        
        rho0_down = self.Pz_down[:,:,qb1].dot(self.rho0).dot(self.Pz_down[:,:,qb1])
        drhox_down = self.Pz_down[:,:,qb1].dot(self.drhox).dot(self.Pz_down[:,:,qb1])
        drhoy_down = self.Pz_down[:,:,qb1].dot(self.drhoy).dot(self.Pz_down[:,:,qb1])
        drhoz_down = self.Pz_down[:,:,qb1].dot(self.drhoz).dot(self.Pz_down[:,:,qb1])
        
        outcome = None
        other_outcome = None
        s_other_outcome = deepcopy(self)
        s_other_outcome.initialise_rho_mats()
        if r < np.abs(qb1_dm[0,0]):
            # project in up state
            if np.abs(np.trace(rho0_up))>1e-8:  #-- sanity if statement
                rho0_up /= np.trace(rho0_up)
                drhox_up /= np.trace(rho0_up)
                drhoy_up /= np.trace(rho0_up)
                drhoz_up /= np.trace(rho0_up)
                self.rho0 =  rho0_up 
                self.drhox = drhox_up 
                self.drhoy = drhoy_up 
                self.drhoz = drhoz_up 
            else:
                print('\n')
                print('EROOR: trace of rho_up<1e-8, UP projection')
                print('Tr(rho_up) = ', np.trace(rho0_up))
                sys.exit()
            
            if np.abs(qb1_dm[1,1])>1e-8: #-- in case trace==0: skip
                if np.abs(np.trace(rho0_down))>1e-8:
                    rho0_down /= np.trace(rho0_down)
                    drhox_down /= np.trace(rho0_down)
                    drhoy_down /= np.trace(rho0_down)
                    drhoz_down /= np.trace(rho0_down)
                    s_other_outcome.rho0 =  rho0_down
                    s_other_outcome.drhox = drhox_down 
                    s_other_outcome.drhoy = drhoy_down 
                    s_other_outcome.drhoz = drhoz_down 
                else:
                    print('\n')
                    print('EROOR: trace of rho_down<1e-8, UP projection')
                    print('Tr(rho_up) = ', np.trace(rho0_down))
                    sys.exit()
            
            outcome = 1
            other_outcome = -1
            obtained_outcome_prob = np.real( qb1_dm[0,0] )
            other_outcome_prob = np.real( qb1_dm[1,1] )
            
        else:
            # project in down state
            if np.abs(np.trace(rho0_down))>1e-8:
                rho0_down /= np.trace(rho0_down)
                drhox_down /= np.trace(rho0_down)
                drhoy_down /= np.trace(rho0_down)
                drhoz_down /= np.trace(rho0_down)
                self.rho0 =  rho0_down 
                self.drhox = drhox_down 
                self.drhoy = drhoy_down 
                self.drhoz = drhoz_down 
            else:
                print('\n')
                print('EROOR: trace of rho_down<1e-8, DOWN projection')
                print('Tr(rho_up) = ', np.trace(rho0_down))
                sys.exit()
            
            if np.abs(qb1_dm[0,0])>1e-8: #-- in case trace==0: skip
                if np.abs(np.trace(rho0_up))>1e-8:
                    rho0_up /= np.trace(rho0_up)
                    drhox_up /= np.trace(rho0_up)
                    drhoy_up /= np.trace(rho0_up)
                    drhoz_up /= np.trace(rho0_up)
                    s_other_outcome.rho0 =  rho0_up
                    s_other_outcome.drhox = drhox_up 
                    s_other_outcome.drhoy = drhoy_up 
                    s_other_outcome.drhoz = drhoz_up 
                else:
                    print('\n')
                    print('EROOR: trace of rho_up<1e-8, DOWN projection')
                    print('Tr(rho_up) = ', np.trace(rho0_up))
                    sys.exit()
                
            outcome = -1
            other_outcome = 1
            obtained_outcome_prob = np.real( qb1_dm[1,1] )
            other_outcome_prob = np.real( qb1_dm[0,0] )
            
            #-- checking that the state is properly projected
            self.eval_rhon()
            exp_z = np.trace( self.Sig_z[:,:,qb1].dot(self.rhon) )
            
            qb1_dm_TEST = self.partial_trace(self.rhon, keep=[qb1])
        
            if np.abs(exp_z - outcome) > 1e-8:
                print('\n')
                print("-- expected outcome = ", outcome)
                print("-- exp_z = ", exp_z)
                print('ERROR 1: state not properly projected after measurement')
                sys.exit()
            elif np.abs(qb1_dm_TEST[0,0] - (outcome+1)/2) > 1e-8:
                print('\n')
                print("-- expected outcome = ", outcome)
                print("-- exp_z = ", exp_z)
                print('ERROR 2: state not properly projected after measurement')
                sys.exit()
            else:
                pass
        
        return obtained_outcome_prob, other_outcome_prob, s_other_outcome
        
    
    def partial_trace(self, rho, keep, optimize=False):
        
        """Calculate the partial trace
    
        ρ_a = Tr_b(ρ)
    
        Parameters
        ----------
        ρ : 2D array
            Matrix to trace
        keep : array
            An array of indices of the spaces to keep after
            being traced. For instance, if the space is
            A x B x C x D and we want to trace out B and D,
            keep = [0,2]
        dims : array
            An array of the dimensions of each space.
            For instance, if the space is A x B x C x D,
            dims = [dim_A, dim_B, dim_C, dim_D]
    
        Returns
        -------
        ρ_a : 2D array
            Traced matrix
        """
        keep = np.asarray(keep)
        dims = np.array( [2]*(self.N_qubits) )
        Ndim = dims.size
        Nkeep = np.prod(dims[keep])
    
        idx1 = [i for i in range(Ndim)]
        idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
        rho_a = rho.reshape(np.tile(dims,2))
        rho_a = np.einsum(rho_a, idx1+idx2, optimize=optimize)
        
        return rho_a.reshape(Nkeep, Nkeep)
