import numpy as np

q_s_a = np.asarray([1, 3, 2, 4])
d = np.array([0,1,1,0])
alpha = 0.1

v_q_s = alpha * np.log(np.sum(np.exp(q_s_a/alpha)))
print(v_q_s)
policy_q_a_s = np.exp((q_s_a-v_q_s)/alpha)
print(policy_q_a_s)
print(np.sum(policy_q_a_s))
x=0
print(np.extract( 1- d, q_s_a))
print(np.reshape(q_s_a,(4,1))/np.sum(q_s_a) *np.reshape(q_s_a,(4,1))* 4)
print(14.03**(1/10))