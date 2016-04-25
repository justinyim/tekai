import numpy as np
import random as rr




def Rollout(MDP,policy):
    #outputs a rollout
    
    MDP.reinitialize()
    
    state=MDP.state

    trajectory=[]
    final=0
    while final!=1:
        rand=rr.random()

        index=MDP.pos[(state[0],state[1])]

        pol=np.cumsum(policy[index,:])<=rand
        for i in range(len(pol)):
            if pol[i]==False:
                break

        action=i+1

        oldstate=state
        (state,r,final,action)=MDP.step(action)
        trajectory.append([oldstate[0],oldstate[1],action,r])

        if final==1:
            trajectory.append([state[0],state[1],0,0])

    trajectory=np.array(trajectory)
    return trajectory
