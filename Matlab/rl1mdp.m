classdef (Abstract) rl1mdp < handle
    properties
        gamma = 0.9; % discount factor
        state % world state
    end
    
    methods (Abstract)
        sp = P(o, s, a) % Transition s' = P(s,a)
        r = R(o, s, a, sp) % Reward r = R(s,a,s')
        [newState, reward, isTerminal] = step(o, action) % Step the world
        reset(o, state) % Reset the world with initial condition "state"
    end 
end