classdef (Abstract) rl1policy < handle
    properties
        params % policy parameters
    end
    
    methods (Abstract)
        a = pi(o,s); % policy 
        g = gradients(o, s, a); % policy gradient
    end
    
end