classdef rl1policyGw1 < rl1policy
    properties
        epsilon = 0;
    end
    
    methods
        function o = rl1policyGw1(varargin)
            % Constructor
            % INPUTS: 
            % 1) parameter vector
            % OUTPUTS:
            % o: [rl1policyGw1]
            
            o.params = varargin{1};
        end
        
        function a = pi(o,s)
            % Policy
            % INPUTS:
            % s: [1x1 double]: current state (index)
            % OUTPUTS:
            % a: [1x1 double]: action in {1,2,3,4}
            
            ps = exp(o.params(s,:))/sum(exp(o.params(s,:)));
            bins = cumsum(ps);
            a = find(rand(1)<bins,1,'first');
            
            %{
            if rand(1) < o.epsilon
                a = ceil(4*rand(1));
            end
            %}
        end
        
        function g = gradients(o, s, a)
            % Policy gradient
            % INPUTS:
            % s: [1x1 double]: state at which to evaluate gradient (index)
            % a: [1x1 double]: action in {1,2,3,4} at which to evaluate
            % OUTPUTS:
            % g: [nx1 double] policy gradient
            
            g = zeros(numel(o.params),1);
            
            denom = sum(exp(o.params(s,:)));
            
            g(s + size(o.params,1)*(a-1)) = 1;
            g(s + size(o.params,1)*(0:3)) = g(s + size(o.params,1)*(0:3)) - ...
                exp(o.params(s,:))'./denom';
        end
        
    end
    
end