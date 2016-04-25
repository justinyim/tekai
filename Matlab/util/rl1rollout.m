function [st, at, r] = rl1rollout(mdp, policy, init, varargin)
% Rollout an mdp and policy pair
% INPUTS:
% mdp: [rl1mdp]
% policy: [rl1policy]
% init: [1x1 double] initial state (index)
% 1) [1x1 double] maximum iterations
% OUTPUTS:
% st: [nx1 double] list of states in the trajectory
% at: [nx1 double] list of actions in the trajectory
% r: [nx1 double] rewards from the trajectory

% 
max_iters = 1e4;
if nargin >= 1
    max_iters = varargin{1};
end

% Run rollout
st = zeros(max_iters+1,1);
at = zeros(max_iters,1);
r = zeros(max_iters,1);

% Set initial conditions
mdp.reset(init);
st(1) = init;

for ii = 1:max_iters
    at(ii) = policy.pi(mpd.state);
    [st(ii+1), r(ii), isTerminal] = mdp.step(a);
    if isTerminal
        st((ii+2):end) = [];
        at((ii+1):end) = [];
        r((ii+1):end) = [];
        break
    end
end

end