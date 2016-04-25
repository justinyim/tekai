classdef rl1gridWorld1 < rl1mdp
    % Simple gridworld with reward function R(s,a,s') = R(s) and transition
    % probability s' = s + "action" with probability o.p and s' = random
    % with probability 1-o.p
    properties
        rewards = zeros(3,3); % grid of rewards R(s)
        terminals = []; % list of terminal states
        p = 0.75;
    end
    
    methods
        function o = rl1gridWorld1(varargin)
            % Constructor
            % INPUTS:
            % 1) grid of rewards R(s)
            % 2) list of terminal states
            % 3) gamma
            % 4) action/random probability
            % 5) random seed
            % OUTPUTS:
            % o: [rl1gridWorld1]
            
            o.state = 1;
            
            if nargin >= 1
                o.rewards = varargin{1};
            end
            if nargin >= 2
                o.terminals = varargin{2};
            end
            if nargin >= 3
                o.gamma = varargin{3};
            end
            if nargin >= 4
                o.p = varargin{4};
            end
            if nargin >= 5
                rng(varargin{5});
            end
        end
        
        function sp = P(o, varargin)
            % Transition probabilities
            % INPUTS: EITHER:
            % 1) action
            % OR:
            % 1) current state
            % 2) action
            % OUTPUTS:
            % sp: [1x1 double]: new state (index)
            
            % Parse inputs
            if nargin == 2
                s = o.state;
                a = varargin{1};
            else
                s = varargin{1};
                a = varargin{2};
            end
            
            % Select between given action and random action
            if rand(1) < o.p % take given action
                %act = act;
            else % take random action
                a = ceil(rand(1)*4);
            end
            
            % Transition
            %[s2,s1] = ind2sub(size(o.rewards), s);
            s2 = mod(s-1,size(o.rewards,1))+1;
            s1 = floor((s-1)/size(o.rewards,1))+1;
            sp = s;
            if a == 1 % move right
                if s1 < size(o.rewards,2)
                    sp = s + size(o.rewards,1);
                end
            elseif a == 2 % move down
                if s2 < size(o.rewards,1)
                    sp = s + 1;
                end
            elseif a == 3 % move left
                if s1 > 1
                    sp = s - size(o.rewards,1);
                end
            else % move up
                if s2 > 1
                    sp = s - 1;
                end
            end
        end
        
        function r = R(o, varargin)
            % Rewards: R(s,a,s') = R(s)
            % INPUTS:
            % 1) state
            % OUTPUTS:
            % r: [1x1 double]: reward
            
            r = o.rewards(varargin{1});
        end
        
        function [newState, reward, isTerminal] = step(o, action)
            % Step the world
            % INPUTS:
            % action: [1x1 double]: action at this time in {1,2,3,4}
            % OUTPUTS:
            % newState [1x1 double]: next state (index)
            % reward: [1x1 double]: reward at this time step
            % isTerminal: [1x1 bool]: is the next state terminal
            
            if any(o.terminals == o.state)
                newState = o.state;
                reward = 0;
                isTerminal = 1;
                return
            end
            o.state = o.P(action);
            newState = o.state;
            reward = o.R(o.state);
            isTerminal = any(o.terminals == newState);
        end
        
        function reset(o, state)
            % Reset the world
            % INPUTS:
            % state: [1x1 double]: initial state (index)
            % no OUTPUTS
            
            o.state = state;
        end
        
        function visTraj(o, policy, st, at, r, varargin)
            % Visualize a gridworld rollout
            % INPUTS:
            % st: state trajectory
            % at: action trajectory
            % no OUTPUTS
            
            step_t = 0.1;
            if nargin >= 4
                step_t = varargin{1};
            end
            
            n_steps = length(at);
            
            % Prepare states and actions for plotting
            [s2,s1] = ind2sub(size(o.rewards),st);
            a1 = (at == 1) - (at == 3);
            a2 = (at == 2) - (at == 4);
            
            % Prepare plot objects
            figure(101)
            clf
            
            % Plot rewards
            imagesc(o.rewards);
            colormap('gray')
            axis equal
            set(gca,'xlim',[0,size(o.rewards,2)+1],'ylim',[0,size(o.rewards,1)+1])
            hold on

            [yt, xt] = ind2sub(size(o.rewards),o.terminals);
            plot(xt,yt,'bx','markersize',20,'linewidth',3);
            
            % Plot policy
            mags = bsxfun(@rdivide, exp(policy.params), sum(exp(policy.params),2))/2;
            [xs,ys] = meshgrid(1:size(o.rewards,2), 1:size(o.rewards,1));
            
            quiver(xs, ys, reshape(mags(:,1),size(o.rewards)), zeros(size(o.rewards)),...
                'g','autoscale','off','showarrowhead','off','linewidth',1);
            quiver(xs, ys, zeros(size(o.rewards)), reshape(mags(:,2),size(o.rewards)), ...
                'g','autoscale','off','showarrowhead','off','linewidth',1);
            quiver(xs, ys, -reshape(mags(:,3),size(o.rewards)), zeros(size(o.rewards)),...
                'g','autoscale','off','showarrowhead','off','linewidth',1);
            quiver(xs, ys, zeros(size(o.rewards)), -reshape(mags(:,4),size(o.rewards)),...
                'g','autoscale','off','showarrowhead','off','linewidth',1);
            
            % Plot agend and action
            h_agent = plot(s1(1),s2(1),'ro','markersize',15,'MarkerFaceColor','r');
            h_action = quiver(s1(1),s2(1),a1(1),a2(1),'b','autoscale','off','linewidth',2);
            
            hold off
            
            R = 0;
            % Visualization loop
            for ii = 1:n_steps
                set(h_agent,'xdata',s1(ii),'ydata',s2(ii));
                set(h_action,'xdata',s1(ii),'ydata',s2(ii),'udata',a1(ii),'vdata',a2(ii));
                
                R = o.gamma^(ii-1)*r(ii) + R;
                title([R, r(ii)]);
                
                pause(step_t)
            end
            set(h_agent,'xdata',s1(end),'ydata',s2(end));
            set(h_action,'xdata',s1(end),'ydata',s2(end),'udata',0,'vdata',0);
            drawnow
            
        end
        
    end
end