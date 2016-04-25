%run
addpath util

% Initialization ======================
trainIters = 2e2;
evalEpoch = 1e1;
rolloutIters = 2e2;
maxRolloutSteps = 1e2;
alpha = 0.1;

%rng(118);

r = -10;
gamma = 0.9;
r_living = 0;

Rs = [10, 0, 0;
      -5, r, 0;
      -10, -5, 0] + r_living;
terminals = [1,3];
init = 9;
%{
Rs = rand(4,5)-1;
terminals = ceil(numel(Rs)*rand(1,2));
init = ceil(numel(Rs)*rand(1));
%}
%{
Rs = rand(5,10)-1;
terminals = 1;
init = numel(Rs);
%}

th_init = rand(numel(Rs),4)-0.5;

world = rl1gridWorld1(Rs, terminals, gamma); 
policy = rl1policyGw1(th_init); 

rewardHistory = zeros(trainIters,1);
gradMags = zeros(trainIters,2);


% Training ============================
sts = cell(rolloutIters,1);
ats = cell(rolloutIters,1);
rs = cell(rolloutIters,1);

world.reset(1);

for ii = 1:trainIters
    % Rollouts ------------------------
    for jj = 1:rolloutIters
        [st, at, r] = rl1rollout(world, policy, init, maxRolloutSteps);
        sts{jj} = st;
        ats{jj} = at;
        rs{jj} = r;
    end
    
    % Update --------------------------
    % Policy gradient form 1
    %{
    J = zeros(numel(policy.params),1);
    for jj = 1:rolloutIters
        st = sts{jj};
        at = ats{jj};
        r = rs{jj};
        n_steps = length(at);
        
        discounts = (world.gamma*ones(1,n_steps)).^(1:n_steps);
        R = discounts*r;
        rewardHistory(ii) = rewardHistory(ii) + R/rolloutIters; % Learning Curve
        for kk = 1:length(at)
            J = J + policy.gradients(st(kk),at(kk))*R;
        end
    end
    J = J/rolloutIters;
    gradMags(ii,1) = norm(J);
    %}
    
    % Policy gradient form 2
    %{
    J = zeros(numel(policy.params),1);
    for jj = 1:rolloutIters
        st = sts{jj};
        at = ats{jj};
        r = rs{jj};
        n_steps = length(at);
        
        discounts = (world.gamma*ones(1,n_steps)).^(1:n_steps); % Learning curve
        rewardHistory(ii) = rewardHistory(ii) + discounts*r/rolloutIters; % Learning curve
    
        for kk = 1:length(at)
            discounts = [zeros(1,kk-1),(world.gamma*ones(1,n_steps-kk+1))].^(1:n_steps);
            R = discounts*r;
            J = J + policy.gradients(st(kk),at(kk))*R;
        end
    end
    J = J/rolloutIters;
    %}
    
    % Policy gradient form 1 with baseline
    %{a
    gradPi = zeros(numel(policy.params),rolloutIters);
    gradPiR = zeros(numel(policy.params),rolloutIters);
    rewards = zeros(1,rolloutIters);
    for jj = 1:rolloutIters
        st = sts{jj};
        at = ats{jj};
        r = rs{jj};
        n_steps = length(at);
        
        discounts = (world.gamma*ones(1,n_steps)).^(1:n_steps);
        rewards(jj) = discounts*r;
        rewardHistory(ii) = rewardHistory(ii) + discounts*r/rolloutIters; % Learning curve
        
        for kk = 1:n_steps
            gradPi(:,jj) = gradPi(:,jj) + policy.gradients(st(kk),at(kk));
            gradPiR(:,jj) = gradPiR(:,jj) + policy.gradients(st(kk),at(kk))*rewards(jj);
        end
    end
    bj = sum(gradPiR,2)./(sum(gradPi.^2,2)+eps);
    
    J = sum(bsxfun(@times,gradPi,bsxfun(@minus,rewards,bj)),2)./rolloutIters;
    J2 = sum(gradPiR,2)./rolloutIters;
    
    gradMags(ii,:) = [norm(J2), norm(J)];
    %}
    
    % Policy gradient form 2 with baseline
    %{
    J = zeros(numel(policy.params),1);
    for jj = 1:rolloutIters
        st = sts{jj};
        at = ats{jj};
        r = rs{jj};
        n_steps = length(at);
        
        discounts = (world.gamma*ones(1,n_steps)).^(1:n_steps); % Learning curve
        rewardHistory(ii) = rewardHistory(ii) + discounts*r/rolloutIters; % Learning curve
    
        for kk = 1:length(at)
            discounts = [zeros(1,kk-1),(world.gamma*ones(1,n_steps-kk+1))].^(1:n_steps);
            R = discounts*r;
            J = J + policy.gradients(st(kk),at(kk))*R;
        end
    end
    J = J/rolloutIters;
    %}
    
    policy.params = policy.params + alpha*reshape(J,[numel(Rs),4]);
    
    % Evaluate ------------------------
    if ~mod(ii-1,10)
    	world.visTraj(policy, st,at,r,0.01)
    end
end

% Evaluation ==========================

