function [neuro,cbf] = neuronal_NVC_model_from_AllenModel(P),
% neuronal_NVC_model is a simple model to generate neuronal response (as balance 
%                 between excitatory and inhibitory activity) followed by blood flow (CBF)
%                 response, based on Havlicek, et al.(2015) NeuroImage  
%
% AUTHOR: Martin Havlicek, 5 August, 2019; adapted by Lea Demelius
%
% REFERENCE: Havlicek, M., Roebroeck, A., Friston, K., Gardumi, A., Ivanov, D., Uludag, K.
%           (2015) Physiologically informed dynamic cousal modeling of fMRI data, NeuroImage (122), pp. 355-372
%--------------------------------------------------------------------------
K      = P.K;

% Neuronal parameter:
%--------------------------------------------------------------------------
sigma   = P.sigma;     % self-inhibitory connection 
mu      = P.mu;        % inhibitory-excitatory connection
lambda  = P.lambda;    % inhibitory gain
Bsigma  = P.Bsigma;    % modulatory parameter of self-inhibitory connection
Bmu     = P.Bmu;       % modulatory parameter of inhibitory-excitatory connection
Blambda = P.Blambda;   % modulatory parameter of inhibitory connection
C       = P.C;
% NVC parameters:
% --------------------------------------------------------------------------
c1      = P.c1;
c2      = P.c2;
c3      = P.c3;

% Initial condtions:
Xn  = zeros(K,2);
yn  = Xn;

% Load firing rates
% supra = load('down/firingrate_tbin10_upper.mat');
% granular = load('down/firingrate_tbin10_middle.mat');
% infra = load('down/firingrate_tbin10_lower.mat');
supra = load('fr_laminar_down_tbin100_upper.mat');
granular = load('fr_laminar_down_tbin100_middle.mat');
infra = load('fr_laminar_down_tbin100_lower.mat');

supra_fr = supra.firingrate; %/16.1603; %/15;
granular_fr = granular.firingrate; %/16.1603;%/15;
infra_fr = infra.firingrate; %16.1603;%/15;
maxmax = max([max(supra_fr),max(granular_fr),max(infra_fr)]);
fr = cat(1,supra_fr/maxmax, granular_fr/maxmax, infra_fr/maxmax);


dt = P.dt; 
T = P.T;

neuro = zeros(T/dt,K);
%neuro_inh = zeros(T/dt,K);


cbf   = zeros(T/dt,K);
for t = 1:T/dt;
    Xn(:,2) = exp(Xn(:,2));

%     A = eye(K)*sigma;
%     MU = ones(K,1)*mu;
%     LAM = ones(K,1)*lambda;
%     for i = 1:size(Bsigma,2)
%         A = A + diag(Bsigma(:,i)).*U.m(t,i);
%     end;
%     for i = 1:size(Bmu,2)
%         MU = MU + Bmu(:,i).*U.m(t,i);
%     end;
%     for i = 1:size(Blambda,2)
%         LAM = LAM + Blambda(:,i).*U.m(t,i);
%     end
    %----------------------------------------------------------------------
    % Neuronal (excitatory & inhibitory)
%     yn(:,1)   = yn(:,1) + dt*(A*Xn(:,1) - MU.*Xn(:,2) + C*U.u(t,:)');
% 
%     yn(:,2)   = yn(:,2) + dt*(LAM.*(-Xn(:,2) +  Xn(:,1)));
    %----------------------------------------------------------------------
    % Vasoactive signal:
    yn(:,1)   = yn(:,1) + dt*(fr(:,t) - c1.*(Xn(:,1)));
    %----------------------------------------------------------------------
    % Inflow:
    df_a      = c2.*Xn(:,1) - c3.*(Xn(:,2)-1);
    yn(:,2)   = yn(:,2) + dt*(df_a./Xn(:,2));
    
    Xn         = yn;
     
    cbf(t,:)   = exp(yn(:,2))';
    neuro(t,:) = fr(:,t)';
    %neuro_inh(t,:) = yn(:,2)';
end