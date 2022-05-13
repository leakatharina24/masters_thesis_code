close all; clear all;

set(0,'DefaultAxesFontSize', 14, ...
      'defaultLineLineWidth', 2, ...
      'defaultLineMarkerSize',15,...
      'DefaultAxesTitleFontWeight', 'normal');      
  
K         = 3;   % number of depths %6

% Specify neuronal and NVC model:
%--------------------------------------------------------------------------
P.N       = neuronal_NVC_parameters(K);  % get default parameters (see inside the function)
P.N.T     = 532;               % Total lenght of the response (in seconds)
P.N.dt    = 0.1;
[neuro, cbf]  = neuronal_NVC_model_from_BillehModel(P.N); % Generate the neuronal and cerebral blood flow response (CBF)

time_axis = [0:P.N.dt:P.N.T-P.N.dt]; % time axis in seconds

figure(1);
% subplot(1,2,1);
title('Firing rate');
plot(time_axis,neuro(:,1));
hold on;
plot(time_axis,neuro(:,2));
plot(time_axis,neuro(:,3));
xlim([time_axis(1), time_axis(end)]);
xlabel('time in s');
ylabel('firing rate in spikes/s');
legend('upper','middle', 'lower')
% subplot(1,2,2);
% title('CBF');
% plot(time_axis, cbf);
% xlim([time_axis(1), time_axis(end)]);

% Specify LBR model:
%--------------------------------------------------------------------------  
P.H       = LBR_parameters(K); % get default parameters (see inside the function), 
                               % NOTE: By default baseline CBV is increasing towards the surface in the ascending vein
P.H.T     = P.N.T;             % copy the lenght of the response from neuronal specification
P.H.dt    = P.N.dt;
  
P.H.alpha_v   = 0.35;          % Choose steady-state CBF-CBV coupling for venules
P.H.alpha_d   = 0.2;           % Choose steady-state CBF-CBV coupling for ascending vein
P.H.tau_d_de  = 30;            % Choose dynamic CBF-CBV uncoupling for ascending vein

[LBR,Y] = LBR_model(P.H,cbf);  % Generate the laminar bold response

figure(2)
plot(time_axis,LBR(:,1));
hold on;
plot(time_axis,LBR(:,2));
plot(time_axis,LBR(:,3));
legend('upper','middle', 'lower')
xlabel('time in s');
ylabel('BOLD responses in %');
title('Laminar balloon model');
xlim([time_axis(1), time_axis(end)]);

bold_up = LBR(:,1);
bold_mid = LBR(:,2);
bold_low = LBR(:,3);

%save('results_down/laminarballoon_upper.mat','bold_up'),
%save('results_down/laminarballoon_middle.mat','bold_mid'),
%save('results_down/laminarballoon_lower.mat','bold_low'),

% Display underlying physiological responses
figure(3),
subplot(231), p=plot(time_axis,cbf); xlim([time_axis(1), time_axis(end)]); %ylim([0.8 1.6]);
xlabel('Time (s)'); ylabel('Relative CBF in MV (%)'); axis square; legend([p(1) p(2) p(end)],{'Upper','Middle','Lower'});
subplot(232), p=plot(time_axis,Y.mv); xlim([time_axis(1), time_axis(end)]); %ylim([0.8 1.6]);
xlabel('Time (s)'); ylabel('Relative CMRO_2 in MV (%)'); axis square; legend([p(1) p(2) p(end)],{'Upper','Middle','Lower'});
subplot(233), p=plot(time_axis,Y.vv); xlim([time_axis(1), time_axis(end)]); %ylim([0.8 1.6]);
xlabel('Time (s)'); ylabel('Relative CBV in MV (%)'); axis square;legend([p(1) p(2) p(end)],{'Upper','Middle','Lower'});
subplot(234), p=plot(time_axis,Y.qv); xlim([time_axis(1), time_axis(end)]); %ylim([0.7 1.2]);
xlabel('Time (s)'); ylabel('Relative dHb in MV (%)'); axis square; legend([p(1) p(2) p(end)],{'Upper','Middle','Lower'});
subplot(235), p=plot(time_axis,Y.vd); xlim([time_axis(1), time_axis(end)]); %ylim([0.8 1.6]);
xlabel('Time (s)'); ylabel('Relative CBV in AV (%)'); axis square; legend([p(1) p(2) p(end)],{'Upper','Middle','Lower'});
subplot(236), p = plot(time_axis,Y.qd); xlim([time_axis(1), time_axis(end)]); %ylim([0.7 1.2]);
xlabel('Time (s)'); ylabel('Relative dHb in AV (%)'); axis square; legend([p(1) p(2) p(end)],{'Upper','Middle','Lower'});

