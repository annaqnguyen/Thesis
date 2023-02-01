clc;
close all;
clear;

%%% Imbalance
% time = 8*pi;
% x = linspace(0, time, 1000);
% y1 = sin(x);
% y2 = sin(x + degtorad(120));
% y3 = 1.75*sin(x + degtorad(240));
% plot(x, y1, 'r');
% hold on;
% plot(x, y2, 'color', [0.4660 0.6740 0.1880]);
% plot(x, y3, 'b');
% axis([0 time -2 2]);
% xlabel('Sample');
% ylabel('Voltage (pu)');
% title('Voltage Imbalance');
%  print -depsc VI

% %%% Sustained Interruptions
% time = 10*pi;
% x = linspace(0, time, 3000);
% x1 = linspace(0, 3*pi, 300);
% y = sin(x1);
% y1 = zeros(1, 2700);
% yfinal = cat(2, y, y1);
% plot(x, yfinal)
% axis([0 time -1.2 1.2]);
% xlabel('Sample');
% ylabel('Voltage (pu)');
% title('Sustained Interruption');
% 
% print -depsc SI

%%%Over/Under voltage
% time = 10*pi;
% x = linspace(0, time, 1000);
% y = sin(x);
% under = 0.8*sin(x);
% over = 1.2*sin(x);

% subplot(1, 2, 1);
% plot(x, under, 'r');
% hold on;
% plot(x, y, 'b');
% axis([0 time -1.5 1.3]);
% xlabel('Sample');
% ylabel('Voltage (pu)');
% title('Under-Voltage');
% 
% subplot(1, 2, 2);
% plot(x, over, 'r');
% hold on;
% plot(x, y, 'b');
% axis([0 time -1.5 1.3]);
% xlabel('Sample');
% ylabel('Voltage (pu)');
% title('Over-Voltage');
% 
% print -depsc OU

%%% Waveform Abnormality Hypotheses

x = [-6:.1:6];
y = normpdf(x,0,1);
y1 = y/3;
y2 = zeros(size(y1));
n = 30;
y2(1:end-n) = y1(n+1:end);

plot(x, y);
hold on;
plot(x, y2);
xlabel('Evaluation Point');
ylabel('Distribution');
title('Noise from differential waveform');
legend('Normal random noise distribution', 'Abnormal component distribution');
print -depsc WAH

