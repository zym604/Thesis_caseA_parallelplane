clc;clear;
load('main3-4.mat')
run('E:\Google\NCSU\Research\2016-09-17-PDTC V6.1 2D linear2nonlinear Non-Newton\code\gpml\startup.m');
%----------------------------- Input data ---------------------------------
trainingdata = [data2;data3;data5];
validatadata = data4;
compara_data =    R4;
% Reid = 6; % uu = 1; vv = 2; ww = 3; uv = 4; uw = 5; vw = 6
startcolDB = 8; %start colume of database (input colume)
%------------------------------ Run code ----------------------------------
% 1. Data refinement of case 2,3,5
% [ xq2,yq2,dpdx2 ] = f_datarefinement( trainingdata,startcolDB,data2 );
% [ xq3,yq3,dpdx3 ] = f_datarefinement( trainingdata,startcolDB,data3 );
% [ xq5,yq5,dpdx5 ] = f_datarefinement( trainingdata,startcolDB,data5 );
% % 2. without data preprocessing
% [ xq2,yq2,dpdx2 ] = f_allocate( startcolDB,data2 );
% [ xq3,yq3,dpdx3 ] = f_allocate( startcolDB,data3 );
% [ xq5,yq5,dpdx5 ] = f_allocate( startcolDB,data5 );
% combineddata = [yq2,zeros(size(dpdx2)),xq2,dpdx2;yq3,zeros(size(dpdx3)),xq3,dpdx3;yq5,zeros(size(dpdx5)),xq5,dpdx5];

% SMOTER preprocessing
tb = readtable('case4_13.csv');
combineddata = table2array(tb(:,2:end));

% Gaussian process
figure('Position',[230 100 1000 700])
for Reid = 1:4
    subplot(2,2,Reid);
    [Rresult{Reid},RMSE{Reid}] = p_trainedresult( combineddata,startcolDB,Reid,validatadata,compara_data );
end
RMSE
% save data
save('main4-5.mat')