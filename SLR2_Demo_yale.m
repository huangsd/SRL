% ====================================================
% An example code for the algorithm proposed in
%
% Smooth Representation Learning for Multi-view Data (SRL)
% Aug., 2023.
% ====================================================
clc;
clear
close all
addpath(genpath(cd)) ;

%==================== data set =========================
name = 'yale_mtv';
load(name)

%=================== parameter settings =================
eta_para = 1000;
gamma_para = 30;
k_para = 2;

%==================== performing SLR ===================
fprintf('Begin performing on dataset %s......\n',name)
result = zeros(10,8);
for i =1:10
    cur_res = SRLMD(data, labels,eta_para,gamma_para, k_para, 10, 2);
    fprintf('Iter = %d: ACC = %4.2f, NMI = %4.2f, Purity = %4.2f, Fscore = %4.2f \n',i,cur_res(2),cur_res(1),cur_res(3),cur_res(4))
    result(i,:) = cur_res;
end
fprintf('The average result is: \n')
avg = mean(result);
fprintf('ACC = %4.2f, NMI = %4.2f, Purity = %4.2f, Fscore = %4.2f \n',avg(2),avg(1),avg(3),avg(4))
