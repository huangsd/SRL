% Smooth Representation Learning for Multi-view Data--20230828

function [result,Tim] = SRLMD(data, labels, eta, gamma, k, Iter,normData)
% multi-view data: cell array, view_num by 1, each array is num_samp by d_v
% labels: groundtruth of the data, num_samp by 1
% num_clus: number of clusters
% num_view: number of views
% num_samp: number of samples
% k: Order of the low-pass filter based on normalized Laplacian Fourier base
tic;
if nargin < 3
    eta = 1;
end
if nargin < 4
    gamma = 1;
end
if nargin < 5
    k = 2;
end
if nargin < 6
    Iter = 15;
end
if nargin < 7
    normData = 1;
end
num_view = size(data,1);
num_samp = size(labels,1);
num_clus = length(unique(labels));
mu = 1/num_view*ones(num_view,1);
opts.record = 0;
opts.mxitr  = 1000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;
out.tau = 1e-3;
Aeq = ones(1,num_view);
Beq = 1;
lb = zeros(num_view,1);
c=max(labels);

%% === Normalization1 ===
if normData == 1
    for i = 1:num_view
        dist = max(max(data{i})) - min(min(data{i}));
        m01 = (data{i} - min(min(data{i})))/dist;
        data{i} = 2 * m01 - 1;
    end
end
% === Normalization2 ===
if normData == 2
    for iter = 1:num_view
        for  j = 1:num_samp
            normItem = std(data{iter}(j,:));
            if (0 == normItem)
                normItem = eps;
            end
            data{iter}(j,:) = (data{iter}(j,:) - mean(data{iter}(j,:)))/normItem;
        end
    end
end

% === initialize === 
FV = cell(num_view, 1);
ZV = cell(num_view, 1);
RV = cell(num_view, 1); 
Rv = eye(num_clus);

for v = 1:num_view
    Zv = (data{v}*data{v}'+ eta*eye(num_samp))\(data{v}*data{v}');
    Zv(find(Zv<0)) = 0;
    Zv = (Zv + Zv')/2;
    Zv = Zv - diag(diag(Zv));
    Lv = diag(sum(Zv)) - Zv;
    [Fv, ~, ev] = eig1(Lv, num_clus, 0);
    FV{v} = Fv;
    ZV{v} = Zv;
    RV{v} = Rv;
end
%% === iteration === 
% fprintf('begin updating ......\n')
for iter = 1:Iter
%      fprintf('the %d -th iteration ...... \n',iter) 
    %
    % === update Zv ===
    LV_norm{v} = cell(num_view, 1);
    data_bar = cell(num_view, 1);
    for v = 1:num_view
        Zv = ZV{v};
        Dv = diag(sum(Zv));
        Lv = eye(num_samp) - Dv^(-1/2) * Zv * Dv^(-1/2);
        LV_norm{v} = Lv;
        XV_bar = data{v};
        for i = 1:k % === k order ===
            XV_bar = (eye(num_samp) - Lv/2)*XV_bar;
        end
        temp = inv((XV_bar*XV_bar') + eta*eye(num_samp)); 
        data_bar{v} = XV_bar;
        XV = XV_bar';
        Fv = FV{v};
        for ij = 1:num_samp
            d = distance(Fv, num_samp, ij);
            XX = XV'*XV;
            Zv(:,ij) = temp*(XX(ij,:)' - (gamma/4)*d'); 
        end
        Zv(find(Zv<0)) = 0;
        Zv = (Zv + Zv')/2;
        Zv = Zv - diag(diag(Zv));
        ZV{v} = Zv;
    end
    %
    % === update Y ===
    if iter > 1
        Y_old = Y;
    end
SumFF = zeros(num_samp, num_samp);
    for v = 1:num_view
        SumFF = SumFF + 2*mu(v)*(FV{v}*FV{v}');
    end
    [Y, ~, ~] = eig1((eye(num_samp) - SumFF), num_clus, 0);
    %
    % === update Fv === 
    for v = 1:num_view
        SumFF = SumFF + mu(v)*(FV{v}*FV{v}');
    end
    LV = cell(num_view, 1);
    for v = 1:num_view
        Lv = diag(sum(ZV{v})) - ZV{v};
        LV{v} = Lv;
        A = gamma*mu(v)*LV{v} + (mu(v)^2)*eye(num_samp) - 2*mu(v)*((Y*Y') - (SumFF - mu(v)*(FV{v}*FV{v}')));
        [FV{v}, ~, ~] = eig1(A, num_clus, 0);
    end
    % 
    % === update \mu === 
    % calculate W(i,j)
    W = zeros(num_view, num_view);
    for ii = 1:num_view
        for jj = 1:num_view
            Fii = FV{ii};
            Fjj = FV{jj};
            W(ii,jj) = trace(Fii*Fii'*(Fjj*Fjj'));
        end
    end
    % calculate b
    for ii = 1:num_view 
        temp1 = (norm(data_bar{ii}'-data_bar{ii}'*ZV{ii},'fro'))^2 + eta*(norm(ZV{ii},'fro'))^2 + gamma*trace(FV{ii}'*LV{ii}*FV{ii});
        b(ii) = temp1 - 2*trace(Y*Y'*FV{ii}*FV{ii}');
    end
    mu = fun_alm(W, b); % paramu 
    mu(find(mu<=0)) = 0.01; 
   %

   % break threshold
   if (iter > 3) && ((norm(Y-Y_old,'fro')/norm(Y_old,'fro')) < 1e-6)
       break
   end 
end

%% result = [nmi ACC Purity Fscore Precision Recall AR Entropy];
   YY = kmeans(Y, c, 'emptyaction', 'singleton', 'replicates', 1, 'display', 'off');
   result = Clustering8Measure(labels, YY);
Tim=toc;
end

function [all] = distance(F,n,ij)
  for ji = 1:n
      all(ji) = (norm(F(ij,:)-F(ji,:)))^2;
  end
end   

function [F,G] = fun1(P,alpha,Y,Q,L)
    G = 2*L*P - 2*alpha*Y*Q';
    F = trace(P'*L*P) + alpha*(norm(Y-P*Q,'fro'))^2;
end
