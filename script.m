%%
clear
clc
addpath ~/Projects/eeglab;eeglab;
hm = headModel.loadDefault;
close all

%% Simulate one column
cc = CorticalColumn({'sigma',diag([1 1 1]*1e-2)});
cc.simulate;
cc.plot

%% One column with spike input
Fs = 1/cc.dt;
u = zeros(cc.nt,3);
u(Fs:Fs+10,2) = 100;
cc = CorticalColumn({'sigma',diag(0.01*ones(3,1)),'u',u});
cc.simulate;
openField = [1, 5];
cc.plot(openField)

%% 100 coupled columns
u = zeros(cc.nt,3);
u(Fs:Fs+10,3) = 0;
cc = CorticalColumn({'sigma',diag([1 1 1]*1e2),'u',u});

n = 100;
corticalColumns = repmat({cc},n,1);
dcm = DynamicCausalModel({'nmmArray',corticalColumns,'SchortRange',2*(triu(rand(n))+tril(rand(n)))/n});
pyCell = 1:6:dcm.nx;
xsim = dcm.simulate;
dcm.plot(xsim(pyCell,:))


%% 100 coupled columns
n = 50;
u = zeros(cc.nt,3);
u(Fs:Fs+10,3) = 0;
cc = CorticalColumn({'sigma',diag([1 1 1]*1e0),'u',u});
corticalColumns = repmat({cc},n,1);

C1 = zeros(n);
C2 = zeros(n);
net1 = unique(randi(n,5,1));
tmp = setdiff(1:n,net1);
net2 = tmp(unique(randi(length(tmp),5,1)))';
%net2(2) = net1(2);
C1(net1,net1) = 0.5;C1(net1,net1) = triu(C1(net1,net1))-tril(C1(net1,net1));
C2(net2,net2) = 0.5;C2(net2,net2) = triu(C2(net2,net2))-tril(C2(net2,net2));

dcm1 = DynamicCausalModel({'nmmArray',corticalColumns,'SchortRange',C1});
dcm2 = DynamicCausalModel({'nmmArray',corticalColumns,'SchortRange',C2});
W1 = dcm1.C;
W2 = dcm2.C;
clear dcm1 dcm2
dcm = DynamicCausalModel({'nmmArray',corticalColumns,'SchortRange',C1});
activation = fliplr(sigmoid(0.05*(-cc.nt/2:cc.nt/2-1)));
activation = activation/max(activation);
%figure;subplot(121);imagesc(C1);subplot(122);imagesc(C2);

pyCell = 1:6:dcm.nx;
xsim = zeros(dcm.nx, dcm.nt);
dcm.x(:) = 0;
for k=2:dcm.nt
    dcm.C = (W1*activation(k) + W2*(1-activation(k)))/2;
    dcm.x = dcm.x + dcm.A*dcm.x+dcm.B*dcm.e;
    xsim(:,k) = dcm.x;
    dcm.tk = k;
end
dcm.plot(xsim(pyCell,:));
x = xsim(pyCell,:);
Cor1 = corr(x(:,activation>0.7)');
Cor2 = corr(x(:,activation<0.3)');
fig = figure;
subplot(321);imagesc(C1);title('Ground truth network 1')
subplot(322);imagesc(Cor1);set(gca,'CLim',[0 1])
title('Functional connectivity estimated by correlation (network 1)')
subplot(323);imagesc(C2);title('Ground truth network 2')
subplot(324);imagesc(Cor1);set(gca,'CLim',[0 1])
title('Functional connectivity estimated by correlation (network 2)')
subplot(325)
[xp,yp,~,auc1] = perfcurve(double(logical(C1(:))),Cor1(:),1);
plot(xp,yp);
[xp,yp,~,auc2] = perfcurve(double(logical(C2(:))),Cor2(:),1);
hold on;
plot(xp,yp);
legend({['Network 1 (AUC=' num2str(auc1) ')'];['Network 2 (AUC=' num2str(auc2) ')']})
title('ROC')
save('x_highly_nonlinear.mat','x','net1','net2','activation','Fs');
savefig(fig,'img_ground_truth_vs_correlation.fig')

%% Simulate supression network
% medial prefrontal cortex (mPFC*), posterior cingulate cortex (PCC*), retrosplenial cortex (ReSp), and bilateral parietal cortex (PC)
Nx = size(hm.K,2);
Adj = geometricTools.getAdjacencyMatrix(hm.cortex.vertices, hm.cortex.faces);
Adj = Adj/max(abs(Adj(:)));
Adj = Adj+speye(Nx);

% FPAN = hm.indices4Structure({...
%     'G_front_sup R','G_front_sup L',...
%     'G_pariet_inf-Angular R',...
%     'G_pariet_inf-Angular L',...
%     'G_parietal_sup R','G_parietal_sup L',...
%     'G_precentral R','G_precentral L'});
FPAN = hm.indices4Structure({...
    'G_front_sup R','G_front_sup L',...
    'G_parietal_sup R','G_parietal_sup L'});
Nroi = size(FPAN,2);
I = triu(ones(Nroi),1)>0;
I = find(I(:));
[indi,indj] = ind2sub([Nroi,Nroi],I);
W = Adj*0;
for i=1:length(indi)
    for j=1:length(indj)
        if indi(i)==indj(j)
            continue;
        end
        W(FPAN(:,indi(i)),FPAN(:,indj(j))) = 1;
        W(FPAN(:,indj(j)),FPAN(:,indi(i))) = -1;
    end
end

cc = CorticalColumn({'sigma',diag([1 1 1]*1e1),'u',0*u});
cc.Clr(1,1) = 0.5;
cc.Csr(cc.Csr~=0) = sign(cc.Csr(cc.Csr~=0))*1;
corticalColumns = repmat({cc},Nx,1);
%dcm = DynamicCausalModel({'nmmArray',corticalColumns,'SchortRange',Adj});
dcm = DynamicCausalModel({'nmmArray',corticalColumns,'SchortRange',Adj,'LongRange',W});
xsim = dcm.simulate;
pyCell = 1:6:dcm.nx;
dcm.plot(xsim(pyCell,:))
% hm.plotOnModel(xsim(pyCell,:));

%%
Dist = zeros(size(cortex.vertices,1));
for k=1:size(cortex.vertices,1)
    ind = find(Csr(k,:));
    d = sqrt(sum(bsxfun(@minus,cortex.vertices(ind,:),cortex.vertices(k,:)).^2,2));
    %Dist(k,ind) = 1./(d+eps)/1./(sum(d)+eps);
    Dist(k,ind) = exp(-0.5*d/sum(d));
end
Clr = Dist;

%%
Nm = size(cortex.vertices,1);
osc = cell(Nm,1);
for k=1:Nm
    osc{k} = NoisyOscillator({'sigma',diag([1e-1 1e-2 1e-2])});
end
nmm = NeuralMassModelNet({'osc',osc,'sr_connections',Csr,'lr_connections',Clr});

%%
xsim = nmm.simulate;
pyCell = 1:6:nmm.nx;
nmm.plot(xsim(pyCell,:))


%%
K = hm.K;
Ny = size(K,1);
H = eye(Ny)-ones(Ny)/Ny;
K = H*hm.K;
x = nmm.x(pyCell);
y = K*x;
hfig = hm.plotOnModel(x,y,'',false,1/nmm.dt);
hfig = hm.plotOnModel(x'.^2,y.^2,'',false,1/nmm.dt);

%%
for i=1:nmm.nm
    for j=1:nmm.nm
        ij = nmm.srConnectivityCellIndices{i,j};
        if ~isempty(ij)
            row = ij(:,1);
            col = ij(:,2);
            nmm.C(row,col) = osc{1}.Csr/7*Dist(i,j); % 7
            nmm.C(col,row) = osc{1}.Csr/7*Dist(j,i);
        end
    end
end
disp('Done')
%%
for i=1:nmm.nm
    for j=1:nmm.nm
        ij = nmm.lrConnectivityCellIndices{i,j};
        if ~isempty(ij)
            row = ij(:,1);
            col = ij(:,2);
            nmm.C(row,col) = osc{1}.Clr/250;
            nmm.C(col,row) = osc{1}.Clr/250;
        end
    end
end
disp('Done')
%%
hfig = hm.plotOnModel(x.^2,y.^2);
set(hfig.hAxes,'Clim',[0 2]);
%set(hfig.hAxes,'Clim',[0 hfig.clim.source(2)]);
colormap(jet(512));
%set(hfig.hAxes,'Clim',[0 hfig.clim.scalp(2)]);

%%
u = zeros(512*10,3);
%u(512:522,2) = 1;
osc = NoisyOscillator({'sigma',diag(0.01*ones(3,1)),'u',u});
osc.simulate;
openField = [1, 5];
osc.plot(openField)

%%
u1 = zeros(512*10,3);
%u1(256:266,3) = 1;
u2 = zeros(512*10,3);
%u2(1024:266,2) = 10;
pyCell = [1,7];
osc1 = NoisyOscillator({'sigma',diag([1e-6 1e-6 1e-6]),'u',u1});
osc2 = NoisyOscillator({'sigma',diag([1e-6 1e-6 1e-6]),'u',u2});

% Random coupling
%osc1.Cij = randn(3)*10;
%osc2.Cij = randn(3)*10;

% Dsync Oscillatory response
%osc1.Cij = osc1.Cij*0;
%osc2.Cij = osc2.Cij*0;

% Sync ERP
osc2.Cij = [40,-15,40;0,0,0;100,-15,100];%[0,0,0;0,0,-22;0,0,0];%[40,-15,40;0,0,0;50,-15,50];
osc1.Cij = [0,0,0;0,0,-22;0,0,0];

% % Dsync ERP
% osc1.Cij = osc1.Cij*0;
% osc2.Cij = osc2.Cij*0;

nmm = NeuralMassModelNet({'nmmArray',{osc1,osc2},'connections',ones(2)});
nmm.simulate;
nmm.plot(pyCell)

%%
x = nmm.x(pyCell);
time = (0:osc1.nt-1)*osc1.dt;
time_loc = time < 1.01;

mx = max(abs(xdsync(:)));
plot(time(time_loc),xdsync(time_loc,:));
xlabel('Time (sec)')
ylabel('Post-synaptic potential (mV)')
title('Desynchronized neural responses')
xlim([min(time(time_loc)) max(time(time_loc))])
ylim(1.001*[-mx mx])
legend({'Pyramidal cell 1','Pyramidal cell 2'})
grid


%% ERPs
figure
% xerp = nmm.x(:,pyCell);
% xerp = bsxfun(@minus,xerp,mean(xerp));
subplot(121)
plot(time(time_loc),xerp(time_loc,:));
xlabel('Time (sec)')
ylabel('Post-synaptic potential (mV)')
title('Desynchronized ERP dynamics')
xlim([min(time(time_loc)) max(time(time_loc))])
ylim(1.125*[min([xerp(:);xserp(:)]) max([xerp(:);xserp(:)])])
legend({'Pyramidal cell 1','Pyramidal cell 2'})
grid


subplot(122)
%xserp = bsxfun(@minus,xserp,mean(xserp));
plot(time(time_loc),xserp(time_loc,:));
xlabel('Time (sec)')
ylabel('Post-synaptic potential (mV)')
title('Synchronised ERP dynamics')
xlim([min(time(time_loc)) max(time(time_loc))])
ylim(1.125*[min([xerp(:);xserp(:)]) max([xerp(:);xserp(:)])])
legend({'Pyramidal cell 1','Pyramidal cell 2'})
grid

%%

figure
xerp = nmm.x(:,pyCell);
mx = max(abs(xerp(:)));
time_loc = time < 1.01;
plot(time(time_loc),x(time_loc,:));
xlim([min(time(time_loc)) max(time(time_loc))])
ylim(1.001*[-mx mx])
legend({'Pyramidal cell 1','Pyramidal cell 2'})
grid


%%
figure
x = nmm.x(:,pyCell);
mx = max(abs(x(:)));
plot(time(time_loc),x(time_loc,:));
xlabel('Time (sec)')
ylabel('Post-synaptic potential (mV)')
title('Synchronized neural responses')
xlim([min(time(time_loc)) max(time(time_loc))])
ylim(1.001*[-mx mx])
legend({'Pyramidal cell 1','Pyramidal cell 2'})
grid