%%
clear
clc
addpath ~/Projects/eeglab;eeglab;
hm = headModel.loadDefault;
close all

%% Simulate supression network
% medial prefrontal cortex (mPFC*), posterior cingulate cortex (PCC*), retrosplenial cortex (ReSp), and bilateral parietal cortex (PC)
Nx = size(hm.K,2);
Adj = geometricTools.getAdjacencyMatrix(hm.cortex.vertices, hm.cortex.faces);
Adj = Adj/max(abs(Adj(:)));
Adj = Adj+speye(Nx);

N1 = hm.indices4Structure({...
    'G_and_S_cingul-Mid-Ant L','G_and_S_cingul-Mid-Ant R',...
    'G_front_inf-Opercular L','G_front_inf-Opercular R',...
    'G_insular_short L','G_insular_short R'});
Nroi = size(N1,2);
I = triu(ones(Nroi),1)>0;
I = find(I(:));
[indi,indj] = ind2sub([Nroi,Nroi],I);
C1 = Adj*0;
for i=1:length(indi)
    for j=1:length(indj)
        if indi(i)==indj(j), continue;end
        C1(N1(:,indi(i)),N1(:,indj(j))) = 1;
        C1(N1(:,indj(j)),N1(:,indi(i))) = -1;
    end
end


N2 = hm.indices4Structure({...
    'G_front_middle R','G_front_middle L',...
    'G_cingul-Post-dorsal R','G_cingul-Post-dorsal L'});
Nroi = size(N2,2);
I = triu(ones(Nroi),1)>0;
I = find(I(:));
[indi,indj] = ind2sub([Nroi,Nroi],I);
C2 = Adj*0;
for i=1:length(indi)
    for j=1:length(indj)
        if indi(i)==indj(j), continue;end
        C2(N2(:,indi(i)),N2(:,indj(j))) = 1;
        C2(N2(:,indj(j)),N2(:,indi(i))) = -1;
    end
end
Fs = 256;
cc = CorticalColumn({'dt',1/Fs,'nt',Fs*11,'sigma',diag([1 1 1]*1e0)});
cc.Clr(1,1) = 0.75;
cc.Csr(cc.Csr~=0) = sign(cc.Csr(cc.Csr~=0))*1;
corticalColumns = repmat({cc},Nx,1);

dcm = DynamicCausalModel({'nmmArray',corticalColumns,'SchortRange',Adj,'LongRange',C1});
tmp = DynamicCausalModel({'nmmArray',corticalColumns,'SchortRange',Adj,'LongRange',C2});
W1 = dcm.C;
W2 = tmp.C;
%%
clear tmp

activation = fliplr(sigmoid(0.05*(-cc.nt/2:cc.nt/2-1)));
activation = activation/max(activation);
a = activation'*0.65+0.35;
b = 1-(activation'*0.65);

pyCell = 1:6:dcm.nx;
xsim = zeros(dcm.nx, dcm.nt);
dcm.x(:) = 0;
hwait = waitbar(0,'Computing...');
for k=2:dcm.nt
    % dcm.C = W1*a(k)*1.075 + W2*(1-b(k))*0.9;
    dcm.C = W1*a(k)*0.78 + W2*(b(k))*0.7;
    dcm.x = dcm.x + dcm.A*dcm.x+dcm.B*dcm.e;
    xsim(:,k) = dcm.x;
    dcm.tk = k;
    waitbar(k/dcm.nt,hwait);
end
close(hwait);
dcm.plot(xsim(pyCell,:))
% hm.plotOnModel(xsim(pyCell,:));

%% EEG simulation and inverse solution
t = (0:dcm.nt-1)/Fs;
loc = t>0.5 & t<10.5;
nt = sum(loc);
t = (0:nt-1)/Fs;

hm = headModel.loadDefault;
norm_K = norm(hm.K);
hm.K = hm.K/norm_K;
hm.L = hm.L/norm_K;
hm.K = bsxfun(@rdivide,hm.K,std(hm.K,[],1));
snr = 4;
y = hm.K*xsim(pyCell,loc);
y = awgn(y,snr,'measured');

solver = bsbl(hm);
x = xsim(pyCell,loc)*0;
winSize = 24;
hwait = waitbar(0,'Computing...');
for k=1:winSize:nt
    ind = k:k+winSize-1;
    ind(ind>nt) = [];
    if isempty(ind), break;end
    x(:,ind) = solver.update(y(:,ind));
    waitbar(k/nt,hwait);
end
close(hwait);
x = filtfilt(filterDesign(Fs,30,35),1,x')';
hm.plotOnModel(x);
%%
dACC = find(ismember(hm.atlas.label,{'G_and_S_cingul-Mid-Ant R','G_and_S_cingul-Mid-Ant L','G_and_S_cingul-Ant L','G_and_S_cingul-Ant R'}));
AI_R = find(ismember(hm.atlas.label,{'G_front_inf-Opercular R', 'G_front_inf-Triangul R', 'G_insular_short R','G_front_inf-Orbital R'}));
AI_L = find(ismember(hm.atlas.label,{'G_front_inf-Opercular L', 'G_front_inf-Triangul L', 'G_insular_short L','G_front_inf-Orbital L'}));

P = [any(hm.indices4Structure(hm.atlas.label(dACC)),2) any(hm.indices4Structure(hm.atlas.label(AI_L)),2) any(hm.indices4Structure(hm.atlas.label(AI_R)),2)];

PCC = find(ismember(hm.atlas.label,{'G_cingul-Post-dorsal R','G_cingul-Post-dorsal L','G_cingul-Post-ventral L','G_cingul-Post-ventral L','S_subparietal R','S_subparietal L','G_and_S_cingul-Mid-Post L','G_and_S_cingul-Mid-Post R'}));
MPFC_L = find(ismember(hm.atlas.label,{'G_front_middle L','S_front_middle L'}));
MPFC_R = find(ismember(hm.atlas.label,{'G_front_middle R','S_front_middle R'}));

P = [P any(hm.indices4Structure(hm.atlas.label(PCC)),2) any(hm.indices4Structure(hm.atlas.label(MPFC_L)),2) any(hm.indices4Structure(hm.atlas.label(MPFC_R)),2)];

ROI = cat(2,{'dACC','IOC_L','IOC_R','PCC','MPFC_L','MPFC_R'},setdiff(hm.atlas.label,{...
    'G_and_S_cingul-Mid-Ant R','G_and_S_cingul-Mid-Ant L','G_and_S_cingul-Ant L','G_and_S_cingul-Ant R',...
    'G_front_inf-Opercular R', 'G_front_inf-Triangul R', 'G_insular_short R','G_front_inf-Orbital R',...
    'G_front_inf-Opercular L', 'G_front_inf-Triangul L', 'G_insular_short L','G_front_inf-Orbital L',...
    'G_cingul-Post-dorsal R','G_cingul-Post-dorsal L','G_cingul-Post-ventral L','G_cingul-Post-ventral L','S_subparietal R','S_subparietal L','G_and_S_cingul-Mid-Post L','G_and_S_cingul-Mid-Post R',...
    'G_front_middle L','S_front_middle L','G_front_middle R','S_front_middle R'}));
    
P = [P hm.indices4Structure(ROI(7:end))];
% P = bsxfun(@rdivide,P,eps+sum(P,1))';
% xroi = P*x;

xroi  = zeros(size(P,2),size(x,2));
xroi_power  = zeros(size(P,2),size(x,2));
for k=1:size(P,2)
    ind = find(P(:,k));
    xroi(k,:) = mean(x(ind,:));
    xroi_power(k,:) = mean(abs(x(ind,:)));
end
xroi(7:end,:) = xroi(7:end,:)*0.3;
xroi_power(7:end,:) = xroi_power(7:end,:)*0.35;
net1 = [1 2 3];
net2 = [4 5 6];
xroi_power(net1,:) = xroi_power(net1,:)*std(std(xroi(net2,:)))/std(std(xroi(net1,:)));
xroi(net1,:) = xroi(net1,:)*std(std(xroi(net2,:)))/std(std(xroi(net1,:)));
bCoeff = ones(4,1);bCoeff = bCoeff/sum(bCoeff);
xroi_power = filtfilt(bCoeff,1,xroi_power')';

save('x_131_roi.mat','xroi','xroi_power','net1','net2','Fs','t','ROI','');

%%
color = bipolar(12);
color = color([1:3 end-length(net2):end],:);
indNoNet = setdiff(1:length(ROI),[net1 net2]);

figure;
plot(t-0.5,xroi(indNoNet,:)','color',[0.75 0.75 0.75])
hold on;
for k=1:length(net1)
    plot(t-0.5,xroi(net1(k),:)','Color',color(k,:),'LineWidth',1);
end
for k=1:length(net2)
    plot(t-0.5,xroi(net2(k),:)','Color',color(k+length(net2),:),'LineWidth',1);
end
title('Potential')
xlim([0 8])
xlabel('Time (sec)');ylabel('Estimated cortical activity power averaged by ROI')
grid on

figure
plot(t-0.5,xroi_power(indNoNet,:)','color',[0.75 0.75 0.75],'LineWidth',0.5)
hold on;
for k=1:length(net1)
    plot(t-0.5,xroi_power(net1(k),:)','Color',color(k,:),'LineWidth',1);
end
for k=1:length(net2)
    plot(t-0.5,xroi_power(net2(k),:)','Color',color(k+length(net2),:),'LineWidth',1);
end
xlim([0 8])
ylim([0 .23])
xlabel('Time (sec)');
ylabel('Root mean square power by ROI')
grid on

%lg = legend({'AON ROIs','DMN ROIs'})

%%
Corr = corr(xroi_power(net1,t<5)',xroi_power(net1,t<5)');
Corr = corr(xroi_power(:,t<5)',xroi_power(:,t<5)');

%%
fig = figure;
Con1 = zeros(length(ROI));
Con1(net1,net1) = 1;
Con2 = zeros(length(ROI));
Con2(net2,net2) = 1;
subplot(121);imagesc(Con1);title('Ground truth network 1')
%subplot(322);imagesc(Cor1);set(gca,'CLim',[0 1])
%title('Functional connectivity estimated by correlation (network 1)')
subplot(122);imagesc(Con2);title('Ground truth network 2')
%subplot(324);imagesc(Cor1);set(gca,'CLim',[0 1])
%title('Functional connectivity estimated by correlation (network 2)')