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
ROI = hm.atlas.label;
dACC = find(ismember(hm.atlas.label,{'G_and_S_cingul-Mid-Ant R','G_and_S_cingul-Mid-Ant L'}));
PCC = find(ismember(hm.atlas.label,{'G_cingul-Post-dorsal R','G_cingul-Post-dorsal L'}));
MPFC_L = find(ismember(hm.atlas.label,{'G_front_middle L','S_front_middle L'}));
MPFC_R = find(ismember(hm.atlas.label,{'G_front_middle R','S_front_middle R'}));
ROI([dACC(2) PCC(2) MPFC_L(2) MPFC_R(2)]) = [];
P = real(hm.indices4Structure(hm.atlas.label))';
P(dACC(1),:) = sum(P(dACC,:));
P(PCC(1),:) = sum(P(PCC,:));
P(MPFC_L(1),:) = sum(P(MPFC_L,:));
P(MPFC_R(1),:) = sum(P(MPFC_R,:));
P([dACC(2) PCC(2) MPFC_L(2) MPFC_R(2)],:) = [];
P = bsxfun(@rdivide,P,eps+sum(P,2));

xroi = P*x;

ind = any(abs(xroi)>0.4,2);
xroi(ind,:) = [];
ROI(ind) = [];

ROI{ismember(ROI,'G_and_S_cingul-Mid-Ant L')} = 'dACC';
ROI{ismember(ROI,'G_cingul-Post-dorsal L')} = 'PCC';
net1 = find(ismember(ROI,{'dACC','G_front_inf-Opercular L','G_front_inf-Opercular R','G_insular_short L','G_insular_short R'}));
net2 = find(ismember(ROI,{'PCC','G_front_middle L','G_front_middle R'}));

net1([3 5]) = [];

ROI2rm = {...
    'G_Ins_lg_and_S_cent_ins L','G_Ins_lg_and_S_cent_ins R',...
    'G_and_S_transv_frontopol L','G_and_S_transv_frontopol R',...
    'G_subcallosal L','G_subcallosal R',...
    'G_temp_sup-G_T_transv L','G_temp_sup-G_T_transv R',...
    'G_temp_sup-Lateral L','G_temp_sup-Lateral R',...
    'G_temp_sup-Plan_polar L','G_temp_sup-Plan_polar R',...
    'G_temp_sup-Plan_tempo L','G_temp_sup-Plan_tempo R',...
    'Lat_Fis-ant-Horizont L','Lat_Fis-ant-Horizont R',...
    'Lat_Fis-ant-Vertical L','Lat_Fis-ant-Vertical R',...
    'S_interm_prim-Jensen L','S_interm_prim-Jensen R',...
    'S_intrapariet_and_P_trans L','S_intrapariet_and_P_trans R',...
    'S_oc_middle_and_Lunatus L','S_oc_middle_and_Lunatus R',...
    'S_orbital-H_Shaped L','S_orbital-H_Shaped R',...
    'S_temporal_transverse L','S_temporal_transverse R'};


plot(t-0.5,xroi','color',[0.75 0.75 0.75])
hold on;
plot(t-0.5,xroi(net1,:)','b');
plot(t-0.5,xroi(net2,:)','r');
xlim([0 8])
xlabel('Time (sec)');ylabel('Estimated cortical activity')
grid on
lg = legend({'AON ROIs','DMN ROIs'})
save('x_139_roi.mat','xroi','net1','net2','Fs','t','ROI','ROI2rm');

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