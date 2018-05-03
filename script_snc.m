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
    'G_front_middle R','G_front_middle L',...
    'G_parietal_sup R','G_parietal_sup L'});
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
    'G_parietal_sup R','G_parietal_sup L',...
    'G_occipital_middle R','G_occipital_middle L',...
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
cc.Clr(1,1) = 0.5;
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

pyCell = 1:6:dcm.nx;
xsim = zeros(dcm.nx, dcm.nt);
dcm.x(:) = 0;
hwait = waitbar(0,'Computing...');
for k=2:dcm.nt
    dcm.C = W1*activation(k) + W2*(1-activation(k))*4/4.4;
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
hm.plotOnModel(x);