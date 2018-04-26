%%
clear
clc
addpath ~/Projects/eeglab;eeglab;
hm = headModel.loadFromFile(which('head_modelColin27_5003_Standard-10-5-Cap339-Destrieux148.mat'));
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
cc = CorticalColumn({'sigma',diag([1 1 1]*1e2),'u',u1});

n = 100;
corticalColumns = repmat({cc},n,1);
dcm = DynamicCausalModel({'nmmArray',corticalColumns,'SchortRange',(triu(rand(n))-tril(rand(n)))/n});
pyCell = 1:6:dcm.nx;
xsim = dcm.simulate;
dcm.plot(xsim(pyCell,:))

%%
n = size(hm.K,2);
Adj = geometricTools.getAdjacencyMatrix(hm.cortex.vertices, hm.cortex.faces);
corticalColumns = repmat({cc},n,1);
dcm = DynamicCausalModel({'nmmArray',corticalColumns,'SchortRange',Adj/n});

%cortex = hm.cortex;
%Csr = geometricTools.getAdjacencyMatrix(cortex.vertices,cortex.faces);

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