classdef DynamicCausalModel < handle
    properties
        A = sparse(0);
        B = sparse(0);
        C = sparse(0);
        x = [];
        u = [];
        w = [];
        nx
        nt
        dt
        tk = 1;
        nm
        mu = [];
        sigma = 0;
        srConnectivityCellIndices = [];
        lrConnectivityCellIndices = [];
        useGPU = false;
        xbuffer = [];
        order = [];
        e0 = 5;
        v0 = 6;
        r = 0.56;
    end
    properties(Dependent)
        z % Algebraic variable
        e % Wiener process
    end
    methods
        function obj = DynamicCausalModel(varargin)
            if isa(varargin,'cell') && length(varargin) == 1
                varargin = varargin{1};
            end
            ind1 = find(ismember(varargin(1:2:end),'nmmArray'));
            ind2 = find(ismember(varargin(1:2:end),'SchortRange'));
            ind3 = find(ismember(varargin(1:2:end),'LongRange'));
            if ~isempty(ind1) && ~isempty(ind2)
                osc = varargin{ind1+1};
                ShortRange = varargin{ind2*2};
                
                obj.nm = length(osc);
                obj.dt = osc{1}.dt;
                obj.nt = osc{1}.nt;
                obj.nx = 0;
                Csr = cell(obj.nm,1);
                Clr = cell(obj.nm,1);
                for k=1:obj.nm
                    obj.nx = obj.nx+osc{k}.nx;
                    Csr{k} = osc{k}.Csr;
                    Clr{k} = osc{k}.Clr;
                end
                if obj.nm > 500
                    D = spdiags(ones(obj.nm,1),0,obj.nm,obj.nm);
                    obj.A = kron(D,osc{1}.A);
                    obj.B = kron(D,osc{1}.B);
                    C0 = kron(D,osc{1}.C);
                    obj.sigma = kron(D,osc{1}.sigma);
                    obj.u = kron(ones(1,obj.nm),osc{1}.u);
                    obj.mu = kron(ones(1,obj.nm),osc{1}.mu);
                else
                    obj.A = [];
                    obj.B = [];
                    obj.sigma = [];
                    C0 = [];
                    for k=1:obj.nm
                        obj.A = diagmx(obj.A,osc{k}.A);
                        obj.B = diagmx(obj.B,osc{k}.B);
                        obj.sigma = diagmx(obj.sigma,osc{k}.sigma);
                        C0 = diagmx(C0,osc{k}.C);
                        obj.u = [obj.u osc{k}.u];
                        obj.mu = [obj.mu osc{k}.mu];
                    end
                end
                
                obj.C = DynamicCausalModel.buildC(C0, Csr, ShortRange);
                if ~isempty(ind3)
                    LongRange = varargin{ind3*2};
                    obj.C = DynamicCausalModel.buildC(obj.C, Clr, LongRange);
                end
%                 Csr = 0;
%                 for i=1:obj.nm
%                     for j=1:obj.nm
%                         if i==j
%                             continue
%                         elseif connections(i,j)~=0
%                             Sr = zeros(obj.nm);
%                             Sr(i,j) = connections(i,j);
%                             Csr = Csr+kron(Sr,osc{j}.Csr);
%                         end
%                     end
%                 end
%                 ind = find(ismember(varargin(1:2:end),'LongRange'));
%                 
%                 obj.C = C0+Csr;%+Clr;
                obj.x = zeros(obj.nx,1);
                
                obj.w = mvnrnd(ones(obj.nt,1)*obj.mu,obj.sigma);
                obj.w = obj.w+obj.u;
            else
                for k=1:2:length(varargin)
                    if isprop(obj,varargin{k})
                        try
                            obj.(varargin{k}) = varargin{k+1};
                        end
                    end
                end
            end
        end
        function z = get.z(obj)
            z = obj.C*obj.x(1:2:end);
        end
        function e = get.e(obj)
            if obj.tk+1 >= obj.nt
                w1 = obj.w(obj.tk,:)';
                z1 = obj.z*0;
            else
                w1 = obj.w(obj.tk+1,:)';
                z1 = obj.C*obj.x(1:2:end);
            end
            w0 = obj.w(obj.tk,:)';
            z0 = obj.z;
            e1 = w1+sigmoid(z0);
            e2 = w1-w0 + z1 + z0.*sigmoid(z0,obj.e0,obj.v0,obj.r,1);
            e = [e1';e2'];
            e = e(:);
            %e = [w1 + sigmoid(z0); w1-w0 + z1 - z0.*sigmoid(z0,obj.e0,obj.v0,obj.r,1)];
        end
        function xsim = simulate(obj,n, isbatch)
            if nargin < 2, n = obj.nt;end
            if nargin < 3, isbatch = false;end
            xsim = zeros(obj.nx, n);
            if ~isbatch
                hwait = waitbar(0,'Computing...');
                cleaner = onCleanup(@() close(hwait));
            end
            for k=1:n
                xsim(:,k) = obj.predict();
                if ~isbatch, waitbar(k/n,hwait);end
            end
        end
        function xpred = predict(obj)
            obj.tk(obj.tk > obj.nt-1) = 1;
            xpred = obj.x + obj.A*obj.x+obj.B*obj.e;
            if any(isnan(xpred)), error('Computation produced NaN.');end
            obj.x = xpred;
            obj.tk = obj.tk+1;
        end
        function [xfilt,Pfilt,R,G] = update(obj,y,P,H,R,G)
            % Kalman gain
            K = P*H'/(H*P*H'+R);
            
            % Innovation
            e = y-H*obj.x(obj.tk,:);
            
            r = diag(R); %#ok
            g = diag(G);
            beta = median(nonzeros(r));
            alpha = median(nonzeros(g));
            [x_update,alpha,beta] = evidence_approx_tikhonov_reg(e,K,P,alpha,beta,options);
            obj.x(obj.tk,:) = obj.x(obj.tk,:)+x_update;
            xfilt = obj.x(obj.tk+1,:);
            Pfilt = P - K*H*P;
            R = diag(r*0+beta); %#ok
            G = diag(g*0+alpha);
        end
        function update_parameters(obj,Y,H)
            longRangeConnections = 1:3:plant.nx/2;
            winSize = size(Y,2);
            if winSize < length(longRangeConnections)*2-1
                warning('Connectivity parameters need to be estimated with data chunks at least twice longer than the number of pyramidal populations.');
                return
            end
            if obj.tk-winSize < 1, return;end
            pyCell = 1:6:obj.nx;    % Pyramidal populations
            Clr = obj.C(longRangeConnections,longRangeConnections);
            Z = Clr*obj.x(pyCell,obj.tk-winSize:obj.tk);
            Clr = (Z*Z')/(H'*Y*Z')*(H'*H);
            obj.C(longRangeConnections,longRangeConnections) = Clr;
        end
        function plot(obj, xsim)
            figure;
            subplot(211)
            time = (0:size(xsim,2)-1)*obj.dt;
            plot(time,xsim');
            xlabel('Time (sec)');
            ylabel('Post-synaptic potential (mV)')
            title('Time response')
            mx = max(abs(xsim(:)));
            ylim([-mx mx]);
            grid;
            
            subplot(212)
            Fs = round(1/obj.dt);           
            [Pxx,freq] = pwelch(xsim',256,25,Fs);
            freq = freq/pi*Fs/2;
            plot(freq,10*log10(Pxx))
            xlim([1 max(freq)])
            xlabel('Frequency (Hz)')
            ylabel('Power Spectral Density (dB/Hz)')
            title('PSD Estimate')
            grid
        end
        function save2file(obj,filename)
            if nargin < 2,
                error('Need to pass in the name of the file where to save the object.')
            end
            pname = properties(obj);
            s = struct();
            for k=1:length(pname)
                s.(pname{k}) = obj.(pname{k});
            end
            save(filename,'-struct','s');
        end
    end
    methods(Static)
        function v = vect(x), v = x(:);end
        function obj = loadFromFile(filename)
            if ~exist(filename,'file')
                error('File does not exist');
            end
            fileContent = load(filename);
            pnames = fieldnames(fileContent);
            inputParameters = cell(length(pnames),2);
            
            for k=1:length(pnames)
                inputParameters{k,1} = pnames{k};
                inputParameters{k,2} = fileContent.(pnames{k});
            end
            inputParameters = inputParameters';
            inputParameters = inputParameters(:)';
            obj = NeuralMassModelNet(inputParameters(:));
        end
    end
    methods(Static, Hidden)
        function [C,cellIndices] = buildC(C0, Cij, connections)
            cellIndices = cell(size(connections));
            connections = connections - diag(diag(connections));
            C = C0;
            n = length(Cij);
            offsets_c = zeros(n,1);
            offsets_r = zeros(n,1);
            for k=1:n
                [offsets_c(k),offsets_r(k)] = size(Cij{k});
            end
            offsets_c = [1;cumsum(offsets_c(1:end-1))+1];
            offsets_r = [1;cumsum(offsets_r(1:end-1))+1];
            
            if n>1000
                hwait = waitbar(0,'Building the system...');
            end
            Sign = sign(connections);
            for row=1:n
                ind = find(connections(row,:));
                for loc=ind
                    cij = Sign(row,loc)*Cij{loc};
                    [r,c] = size(cij);
                    C(offsets_r(row):offsets_r(row)+r-1,offsets_c(loc):offsets_c(loc)+c-1) = cij;
                    cellIndices{row,loc} = [offsets_r(row):offsets_r(row)+r-1;offsets_c(loc):offsets_c(loc)+c-1]';
                end
                if n>1000
                    waitbar(row/n,hwait);
                end
            end
            if n>1000
                close(hwait);
            end
        end
    end
end