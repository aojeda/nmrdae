classdef NeuralMassModelNet < handle
    properties
        A = [];
        B = [];
        C = [];
        x = [];
        w = [];
        nx
        nt
        dt
        tk = 1;
        nm
        mu = 0;
        sigma = 1e-2;
        srConnectivityCellIndices = [];
        lrConnectivityCellIndices = [];
        useGPU = false;
        xbuffer = [];
        order = [];
    %end
    %properties(Constant)
        e0 = 5;
        v0 = 6;
        r = 0.56;
    end
    properties(Dependent)
        z % Algebraic variable
        e % Wiener process
    end
    methods
        function obj = NeuralMassModelNet(varargin)
            if isa(varargin,'cell') && length(varargin) == 1
                varargin = varargin{1};
            end
            ind1 = find(ismember(varargin(1:2:end),'osc'));
            ind2 = find(ismember(varargin(1:2:end),'sr_connections'));
            if ~isempty(ind1) && ~isempty(ind2)
                osc = varargin{ind1+1};
                sr_connections = varargin{ind2*2};
                obj.nm = size(sr_connections,1);
                obj.dt = osc{1}.dt;
                obj.nt = osc{1}.nt;
                obj.nx = 0;
                for k=1:length(osc)
                    obj.nx = obj.nx+osc{k}.nx;
                end
                obj.x = zeros(obj.nx,1);
                obj.A = kron(speye(obj.nm),osc.A);
                obj.B = kron(speye(obj.nm),osc.B);
                C0 = kron(speye(obj.nm),osc.C);
                Csr = kron(speye(obj.nm),osc.Csr);
                Clr = kron(speye(obj.nm),osc.Clr);
                obj.C = C0+Csr+Clr;
                obj.mu = mean(osc.mu);
                obj.sigma = mean(diag(osc.sigma));
                ind = find(ismember(varargin(1:2:end),'useGPU'));
                if isempty(ind)
                    obj.useGPU = false;
                else
                    obj.useGPU = varargin{ind*2};
                end
            else
                for k=1:2:length(varargin)
                    if isprop(obj,varargin{k})
                        try
                            obj.(varargin{k}) = varargin{k+1};
                        end
                    end
                end
            end
            if obj.useGPU
                try obj.useGPU = gpuDeviceCount > 0;end
            end
            if obj.useGPU
                try
                    obj.A = gpuArray(obj.A);
                    obj.B = gpuArray(obj.B);
                    obj.C = gpuArray(obj.C);
                    obj.x = gpuArray(obj.x);
                    obj.w = gpuArray(obj.w);
                catch ME
                    disp(ME)
                    disp('We will use CPU instead.')
                    if isa(obj.A,'gpuArray'), obj.A = sparse(gather(obj.A));end
                    if isa(obj.B,'gpuArray'), obj.B = sparse(gather(obj.B));end
                    if isa(obj.C,'gpuArray'), obj.C = sparse(gather(obj.C));end
                    if isa(obj.x,'gpuArray'), obj.x = sparse(gather(obj.x));end
                    if isa(obj.w,'gpuArray'), obj.w = sparse(gather(obj.w));end
                end
            end
        end
        function w = get.w(obj)
            w = obj.sigma.*randn(obj.nm*3,1)+obj.mu;
        end
        function z = get.z(obj)
            if isempty(obj.order)
                obj.order = size(obj.C,2)/(obj.nx/2);
            elseif obj.order ~= size(obj.C,2)/(obj.nx/2)
                obj.order = size(obj.C,2)/(obj.nx/2);
                obj.xbuffer = [];
            end
            if isempty(obj.xbuffer)
                obj.xbuffer = zeros(obj.nx/2,obj.order);
                if isa(obj.x, 'gpuArray')
                    obj.xbuffer = gpuArray(obj.xbuffer);
                end
            end
            %p = size(obj.C,2)/(obj.nx/2);
            %xpast = flipud(obj.x(obj.tk-p+1:obj.tk,1:2:end))';
            %z = obj.C*xpast(:);
            z = obj.C*obj.xbuffer(:);
        end
        function e = get.e(obj)
            if obj.tk+1 >= obj.nt
                %w1 = obj.w(obj.tk,:)';
                z1 = obj.z;
            else
                %w1 = obj.w(obj.tk+1,:)';
                obj.tk = obj.tk+1;
                z1 = obj.z;
                obj.tk = obj.tk-1;
            end
            z0 = obj.z;
            z1 = obj.C*[0*obj.x(1:2:end);obj.vect(obj.xbuffer(:,1:end-1))];
            % z1 = obj.z;
            w0 = obj.w;
            w1 = obj.w;
            %w0 = obj.w(obj.tk,:)';
            
            e1 = w1+sigmoid(z0);
            e2 = w1-w0 + z1 + z0.*sigmoid(z0,obj.e0,obj.v0,obj.r,1);
            % e = [e1';e2'];
            e = [e1,e2]';
            e = e(:);
        end
        function xsim = simulate(obj,n, isbatch)
            if nargin < 2, n = obj.nt;end
            if nargin < 3, isbatch = false;end
            xsim = zeros(obj.nx, n);
            if isa(obj.x,'gpuArray'), xsim = gpuArray(xsim);end
            %p = size(obj.C,2)/(obj.nx/2);
            %obj.x(1:p,:) = ones(p,1)*x0;
            if ~isbatch
                hwait = waitbar(0,'Computing...');
                cleaner = onCleanup(@() close(hwait));
            end
            %obj.tk = p;
            for k=1:n
                xsim(:,k) = obj.predict();
                if ~isbatch, waitbar(k/n,hwait);end
            end
        end
        function xpred = predict(obj)
            if obj.tk > obj.nt-1,obj.tk = 1;end
            xpred = obj.x + obj.A*obj.x+obj.B*obj.e;
            % xpred = obj.x(obj.tk,:) + (obj.A*obj.x(obj.tk,:)')'+(obj.B*obj.e)';
            if any(isnan(xpred)), error('Computation produced NaNs.');end
            obj.x = xpred;
            obj.update_xbuffer();
            %obj.x(obj.tk+1,:) = xpred;
            %obj.tk = obj.tk+1;
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
            subplot(121)
            time = (0:size(xsim,2)-1)*obj.dt;
            plot(time,xsim');
            xlabel('Time (sec)');
            ylabel('Post-synaptic potential (mV)')
            title('Time response')
            mx = max(abs(xsim(:)));
            if obj.useGPU, mx = gather(mx);end
            ylim([-mx mx]);
            grid;
            
            subplot(122)
            Fs = round(1/obj.dt);
            n = length(time(Fs:end));
            x = xsim(:,2*Fs:end-Fs)';
            if obj.useGPU, x = gather(x);end
            try
                [Pxx,freq]=pmtm(x,1.5,n,Fs);
            catch
                [pxx,freq]=pmtm(x(:,1),1.5,n,Fs);
                nx = size(xsim,1);
                Pxx = zeros(length(pxx),nx);
                for k=1:nx
                    [pxx,freq]=pmtm(x(:,k),1.5,n,Fs);
                    Pxx(:,k) = pxx;
                end
            end
            loc = freq < 128;
            plot(freq(loc),10*log10(Pxx(loc,:)))
            xlim([min(freq(loc)) max(freq(loc))])
            xlabel('Frequency (Hz)')
            ylabel('Power Spectral Density (dB/Hz)')
            title('Thompson Multitaper Power Spectral Density Estimate')
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
    methods(Hidden)
        function update_xbuffer(obj)
            obj.xbuffer(:,2:end) = obj.xbuffer(:,1:end-1);
            obj.xbuffer(:,1) = obj.x(1:2:end);
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
            for k=1:length(Cij)
                [offsets_c(k),offsets_r(k)] = size(Cij{k});
            end
            offsets_c = [1;cumsum(offsets_c(1:end-1))+1];
            offsets_r = [1;cumsum(offsets_r(1:end-1))+1];
            
            for row=1:size(connections,1)
                ind = find(connections(row,:));
                for loc=ind
                    cij = Cij{loc};
                    [r,c] = size(cij);
                    C(offsets_r(row):offsets_r(row)+r-1,offsets_c(loc):offsets_c(loc)+c-1) = cij;
                    cellIndices{row,loc} = [offsets_r(row):offsets_r(row)+r-1;offsets_c(loc):offsets_c(loc)+c-1]';
                end
            end
        end
    end
end