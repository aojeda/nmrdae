classdef NoisyOscillator < handle
        
    properties
        % Parameters of the nonlinear sigmoid function
        e0 = 5;     % s^{-1}
        v0 = 6;     % mV	
        r = 0.56;   % mV^{-1}
        
        % Average synaptic gain for Excitatory nd Inhibitory neurons
        Ae = 3.25;  % mV
        Ai = 22;    % mV
               
        % Average synaptic time constants for Excitatory and Inhibitory populations
        ae = 100;   % s^{-1}
        ai = 50;    % s^{-1}        
        
        
        % Synaptic contacts made by pyramidal cells on excitatory and inhibitory
        % interneurons within a cortical module
        c1 = 150;
        c3 = 40;
        
        % Synaptic contacts made by excitatory and inhibitory interneurons 
        % on pyramidal cells within a cortical module
        c2 = 120;
        c4 = 40
        
        % Synaptic contacts between pyramidal cells within a cortical module
        c5 = 150;
        
        % Synaptic contacts made by pyramidal cells on pyramidal cells of different
        % cortical module corresponding to short and long
        c6 = 100;
        c7 = 100;
        
        % Synaptic contacts made by excitatory and inhibitory interneurons on pyramidal
        % cells of voxels of the same area
        c8 = 100; 
        c9 = 100;
        
        % State variable
        x = [];
        
        % Wiener process
        w = [];
        mu = zeros(1,3);
        sigma = diag(1*ones(3,1));
        
        % Extrinsic input
        u = 0
        
        % Simulation parameters
        nx = 6;
        dt = 1/512; % Integration time step
        tk = 1;     % Time step
        nt = 512*10;
        
        % Short range connectivity
        Csr
        % Long range connectivity
        Clr
    end
    properties(Dependent)
        A
        B
        C
        Fx
        Fz
        z % Algebraic variable
        e % Wiener process
    end
    
    
    methods
        function obj = NoisyOscillator(parameters)
            if nargin > 0,
                for k=1:2:length(parameters)
                    if isprop(obj,parameters{k})
                        obj.(parameters{k}) = parameters{k+1};
                    end
                end
            end
            obj.x = zeros(obj.nt,obj.nx);
            obj.w = mvnrnd(ones(obj.nt,1)*obj.mu,obj.sigma);
            if any(size(obj.u) ~= size(obj.w))
                obj.u = zeros(size(obj.w)) + obj.u;
            end
            obj.w = obj.w+obj.u;
            obj.Csr = sparse([obj.c6, -obj.c9, obj.c8;0 0 0;0 0 0]);
            obj.Clr = sparse([obj.c7, 0, 0;0 0 0;0 0 0]);
        end
        function C = get.C(obj)
            C = sparse([obj.c5, -obj.c4, obj.c2;obj.c3 0 0;obj.c1 0 0]);
        end
        function Fx = get.Fx(obj)
            Fxe = [0 1;-obj.ae^2 -2*obj.ae];
            Fxi = [0 1;-obj.ai^2 -2*obj.ai];
            Fx = kron(spdiags([1 0 1]',0,obj.nx/2,obj.nx/2),Fxe)+kron(spdiags([0 1 0]',0,obj.nx/2,obj.nx/2),Fxi);
        end
        function Fz = get.Fz(obj)
            s = [obj.ae*obj.Ae, obj.ai*obj.Ai, obj.ae*obj.Ae]'.* sigmoid(obj.z);
            Fze = [0 0;0 obj.ae*obj.Ae];
            Fzi = [0 0;0 obj.ai*obj.Ai];
            Fz = kron(spdiags([1 0 1]',0,obj.nx/2,obj.nx/2),Fze)+kron(spdiags([0 1 0]',0,obj.nx/2,obj.nx/2),Fzi);
            for k=1:length(sz)
                Fz(k*2,k*2-1) = s(k);
            end
        end
        function A = get.A(obj)
            Aei = [exp(-obj.ae*obj.dt)*(-obj.ae*obj.dt+1)-1, exp(-obj.ae*obj.dt)*obj.dt;...
                -obj.ae^2*exp(-obj.ae*obj.dt)*obj.dt, exp(-obj.ae*obj.dt)*(1-obj.ae*obj.dt)-1];
            Aii = [exp(-obj.ai*obj.dt)*(-obj.ai*obj.dt+1)-1, exp(-obj.ai*obj.dt)*obj.dt;...
                -obj.ai^2*exp(-obj.ai*obj.dt)*obj.dt, exp(-obj.ai*obj.dt)*(1-obj.ai*obj.dt)-1];
            A = kron(spdiags([1 0 1]',0,obj.nx/2,obj.nx/2),Aei)+kron(spdiags([0 1 0]',0,obj.nx/2,obj.nx/2),Aii);
        end
        function B = get.B(obj)
            Bei= [obj.Ae*exp(-obj.ae*obj.dt)*(-obj.ae*obj.dt+exp(obj.ae*obj.dt)-1)/obj.ae, obj.Ae*exp(-obj.ae*obj.dt)*(obj.ae*obj.dt+exp(obj.ae*obj.dt)*(obj.ae*obj.dt-2)+2)/(obj.ae^2*obj.dt);...
                obj.ae*obj.Ae*exp(-obj.ae*obj.dt)*obj.dt, obj.Ae*exp(-obj.ae*obj.dt)*(-obj.ae*obj.dt+exp(obj.ae*obj.dt)-1)/(obj.ae*obj.dt)];
            Bii= [obj.Ai*exp(-obj.ai*obj.dt)*(-obj.ai*obj.dt+exp(obj.ai*obj.dt)-1)/obj.ai, obj.Ai*exp(-obj.ai*obj.dt)*(obj.ai*obj.dt+exp(obj.ai*obj.dt)*(obj.ai*obj.dt-2)+2)/(obj.ai^2*obj.dt);...
                obj.ai*obj.Ai*exp(-obj.ai*obj.dt)*obj.dt, obj.Ai*exp(-obj.ai*obj.dt)*(-obj.ai*obj.dt+exp(obj.ai*obj.dt)-1)/(obj.ai*obj.dt)];
            B = kron(spdiags([1 0 1]',0,obj.nx/2,obj.nx/2),Bei)+kron(spdiags([0 1 0]',0,obj.nx/2,obj.nx/2),Bii);
        end
        function z = get.z(obj)
            z = obj.C*obj.x(obj.tk,1:2:end)';
        end
        function e = get.e(obj)
            if obj.tk+1 >= obj.nt
                w1 = obj.w(obj.tk,:)';
                z1 = obj.z*0;
            else
                w1 = obj.w(obj.tk+1,:)';
                z1 = obj.C*obj.x(obj.tk+1,1:2:end)';
            end
            w0 = obj.w(obj.tk,:)';
            z0 = obj.z;
            e1 = w1+sigmoid(z0);
            e2 = w1-w0 + z1 + z0.*sigmoid(z0,obj.e0,obj.v0,obj.r,1);
            e = [e1';e2'];
            e = e(:);
            %e = [w1 + sigmoid(z0); w1-w0 + z1 - z0.*sigmoid(z0,obj.e0,obj.v0,obj.r,1)];
        end
        function simulate(obj,x0)
            if nargin < 2, x0 = obj.x(1,:)*0;end
            obj.x(1,:) = x0;
            obj.tk = 1;
            A = obj.A;
            B = obj.B;
            hwait = waitbar(0,'Computing...');
            for tk=1:obj.nt-1
                e = obj.e;
                obj.x(tk+1,:) = obj.x(tk,:) + (A*obj.x(tk,:)')'+(B*e)';
                obj.tk = tk;
                waitbar(tk/(obj.nt-1),hwait);
            end
            close(hwait);
        end
        function plot(obj,plotThis)
            if nargin < 2, plotThis = 1:2:obj.nx;end
            figure;
            subplot(211)
            time = (0:obj.nt-1)*obj.dt;
            plot(time,obj.x(:,plotThis));
            xlabel('Time (sec)');
            grid;
            
            subplot(212)
            Fs = round(1/obj.dt);
            n = length(time(Fs:end-Fs));
            [pxx,freq]=pmtm(obj.x(Fs:end-Fs,1),2.5,n,Fs);
            Pxx = zeros(length(pxx),length(plotThis));
            for k=1:length(plotThis)
                [pxx,freq]=pmtm(obj.x(Fs:end-Fs,plotThis(k)),2.5,n,Fs);
                Pxx(:,k) = pxx;
            end
            loc = freq < 45;
            plot(freq(loc),10*log10(Pxx(loc,:)))
            xlabel('Frequency (Hz)')
            grid
        end
    end
end

%parameters = {'e0',5,'v0',6,'r',0.56,'a',100,'b',50,'at',100,'bt',40,'c1',150,'c3',40,'c2',120,'c4',40,'c5',150,'c6',50,'c7',10,'c7t',100,'c1t',50,'c3t',80};
%osc = NoisyOscillator();
%plot(x(:,1:2:end))
%pmtm(x(250:end,3),10,512,512);