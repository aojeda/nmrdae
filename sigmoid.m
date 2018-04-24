function s = sigmoid(v,e0,v0,r,order)
% The sigmoid function is defined in [1], where 2e0 is the maximum firing
% rate, v0 is the post synaptic potential corrsponding to the firing rate
% e0, and the parameter r controls the stepness of the sigmoid.
%
% [1] David, O., Harrison, L., & Friston, K. (2005). Modelling event-related
%     responses in the brain. Neuroimage, 25(3), 756?770.
% 
% Author: Alejandro Ojeda June 2013

if nargin < 1, error('Not enough input arguments.');end
if nargin < 2, e0 = 5/2;end    % 1/sec
if nargin < 3, v0 = 6;end    % mV
if nargin < 4, r  = 0.56;end % 1/mV
if nargin < 5, order  = 0;end
e = exp(r*(v0-v));
if order
    s = 2*e0./(1+e).^2*r.*e;
else
    s = 2*e0./(1+e);
end