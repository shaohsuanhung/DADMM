classdef TrackingModels
    methods (Static)
        function model = linearize_measurements(anchor, network_topo, env)
            % Fixed affine model h(anchor) + H(anchor)*(x-anchor).
            % Values and Jacobians stay in physical, unwhitened coordinates.
            validateattributes(anchor, {'numeric'}, {'vector','numel',4,'finite'});
            validateattributes(network_topo.numNodes, {'numeric'}, ...
                {'scalar','integer','positive'});
            validateattributes(network_topo.radar_pos, {'numeric'}, ...
                {'size',[network_topo.numNodes,2],'finite'});
            validateattributes(env.lambda, {'numeric'}, {'scalar','positive','finite'});
            model.anchor = anchor(:);
            model.value = zeros(2, network_topo.numNodes);
            model.H = zeros(2, 4, network_topo.numNodes);
            for n = 1:network_topo.numNodes
                [model.value(:,n), model.H(:,:,n)] = ...
                    TrackingModels.MeasureModelAndJacobian(model.anchor, n, network_topo, env);
            end
        end

        function [z, H] = MeasureModelAndJacobian( ...
                state, idx, network_topo, env)
            % Canonical unwhitened global range/Doppler model.
            %
            %   delta = target_position - radar_position
            %   range = ||delta||
            %   fd    = (2/lambda) * velocity' * delta / range
            %
            % State convention: [x; y; vx; vy] in global coordinates.
            position = state(1:2);
            velocity = state(3:4);
            position = position(:);
            velocity = velocity(:);

            radar_position = network_topo.radar_pos(idx, :).';
            delta = position - radar_position;
            range = norm(delta);

            if range <= 1e-12
                error('models:TargetAtRadar', ...
                    'The range/Doppler model is undefined at radar %d.', idx);
            end

            c0 = 2 / env.lambda;
            radial_numerator = velocity.' * delta;
            fd = c0 * radial_numerator / range;

            z = [range; fd];

            dr_dp = delta.' / range;
            dfd_dp = c0 * (velocity.' / range - ...
                radial_numerator * delta.' / range^3);
            dfd_dv = c0 * delta.' / range;

            H = [dr_dp, zeros(1, 2); ...
                 dfd_dp, dfd_dv];
        end
    end
end
