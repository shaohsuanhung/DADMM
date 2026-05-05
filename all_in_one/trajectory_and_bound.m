function plot_trajectory_and_network(true_trajectory, estimated_trajectory,network_topo,cov_xy)
            % Shape of the inputs:
            % true_trajectory: [Num target, track_time, 2]
            % estimated_trajectory: cell(track_time): [4 x 1]
            estimated_trajectory = cell2mat(estimated_trajectory);
            fig = figure;
            set(gcf,'Color','white');
            set(gca,'FontSize',30);
            hold on;
            plot(true_trajectory(1, :, 1), true_trajectory(1, :, 2), '--ok', 'LineWidth', 0.1, 'DisplayName', 'True Trajectory');
            plot(network_topo.radar_pos(:,1), network_topo.radar_pos(:,2), 'r.', 'MarkerSize', 50, 'DisplayName', 'Sensor Nodes');
            % plot(true_trajectory(1, 1:size(true_trajectory,2)-64, 1),true_trajectory(1, 1:size(true_trajectory,2)-64, 2), '-r', 'LineWidth', 1, 'DisplayName', 'Ground truth location');
            plot(estimated_trajectory(1,:), estimated_trajectory(2,:), '--ob', 'LineWidth', 2, 'DisplayName', 'Estimated Trajectory');
            
            % % covariance ellipses (k-sigma)
                hLine = gca; %#ok<NASGU>
                % use the last line's color
                lines = findobj(gca,'Type','Line');
                if ~isempty(lines)
                    c = lines(1).Color;
                else
                    c = [0 0 0];
                end
                mu_xy= [estimated_trajectory(1,:); estimated_trajectory(2,:)]
                T = size(mu_xy,2);
                for t = 1:1:T
                    C = cov_xy(1,t);
                    [ex, ey] = cov_ellipse(mu_xy(:,t), C, 3, 60);
                    fill(ex, ey, c, 'FaceAlpha', 1, 'EdgeColor', 'black', 'HandleVisibility', 'off');
                end
                uistack(findobj(gca,'Type','Line','-depth',1), 'top');

            % Plot communication link
            for n = 1:network_topo.numNodes 
                neighbors_idx = find(network_topo.laplacian_matrix(n,:) == -1).'; 
                % pairs = nchoosek(neighbors_idx,2);
                for j = 1: size(neighbors_idx,1)
                     if (n == 1 & j == 1)
                         plot([network_topo.radar_pos(n,1),network_topo.radar_pos(neighbors_idx(j),1)],...
                         [network_topo.radar_pos(n,2),network_topo.radar_pos(neighbors_idx(j),2)],...
                         '--k','LineWidth',1.5,'DisplayName','Communication link');
                     end
                     plot([network_topo.radar_pos(n,1),network_topo.radar_pos(neighbors_idx(j),1)],...
                         [network_topo.radar_pos(n,2),network_topo.radar_pos(neighbors_idx(j),2)],...
                         '--k','LineWidth',1.5);
                end
            end

            hold off;
            xlabel('Position x (m)');
            ylabel('Position y (m)');
            % title('Target Trajectory');
            % legend('Location', 'best');
            objs = findobj(gca, '-property', 'DisplayName');
            objs = objs(arrayfun(@(h) ~isempty(h.DisplayName), objs));  
            legend(flipud(objs), 'Location', 'bestoutside');  
            grid on;box on;ax=gca;ax.LineWidth=1.5;
end
function [x, y] = cov_ellipse(mu, C, k, nPts)
% Points for ellipse: (p-mu)' inv(C) (p-mu) = k^2
% mu = mu(:);
C = C{1};
C = (C + C.')/2;

[V, D] = eig(C);
d = max(diag(D), 0);
A = V * diag(sqrt(d));

th = linspace(0, 2*pi, nPts);
circ = [cos(th); sin(th)];
ell = mu + k * (A * circ);

x = ell(1,:); y = ell(2,:);
end

clc;clear;close all;
S = load("./data_log/MAP_mc2/log.mat");
network_topo = S.log.network_topo;
estimated_trajectory = S.log.Results.consensus_estimates;
estimated_trajectory = estimated_trajectory(1,:);
true_trajectory = S.log.target.target_position;
cov_xy = S.log.Results.node_wise_cov;
plot_trajectory_and_network(true_trajectory, estimated_trajectory,network_topo,cov_xy)