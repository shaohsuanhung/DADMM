classdef make_figs
    properties (Constant)
        labels_params = {'Position x ', 'Position y ', 'Velocity x ', 'Velocity y '};
    end
    
    properties 
        colors;  
    end
    methods 
        function obj = make_figs(numNodes)
            obj.colors = lines(numNodes); % MATLAB function that provides distinct colors
        end
        function plot_converge_across_node(obj,all_estimations_every_iter,true_params,network_topo)
            figure;
            set(gcf,'Color','white');
            set(gca,'FontSize',24);
            for param = 1:4
                subplot(2, 2, param);
                hold on;  % Allows multiple plots on the same axes
            
                % Plot estimations for each node
                for node = 1:network_topo.numNodes
                    plot(squeeze(all_estimations_every_iter(param, node, :)), 'Color', obj.colors(node, :));
                end
            
                % Plot true parameter as a dotted line
                h_true = yline(true_params(param), '--k', 'LineWidth', 1.5);
            
                hold off;
                xlabel('Iterations (k)');
                ylabel(['Parameter ' obj.labels_params{param} 'estimates']);
                title([obj.labels_params{param}]);
                legend_entries = arrayfun(@(x) ['Node ' num2str(x)], 1:network_topo.numNodes, 'UniformOutput', false);
                legend_entries{end+1} = 'True Parameter';
                legend([legend_entries], 'Location', 'northeastoutside');
            end
            sgtitle('Parameters of interest (\theta) Estimations');
        end 

        function plot_dual_primal_residual(obj,dual_residual_all,primal_residual_CR, all_estimations_every_iter_CR,laplacian_matrix_CR,network_topo)
            % Dual Residual Convergence
            figure;
            semilogy(dual_residual_all);
            xlabel("Iterations (k)");
            % ylabel('|| \nu(k+1) - \nu(k)||_2^2');
            ylabel('$\sum_{n=1}^{N} \sum_{j \in \mathrm{Neighbors}(n)} || \nu_{n|j}(k+1) - \nu{n|j}(k)||_2^2$', 'Interpreter', 'latex');
            title('Dual Residual (s(k))');
            
            % Primal and Dual Convergence for various node ratio
            numColors = length(all_estimations_every_iter_CR);  % Define number of distinct colors needed
            colors = hsv(numColors);  % Creates a colormap with numColors distinct colors
            
            display_node = 1;
            legendNames = cell(1, length(primal_residual_CR)); 
            for i = 1:length(primal_residual_CR)
                legendNames{i} = ['Neighbors: ', num2str(laplacian_matrix_CR{i}(display_node, display_node)), '/', num2str(network_topo.numNodes)];
            end
            
            % Plot Primal Residual Convergence for various node ratios
            figure;
            for CR = 1:numColors
                semilogy(primal_residual_CR{CR}, 'Color', colors(CR, :));
                hold on;
            end
            xlabel('Iterations (k)');
            ylabel('$\sum_{n=1}^{N} \sum_{j \in \mathrm{Neighbors}(n)} ||\theta_m(k+1) - \vartheta_{nj}(k+1)||_2^2$', 'Interpreter', 'latex');
            title('Primal Residual (r(k)) for different node ratios');
            legend(legendNames, 'Location', 'best');  % Add legend with node ratios

        end

        function plot_specific_node_converg(obj,nodeID,com_rad_CR,laplacian_matrix_CR,all_estimations_every_iter_CR,estimated_params_CA, network_topo,true_params)
            % Plot Convergance of theta for various node ratios
            display_node = nodeID;
            legendNames = cell(1, length(com_rad_CR)); 
            
            % Construct the Node Ratio as a fraction in the format 'numerator/denominator'
            for i = 1:length(com_rad_CR)
                legendNames{i} = ['Neighbors: ', num2str(laplacian_matrix_CR{i}(display_node, display_node)), '/', num2str(network_topo.numNodes)];
            end

            figure;
            for param = 1:4
                subplot(2,2,param);
                hold on;
            
                % Initialize a cell array to store plot handles
                hPlots = cell(1, length(all_estimations_every_iter_CR) + 2); % +2 for true parameter and centralized approach
            
                % Plot all estimations with specific color and store handles
                for j = 1:length(all_estimations_every_iter_CR)
                    hPlots{j} = plot(squeeze(all_estimations_every_iter_CR{j}(param, display_node, :)), 'Color', obj.colors(j, :));        
                end
            
                % Plot the centralized approach as a yline
                hPlots{end-1} = yline(estimated_params_CA(param), 'k', 'LineWidth', 1.5, 'DisplayName', 'Centralized Approach');
            
                % Plot the true parameter line and store the handle
                hPlots{end} = yline(true_params(param), '--k', 'LineWidth', 1.5, 'DisplayName', 'True Parameter');
            
                % Add the legend
                legend([hPlots{:}], [legendNames, {'Centralized Approach', 'True Parameter'}]);
            
                hold off;
                xlabel('Iterations (k)');
                ylabel(['Parameter ' obj.labels_params{param} ' estimates']);
                title([obj.labels_params{param}]);
            
                hold off;
            
            
            end
            
            sgtitle({'Convergence of parameters of interest ($\theta_n$) at $n^{\mathrm{th}}$ node for different communication radius'}, 'Interpreter', 'latex');
            % sgtitle({'Convergence of parameters of interest ($\theta_n$) at',nodeID ,'th node for different communication radius'}, 'Interpreter', 'latex');
        end

        function plot_sepcific_node_error_converg(obj,all_estimations_every_iter_mc,estimates_mc_CA,direction_mc,true_params_mc)
            
            colors_size = size(direction_mc, 1);
            colors = hsv(colors_size);
            legendNames = cell(1, size(direction_mc, 1)); 
            
            % Construct the legend names based on the rows of direction_mc
            for i = 1:size(direction_mc, 1)
                    % Format the direction values to two decimal places
                    directionStr = num2str(direction_mc(i, :), '%0.2f ');
                    legendNames{i} = ['Direction: ', directionStr];
            end
            
            figure;
            for param = 1:4
                subplot(2,2,param);
                hold on;
                display_node = 1;
                % Initialize a cell array to store plot handles
                hPlots = cell(1, length(all_estimations_every_iter_mc)+2); % +2 for centralized approach and true parameter error line
            
                % Plot estimation errors with specific color and store handles
                for j = 1:length(all_estimations_every_iter_mc)
                    % Calculate estimation error as estimation minus true parameter
                    errors = squeeze(all_estimations_every_iter_mc{j}{:}(param, display_node, :)) - true_params_mc(j,param);
                    hPlots{j} = semilogy(errors, 'Color', colors(j, :));        
                end
            
                % Plot the centralized approach error as a horizontal line at zero (assuming centralized approach estimates the parameter correctly)
                hPlots{end-1} = yline(estimates_mc_CA(j,param) - true_params_mc(j,param), 'k', 'LineWidth', 1.5, 'DisplayName', 'Centralized Approach Error');
            
                % Plot the true parameter error line (which should be zero) and store the handle
                hPlots{end} = yline(0, '--k', 'LineWidth', 1.5, 'DisplayName', 'Zero Error');
            
                % Add the legend
                legend([hPlots{:}], [legendNames, {'Centralized Approach Error', 'Zero Error'}]);
            
                hold off;
                xlabel('Iterations (k)');
                ylabel(['Error for ' obj.labels_params{param}]);
                title(['Error in ' obj.labels_params{param}]);
            
            end
            
            sgtitle({'Error convergence for parameters of interest ($\theta_n$) at $n^{\mathrm{th}}$ node for different communication radius'}, 'Interpreter', 'latex');

        end

        function plot_MSE_error(obj,direction_mc,all_estimations_every_iter_mc,true_params_mc)
            % Plot MSE Error (Estimated(with neighbors ratio) - True Params).^2
            % Plotting Errors (\hat{\theta} - \theta).^2 results
            display_node = 1;
            legendNames = cell(1, size(direction_mc, 1)); 
            
            % Construct the legend names based on the rows of direction_mc
            for i = 1:size(direction_mc, 1)
                    % Format the direction values to two decimal places
                    directionStr = num2str(direction_mc(i, :), '%0.2f ');
                    legendNames{i} = ['Direction: ', directionStr];
            end
            
            figure;
            for param = 1:4
                subplot(2,2,param);
                hold on;

                % Initialize a cell array to store plot handles
                hPlots = cell(1, length(all_estimations_every_iter_mc)); % +2 for centralized approach and true parameter error line
            
                % Plot estimation errors with specific color and store handles
                for j = 1:length(all_estimations_every_iter_mc)
                    % Calculate estimation error as estimation minus true parameter
                    errors = (squeeze(all_estimations_every_iter_mc{j}{:}(param, display_node, :)) - true_params_mc(j,param)).^2;
                    hPlots{j} = semilogy(errors, 'Color', obj.colors(j, :));        
                end
                set(gca, 'YScale', 'log');
            
                % Plot the centralized approach error as a horizontal line at zero (assuming centralized approach estimates the parameter correctly)
            %     hPlots{end-1} = yline((estimates_mc_CA(j,param) - true_params_mc(j,param)).^2, 'k', 'LineWidth', 1.5, 'DisplayName', 'Centralized Approach Error');
            %     
            %     % Plot the true parameter error line (which should be zero) and store the handle
            %     hPlots{end} = yline(0, '--k', 'LineWidth', 1.5, 'DisplayName', 'Zero Error');
            %     
                % Add the legend
            %     legend([hPlots{:}], [legendNames, {'Centralized Approach Error', 'Zero Error'}]);
                legend([hPlots{:}], legendNames);
                hold off;
                xlabel('Iterations (k)');
                ylabel(['MSE for ' obj.labels_params{param}]);
                title(['MSE in ' obj.labels_params{param}]);
            
            end    
            sgtitle({'MSE convergence for parameters of interest ($\theta_n$) at $n^{\mathrm{th}}$ node for different communication radius'}, 'Interpreter', 'latex');
        end 
        
        function plot_errors_all_neighbors(obj, com_rad_CR, laplacian_matrix_CR,all_estimations_every_iter_CR, estimated_params_CA, network_topo,true_params)
            % Plotting Errors (\hat{\theta} - \theta) results for all neighbors
            display_node = 1;
            numColors = length(all_estimations_every_iter_CR); 
            colors = hsv(numColors);
            labels_params = {'Position x', 'Position y', 'Velocity x', 'Velocity y'};
            legendNames = cell(1, length(com_rad_CR)); 
            
            % Construct the Node Ratio as a fraction in the format 'numerator/denominator'
            for i = 1:length(com_rad_CR)
                legendNames{i} = ['Neighbors: ', num2str(laplacian_matrix_CR{i}(display_node, display_node)), '/', num2str(network_topo.numNodes)];
            end
            
            figure;
            for param = 1:4
                subplot(2,2,param);
                hold on;
            
                % Initialize a cell array to store plot handles
                hPlots = cell(1, length(all_estimations_every_iter_CR) + 2); % +2 for centralized approach and true parameter error line
            
                % Plot estimation errors with specific color and store handles
                for j = 1:length(all_estimations_every_iter_CR)
                    % Calculate estimation error as estimation minus true parameter
                    errors = squeeze(all_estimations_every_iter_CR{j}(param, display_node, :)) - true_params(param);
                    hPlots{j} = semilogy(errors, 'Color', colors(j, :));        
                end
            
                % Plot the centralized approach error as a horizontal line at zero (assuming centralized approach estimates the parameter correctly)
                hPlots{end-1} = yline(estimated_params_CA(param) - true_params(param), 'k', 'LineWidth', 1.5, 'DisplayName', 'Centralized Approach Error');
            
                % Plot the true parameter error line (which should be zero) and store the handle
                hPlots{end} = yline(0, '--k', 'LineWidth', 1.5, 'DisplayName', 'Zero Error');
            
                % Add the legend
                legend([hPlots{:}], [legendNames, {'Centralized Approach Error', 'Zero Error'}]);
            
                hold off;
                xlabel('Iterations (k)');
                ylabel(['$(', '\hat{\theta}_n', ' - \theta)$' ], 'Interpreter', 'latex');
                title(['For ' obj.labels_params{param}]);
            
            end
            
            sgtitle({'$\left(\hat{\theta}_n - \theta\right)$ for different neighbor nodes'}, 'Interpreter', 'latex');
        end 

        function plot_MSE_for_all_neightbors(obj,com_rad_CR,laplacian_matrix_CR,estimated_params_CA,all_estimations_every_iter_CR,true_params,network_topo)
            % For MSE fo all neighbors
            display_node = 1;
            legendNames = cell(1, length(com_rad_CR)); 
            
            % Construct the Node Ratio as a fraction in the format 'numerator/denominator'
            for i = 1:length(com_rad_CR)
                legendNames{i} = ['Neighbors: ', num2str(laplacian_matrix_CR{i}(display_node, display_node)), '/', num2str(network_topo.numNodes)];
            end
            figure;
            for param = 1:4
                subplot(2,2,param);
                hold on;
            
                % Initialize a cell array to store plot handles
                hPlots = cell(1, length(all_estimations_every_iter_CR) + 2); % +2 for centralized approach and true parameter error line
            
                % Plot estimation errors with specific color and store handles
                for j = 1:length(all_estimations_every_iter_CR)
                    % Calculate estimation error as estimation minus true parameter
                    errors = (squeeze(all_estimations_every_iter_CR{j}(param, display_node, :)) - true_params(param)).^2;
                    hPlots{j} = semilogy(errors, 'Color', obj.colors(j, :));        
                end
            
                % Plot the centralized approach error as a horizontal line at zero (assuming centralized approach estimates the parameter correctly)
                hPlots{end-1} = yline((estimated_params_CA(param) - true_params(param)).^2, 'k', 'LineWidth', 1.5, 'DisplayName', 'Centralized Approach Error');
            
                % Plot the true parameter error line (which should be zero) and store the handle
                hPlots{end} = yline(0, '--k', 'LineWidth', 1.5, 'DisplayName', 'Zero Error');
            
                % Add the legend
                legend([hPlots{:}], [legendNames, {'Centralized Approach Error', 'Zero Error'}]);
            
                set(gca, 'YScale', 'log');  % Set the Y-axis to logarithmic scale
            
                hold off;
                xlabel('Iterations (k)');
                ylabel(['$(', '\hat{\theta}_n', ' - \theta)^2$'], 'Interpreter', 'latex');
                title(['For ' obj.labels_params{param}]);
            
                % The rest of your code for the inset plots
                % Ensure to set 'YScale' to 'log' for the inset axes if needed
            end
            
            sgtitle({'$\left(\hat{\theta}_n - \theta\right)^2$ for Different neighbors ratio'}, 'Interpreter', 'latex');
        end

        function plot_MSE_error_compare_DA_DS(obj,direction_mc,all_estimations_every_iter_mc,estimates_mc_CA, true_params_mc)
            % Plot MSE error for (\hat{\theta} - \theta).^2 with Decentralized as straight line and distributed as dashed
            display_node = 5;
            legendNames = cell(1, size(direction_mc, 1)); 
            colors_size = size(direction_mc, 1);
            colors = hsv(colors_size);
            % Construct the legend names based on the rows of direction_mc
            for i = 1:size(direction_mc, 1)
                    % Format the direction values to two decimal places
                    directionStr = num2str(direction_mc(i, :), '%0.2f ');
                    legendNames{i} = ['Direction: ', directionStr];
            end
            figure;
            legendEntries = {};  % Initialize an empty cell array to hold legend entries
            for param_idx = 1:4
                subplot(2,2,param_idx);
                hold on;
                legendEntries = {};
                % Initialize a cell array to store plot handles
                hPlots = []; % Use a simple array to store handles
            
                for j = 1:length(all_estimations_every_iter_mc)
                    % Define colors for this particular direction
                    color = colors(j, :);
            
                    % Calculate errors for decentralized
                    errors = (squeeze(all_estimations_every_iter_mc{j}{:}(param_idx, display_node, :)) - true_params_mc(j,param_idx)).^2;
                    % Decentralized plot (straight line)
                    hPlotDecentralized = semilogy(errors, 'Color', color, 'LineStyle', '-');
                    hPlots = [hPlots, hPlotDecentralized];  % Append handle
                    legendEntries{end+1} = [legendNames{j}, ' Decentralized'];  % Append legend entry
            
                    % Calculate centralized estimation error
                    ca_error = (estimates_mc_CA(j,param_idx) - true_params_mc(j,param_idx)).^2;
                    % Centralized plot (dashed line) using the same color
                    hPlotCentralized = semilogy(1:length(errors), repmat(ca_error, 1, length(errors)), 'Color', color, 'LineStyle', '--');
                    hPlots = [hPlots, hPlotCentralized];  % Append handle
                    legendEntries{end+1} = [legendNames{j}, ' Centralized'];  % Append legend entry
                end
            
                set(gca, 'YScale', 'log');
                legend(hPlots, legendEntries, 'Location', 'best');
            
                hold off;
                xlabel('Iterations (k)');
                ylabel(['$(', '\hat{\theta}_n', ' - \theta)^2$'], 'Interpreter', 'latex');
                title(['For ' obj.labels_params{param_idx}]);
            end
            
            sgtitle('$\left(\hat{\theta}_n - \theta\right)^2$ for different directions with 19/20 neighbors', 'Interpreter', 'latex');

        end
    end
end