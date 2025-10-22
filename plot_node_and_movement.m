clc;clear;close all;

%--- Target
NUM_TAR=1;
mc = 1;
NUM_CPI_PER_MEA = 64;
TRACK_TIME = 1000;
target.initial_position = [1000, 1000];
target.speed = 20;
target.angle_degrees = [135]; % In deg.
angle_degrees = target.angle_degrees(mc);
target.direction = [cos(angle_degrees * pi / 180), sin(angle_degrees * pi / 180)];%     direction = direction / norm(direction);
target.true_params = [target.initial_position(1), target.initial_position(2), target.speed * target.direction(1), target.speed * target.direction(2)];
target.target_position = zeros(NUM_TAR,NUM_CPI_PER_MEA*TRACK_TIME, 2);
for i = 1 : NUM_TAR
    target.target_position(i, 1, :) = target.initial_position; % Initial position
end

%-- Network topo
network_topo.numNodes = 10;
theta = linspace(0,2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.com_rad_CR = 3000; % communication radius range
network_topo.radius = 3000;     % spatial placement radius 
network_topo.radar_pos = network_topo.radius * [cos(network_topo.theta); sin(network_topo.theta)]';
network_topo.C_distance = 1;  % Cost per meter
network_topo.C_data = 1;      % Cost per byte
network_topo.distances_between_radar_nodes = zeros(network_topo.numNodes,network_topo.numNodes);
network_topo.labels = {'N1','N2','N3','N4','N5','N6','N7','N8','N9','N10'};



%-- Target moving
% Target position move over in the M burst time.
time_step = 1e-4;
for i = 1:NUM_TAR
    for j = 1: TRACK_TIME
        for k = 2:NUM_CPI_PER_MEA
            target.target_position(i, (j-1)* NUM_CPI_PER_MEA + k , :) = target.target_position(i, (j-1)* NUM_CPI_PER_MEA + (k - 1), :) + reshape(target.speed * target.direction * time_step, 1,1,2);
        end
    end
end

%-- plot
figure();
plot(network_topo.radar_pos(:,1),network_topo.radar_pos(:,2),...
                        'o','MarkerSize',10,'MarkerFaceColor','r','MarkerEdgeColor','k');
labelpoints(network_topo.radar_pos(:,1),network_topo.radar_pos(:,2),network_topo.labels,'SE' ,0.2,1);
hold on;
plot(target.target_position(1,:,1),target.target_position(1,:,2),...
                        'o','MarkerSize',10,'LineWidth',3);

legend('Radar nodes','Target')
xlabel("X position (m)")
ylabel("Y position (m)")
title('Radar node and target simulation')
