classdef academic_plot
    % Source code from :https://nl.mathworks.com/matlabcentral/fileexchange/48048-academic-figures
    methods
        function handles = abar(varargin)
            afigure(gcf);
            hnds = bar(varargin{:});
            if nargout > 0
                handles = hnds;
            end
        end

       function config = aconfig(varargin)
            config.BackgroundColor = [1 1 1];
            config.FontSize = 20;
            config.LineWidth = 2;
            config.DrawBox = true;
            config.Grid = true;
            config.RemoveMargins = true;
            config.Width = [];
            config.Height = [];
            config.SizeRatio = 4/3; 
            config.Colormap = 'cmr';
            config.LineStyles = '-|-.|--';  
            for i = 1:2:numel(varargin)
                item = varargin{i};
                if isfield(config, item)
                if i + 1 > numel(varargin),
                    warning('AcademicFigures:aconfig', ...
                    'No value given for property %s', item);
                else
                    config.(item) = varargin{i+1};
                end
                else
                warning('AcademicFigures:aconfig', 'Unknown field: %s', item);
                end
            end 
        end
        
        function handle = afigure(h, config)
            if nargin == 0
                h = figure;
                config = [];
            elseif nargin == 1
                if isstruct(h)
                config = h;
                h = figure;
                else
                figure(h);
                config = [];
                end
            else
                if isstruct(h)
                h2 = config;
                config = h;
                h = figure(h2);
                else
                figure(h);
                end
            end
            
            if nargout > 1
                handle = h;
            end
            
            if ~isempty(config)
                colors = getColors(config.Colormap);
                guidata(h, config);
                apply_config_fig(h, config, colors);
            elseif isempty(guidata(h))
                config = aconfig;
                colors = getColors(config.Colormap);
                guidata(h, config);
                apply_config_fig(h, config, colors);
            end
        end
  
        function apply_config_fig(h, config, colors)
        
        set(h, 'Color', config.BackgroundColor);
        set(h, 'Colormap', getColors(config.Colormap));
        set(h, 'DefaultAxesColorOrder', colors);
        set(h, 'DefaultAxesLineStyleOrder', config.LineStyles);
        set(h, 'DefaultAxesFontSize', config.FontSize);
        set(h, 'DefaultLineLineWidth', config.LineWidth);
        if config.RemoveMargins
            set(h, 'DefaultAxesLooseInset', [0 0 0 0]);
        end
        if config.DrawBox
            set(h, 'DefaultAxesBox', 'on');
        else
            set(h, 'DefaultAxesBox', 'off');
        end
        if config.Grid
            set(h, 'DefaultAxesXGrid', 'on', 'DefaultAxesYGrid', 'on', ...
            'DefaultAxesZGrid', 'on');
        else
            set(h, 'DefaultAxesXGrid', 'off', 'DefaultAxesYGrid', 'off', ...
            'DefaultAxesZGrid', 'off');
        end
 
        p = get(h, 'Position');
        if ~isempty(config.Width) && ~isempty(config.Height)
            set(h, 'Position', [p(1:2), config.Width, config.Height]);
        elseif ~isempty(config.Width) && ~isempty(config.SizeRatio)
            set(h, 'Position', [p(1:2), config.Width, ...
            config.Width / config.SizeRatio]);
        elseif ~isempty(config.Height) && ~isempty(config.SizeRatio)
            set(h, 'Position', [p(1:2), config.Height * config.SizeRatio, ...
            config.Height]);
        elseif ~isempty(config.Width)
            set(h, 'Position', [p(1:2), config.Width, p(4)]);
        elseif ~isempty(config.Height)
            set(h, 'Position', [p(1:3), config.Height]);
        elseif ~isempty(config.SizeRatio)
            set(h, 'Position', [p(1:3), p(3) / config.SizeRatio]);
        end
        
        ha = get(h, 'CurrentAxes');
        if ~isempty(ha)
            apply_config_axis(ha, config, colors);
        end
    end
  
    function apply_config_axis(h, config, colors)
        hh = [h; get(h, 'XLabel'); get(h, 'YLabel'); get(h, 'ZLabel'); ...
            get(h, 'Title')];
        set(hh, 'FontSize', config.FontSize);
        
        set(h, 'ColorOrder', getColors(config.Colormap));
        set(h, 'LineStyleOrder', config.LineStyles);
        set(h, 'FontSize', config.FontSize);
        if config.DrawBox
            set(h, 'Box', 'on');
        else
            set(h, 'Box', 'off');
        end
        
        if config.Grid
            set(h, 'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on');
        else
            set(h, 'XGrid', 'off', 'YGrid', 'off', 'ZGrid', 'off');
        end
        
        if config.RemoveMargins
            set(h, 'LooseInset', [0 0 0 0]);
        end
    end
    function colors = getColors(colormap)
    % Original authors of colormaps:
    % http://www.mathworks.com/matlabcentral/fileexchange/2662
    % http://www.mathworks.com/matlabcentral/fileexchange/41583
        colors = [];
        if ischar(colormap)
            while isempty(colors)
                switch colormap
                    case 'thermal' 
                    colors = [ 0 0 0; 0.3 0 0.7; 1 0.2 0; 0.9 0.9 0 ];
                    case 'cmr'
                    colors = [ 0 0 0; 1 0.25 0.15; 0.9 0.5 0; 0.9 0.75 0.1; ...
                                0.85 0.85 0.5 ];
                    case 'dusk'
                    colors = [ 0 0 0; 0 0 0.5; 0 0.5 0.5; 0.5 0.5 0.5; 1 0.5 0.5; ...
                                1 1 0.5 ];
                    case 'hsv2'
                    colors = [ 0 0 0; 0.5 0 0.5; 0 0 0.9; 0 1 1; 0 1 0; 1 1 0; 1 0 0 ];
                    case 'gray'
                    colors = [ 0 0 0; 0.4 0.4 0.4; 0.65 0.65 0.65; 0.8 0.8 0.8 ];
                    otherwise
                    warning('AcademicFigures:colormap', ...
                        'Unknown colormap: %s', colormap);
                    colormap = 'cmr';
                end
            end
        else
            colors = colormap;
        end
    end
    function handles = aplot(varargin)
        afigure(gcf);
        hnds = plot(varargin{:});
        if nargout > 0
            handles = hnds;
        end
    end

    end
end 