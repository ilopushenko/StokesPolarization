classdef StokesVectorSet < handle
    % A class to store Stokes vector components and compute various
    % dependent parameters. Equation references are made mostly according
    % to M. Born and E. Wolf, "Principles of Optics", Pergamon Press, 1980.
    % Intented usage: evaluation of all common parameters of polarized
    % light from Stokes vector measurements, assuming that any other
    % information (e.g., field complex vector) is not available.
	% Ivan Lopushenko, 2024.
    % This source code is available under MIT license.
    % Repository:
    % https://github.com/ilopushenko/StokesPolarization
	% Online version (WebAssembly):
    % https://ilopushenko.github.io/projects/stokes

    properties(SetAccess=private)
        % Stokes parameters. 1xN arrays, N is amount of analyzed signals.
        % Units: arbitrary! Can be watts, or whatever. User-defined.
		% Key rule: all parameters must have the same unit system.
        I %S0
        Q %S1
        U %S2
        V %S3

        % Stokes parameters, normalized to the fully polarized part
        s1
        s2
        s3

        % Light intensities
        IH     % Horizontally polarized (x-oriented)
        IV     % Vertically polarized (y-oriented)
        ID     % Diagonally polarized (Oriented under +45deg to x axis)
        IA     % Antidiagonally polarized (Oriented under -45deg to x axis)
        IR     % Right circularly polarized
        IL     % Left circularly polarized

        % Elements of the Wolf's coherency matrix
        Jxx
        Jxy
        Jyx
        Jyy

        % Additional intensities
        Ipol   % intensity of the fully polarized part
        Idepol % intensity of the depolarizaed part

        % Dependent (computed) parameters
        DOP    % total degree of polarization
        DOLP   % degree of linear polarization
        DOCP   % degree of circular polarization
        delta  % phase difference
        chi    % ellipticity, -pi/4<=chi<=pi/4
        psi    % azimuth, 0<=psi<pi
        a      % larger ellipse semi-axis, a>=b
        b      % smaller ellipse semi-axis
    end

    methods
        %% Constructor
        function this = StokesVectorSet(varargin) %#codegen
            % Construct an instance of the StokesVectorSet object
            %   Accepts the following data as input:
            %   (a) row or column vector;
            %   (b) 4xN matrix.
            if ~isempty(varargin)
                this.addCustomStokesData(varargin{1});
            end
        end

        %% Add or remove columns
        function addVector(this, I, Q, U ,V, s1, s2, s3, ...
                                 IH, IV, ID, IA, IR, IL, ...
                                 Jxx, Jxy, Jyx, Jyy,     ...
                                 DOP, DOLP, DOCP,        ...
                                 Ipol, Idepol,           ...
                                 delta, chi, a, b, psi,  ...
                                 varargin)
            % Adds more data to the stored set, or overwrites data with the
            % selected index (can be specified by defining varargin{1}).
			% For now, implemented in a very straightforward manner.

            if ~isempty(varargin)
                idx              = varargin{1};

                this.I(idx)      = I;
                this.Q(idx)      = Q;
                this.U(idx)      = U;
                this.V(idx)      = V;

                this.s1(idx)     = s1;
                this.s2(idx)     = s2;
                this.s3(idx)     = s3;

                this.IH(idx)     = IH;
                this.IV(idx)     = IV;
                this.ID(idx)     = ID;
                this.IA(idx)     = IA;
                this.IR(idx)     = IR;
                this.IL(idx)     = IL;

                this.Jxx(idx)    = Jxx;
                this.Jxy(idx)    = Jxy;
                this.Jyx(idx)    = Jyx;
                this.Jyy(idx)    = Jyy;

                this.DOP(idx)    = DOP;
                this.DOLP(idx)   = DOLP;
                this.DOCP(idx)   = DOCP;
                this.Ipol(idx)   = Ipol;
                this.Idepol(idx) = Idepol;

                this.delta(idx)  = delta;
                this.chi(idx)    = chi;
                this.a(idx)      = a;
                this.b(idx)      = b;
                this.psi(idx)    = psi;
            else
                this.I          = [this.I I];
                this.Q          = [this.Q Q];
                this.U          = [this.U U];
                this.V          = [this.V V];

                this.s1         = [this.s1 s1];
                this.s2         = [this.s2 s2];
                this.s3         = [this.s3 s3];

                this.IH         = [this.IH IH];
                this.IV         = [this.IV IV];
                this.ID         = [this.ID ID];
                this.IA         = [this.IA IA];
                this.IR         = [this.IR IR];
                this.IL         = [this.IL IL];

                this.Jxx        = [this.Jxx Jxx];
                this.Jxy        = [this.Jxy Jxy];
                this.Jyx        = [this.Jyx Jyx];
                this.Jyy        = [this.Jyy Jyy];

                this.DOP        = [this.DOP DOP];
                this.DOLP       = [this.DOLP DOLP];
                this.DOCP       = [this.DOCP DOCP];
                this.Ipol       = [this.Ipol Ipol];
                this.Idepol     = [this.Idepol Idepol];

                this.delta      = [this.delta delta];
                this.chi        = [this.chi chi];
                this.a          = [this.a a];
                this.b          = [this.b b];
                this.psi        = [this.psi psi];
            end
        end

        function deleteVector(this, idx)
            % Delete Stokes vector with specified index from the dataset
            this.I(idx)      = [];
            this.Q(idx)      = [];
            this.U(idx)      = [];
            this.V(idx)      = [];

            this.s1(idx)     = [];
            this.s2(idx)     = [];
            this.s3(idx)     = [];

            this.IH(idx)     = [];
            this.IV(idx)     = [];
            this.ID(idx)     = [];
            this.IA(idx)     = [];
            this.IR(idx)     = [];
            this.IL(idx)     = [];

            this.Jxx(idx)    = [];
            this.Jxy(idx)    = [];
            this.Jyx(idx)    = [];
            this.Jyy(idx)    = [];

            this.DOP(idx)    = [];
            this.DOLP(idx)   = [];
            this.DOCP(idx)   = [];
            this.Ipol(idx)   = [];
            this.Idepol(idx) = [];

            this.delta(idx)  = [];
            this.chi(idx)    = [];
            this.a(idx)      = [];
            this.b(idx)      = [];
            this.psi(idx)    = [];
        end

        %% Populate object with data
        function addCustomStokesData(this,inputRAW,varargin)
            % initialize variable-size arrays for MATLAB Coder
            I       = 1; I           = zeros(1,0);
            Q       = 1; Q           = zeros(1,0);
            U       = 1; U           = zeros(1,0);
            V       = 1; V           = zeros(1,0);

            [isValid, input] = StokesVectorSet.checkIntegrity(inputRAW);

            if ~any(isValid)
                warning('Invalid Stokes vector(s) omitted');
                % can be commented out for wasm-codegen
            end

            % extract I,Q,U,V from the input array
            if numel(isValid)==1 & isValid==false
                I = [];
                Q = [];
                U = [];
                V = [];
            else
                I = input(1,logical(isValid));
                Q = input(2,logical(isValid));
                U = input(3,logical(isValid));
                V = input(4,logical(isValid));
            end

            this.computeEverythingFromStokes(I,Q,U,V,varargin{:});
        end


        function addNormalizedStokesData(this,inputRAW,varargin)
            % initialize variable-size arrays for MATLAB Coder
            power   = 1; power     = zeros(1,0);
            dop     = 1; dop       = zeros(1,0);
            s1      = 1; s1        = zeros(1,0);
            s2      = 1; s2        = zeros(1,0);
            s3      = 1; s3        = zeros(1,0);

            % check dimensions of the input array
            if isrow(inputRAW)
                % if this is a row, we expect one vector in format:
                % [power dop s1 s2 s3]
                if size(inputRAW,2)==5
                    input = inputRAW.'; % we transpose it for further use
                else
                    msg     = 'Wrong row vector (must have 5 elements)';
                    error(msg);
                    %return;
                end
            else
                input = inputRAW;
            end

            if size(input,1)~=5
                % check if supplied 2D array has 5xN size, N>=1.
                msg = 'Wrong column size (must have 5 elements)';
                error(msg);
             	%return;
            end

            % extract data from input array
            power   = input(1,:);
            dop     = input(2,:);
            s1      = input(3,:);
            s2      = input(4,:);
            s3      = input(5,:);

            map     = s1.^2+s2.^2+s3.^2<=1;
            if min(map)==0
                warning('Invalid Stokes vector(s) omitted');
            end
            [Q,U,V] = StokesVectorSet.unNormalizeEachVectorUsingPowerAndDOP(power(map),dop(map),s1(map),s2(map),s3(map));

            this.computeEverythingFromStokes(power(map),Q,U,V,varargin{:});
        end

        function addIntensityData(this, IH, IV, ID, IA, IR, IL, varargin)
            % !preliminary implementation without integrity checks!
            [I,Q,U,V] = StokesVectorSet.computeStokesFromIntensity(IH,IV,ID,IA,IR,IL);
            this.computeEverythingFromStokes(I,Q,U,V,varargin{:});
        end

        function computeEverythingFromStokes(this,I,Q,U,V,varargin)
            % compute all other parameters from I,Q,U,V
            [s1,s2,s3]                  = StokesVectorSet.normalizeEachVectorToFullyPolarizedPower(Q,U,V);
            [IH,IV,ID,IA,IR,IL]         = StokesVectorSet.computeIntensityFromStokes(I,Q,U,V);
            [DOP,DOLP,DOCP,Ipol,Idepol] = StokesVectorSet.evaluatePolarizationDegrees(I,Q,U,V);
            [Jxx,Jxy,Jyx,Jyy]           = StokesVectorSet.computeWolfMatrixFromStokes(I,Q,U,V);

            delta          = StokesVectorSet.evaluatePhase(U,V);
            chi            = StokesVectorSet.evaluateEllipticity(Ipol,V);
            [a,b]          = StokesVectorSet.evaluateEllipseAxes(Ipol,chi);
            psi            = StokesVectorSet.evaluateAzimuth(Q,U);

            this.addVector(I, Q, U ,V, s1, s2, s3, ...
                           IH, IV, ID, IA, IR, IL, ...
                           Jxx, Jxy, Jyx, Jyy,     ...
                           DOP, DOLP, DOCP,        ...
                           Ipol, Idepol,           ...
                           delta, chi, a, b ,psi,  ...
                           varargin{:});
        end

        %% Get methods
        % get Stokes vector values via S0,S1,S2,S3 notations
        function S0 = S0(this)
            S0 = this.I;
        end

        function S1 = S1(this)
            S1 = this.Q;
        end

        function S2 = S2(this)
            S2 = this.U;
        end

        function S3 = S3(this)
            S3 = this.V;
        end

        % get Wolf matrix for the 1st vector, or for a vector with
        % specified index
        function J = J(this,varargin)
            J = getWolfMatrix(this,varargin{:});
        end

        function J = getWolfMatrix(this,varargin)
            if ~isempty(varargin)
                idx = varargin{1};
            else
                idx = 1;
            end

            J = [this.Jxx(idx) this.Jxy(idx); ...
                 this.Jyx(idx) this.Jyy(idx)];
		end

        % get x-y points required to draw polarization ellipse
        function [X,Y] = getEllipse(this,varargin)
            if numel(this.I)>0
                if ~isempty(varargin)
                    idx = varargin{1};
                else
                    idx = 1;
                end
			    [X,Y] = StokesVectorSet.calculateEllipse(...
                    0, 0, this.a(idx), this.b(idx), this.psi(idx));
            else
                X = [];
                Y = [];
            end
        end

        %% Plotting procedures
        function plotEllipse(this,varargin)
            if ~isempty(varargin)
                idx = varargin{1};
                if isempty(idx)
                    idx = 1:numel(this.I);
                end
            else
                idx = 1:numel(this.I);
            end
            N   = numel(idx);
            lgd = cell([N 1]);

            if numel(varargin)>1
                ax = varargin{2};
            else
                ax = [];
            end

            if isempty(ax)
                figure;
                ax = gca;
                title('Polarization ellipse');
            end
            %ax.NextPlot = 'add';
            set(ax,'NextPlot','add');

            for i=1:N
                [X,Y] = getEllipse(this,idx(i));
                plot(ax,X,Y,'LineWidth',2);
                lgd{i} = ['id' ' ' num2str(idx(i))];
            end
            legend(ax,lgd{:},'Location','NorthEastOutside');
            grid on;
            axis(ax,'equal');
            set(gca,'FontSize',16);
        end

        function plotSphere(this,varargin)
            if ~isempty(varargin)
                idx = varargin{1};
                if isempty(idx)
                    idx = 1:numel(this.I);
                end
            else
                idx = 1:numel(this.I);
            end
            N      = numel(idx);
            lgd    = cell([N+1 1]);
            lgd{1} = 'sphere';

            if numel(varargin)>1
                ax = varargin{2};
            else
                ax = [];
            end

            if isempty(ax)
                figure;
                ax = gca;
                title('Poincaré sphere');
            end
            %ax.NextPlot = 'add';
            set(ax,'NextPlot','add');

            [x,y,z] = sphere;
            s = surf(ax,x,y,z);
            set(s,'EdgeColor','none');
            set(s,'FaceColor',[0.7 0.7 0.7],'FaceAlpha', 0.2);
            for i=1:N
                scatter3(this.s1(i),this.s2(i),this.s3(i), 88, 'o', 'filled');
                lgd{i+1} = ['id' ' ' num2str(idx(i))];
            end
            legend(ax,lgd{:},'Location','NorthEastOutside');
            grid on;
            axis(ax,'equal');
            xlabel('S_1');
            ylabel('S_2');
            zlabel('S_3');
            set(gca,'FontSize',16);
            view(-48,24);
        end

		%% helpers
		function T = serializeToTable(this)
            props                      = properties(this);
            props(strcmp(props,'Jxx')) = [];
            props(strcmp(props,'Jxy')) = [];
            props(strcmp(props,'Jyx')) = [];
            props(strcmp(props,'Jyy')) = [];
            types    = strings([1 numel(this.I)]);
            types(:) = 'double';
            T        = table('Size',[23 numel(this.I)], ...
                             'VariableTypes',types,     ...
                             'RowNames',props,          ...
                             'VariableNames',arrayfun(@num2str, 1:1:numel(this.I), ...
                                                     'UniformOutput', 0) );
            for i = 1:numel(props)
                values = this.(props{i});
                if isempty(values)
                    error('No data in the object');
                end
                T{i,:} = values;
            end

            T      = rows2vars(T);
            T(:,1) = [];
        end

        function structure = serializeToStructure(this)
            props     = properties(this);
            structure = struct;
            for i = 1:numel(props)
                structure.(props{i}) = this.(props{i});
            end
        end
    end

    methods(Static)
        %% Integrity checks for the supplied Stokes vector(s)
        function [isValid, vector, N, msg] = checkIntegrity(vectorINP)
            % check if the supplied 1D/2D array can be treated as
            % single Stokes vector or multiple Stokes vectors
            if isrow(vectorINP)
                % if this is a row, we expect one Stokes vector in format:
                % [S0 S1 S2 S2]
                if size(vectorINP,2)==4
                    vector = vectorINP.'; % we transpose it for further use
                else
                    %warning('Incorrect Stokes vector provided');
                    isValid = 0;
                    vector  = vectorINP;
                    N       = 0;
                    msg     = 'Wrong row vector (must have 4 elements)';
                    warning(msg);
                    return;
                end
            else
                vector = vectorINP; % vectorINP added for WASM compatibility
                %{
                Size mismatch (size [1 x 4] ~= size [4 x 1]).
                The size to the left is the size of the left-hand
                side of the assignment.

                Error in ==> StokesVectorSet Line: 127 Column: 21
                Code generation failed: View Error Report
                %}
            end

            if size(vector,1)~=4
                % check if supplied 2D array has 4xN size, N>=1.
                % here, we expect multiple Stokes vectors in format:
                % [ S0_1 S0_2 S0_3; ...
                %   S1_1 S1_2 S1_3; ...
                %   S2_1 S2_2 S2_3; ...
                %   S3_1 S3_2 S3_3 ]
            	%warning('Incorrect Stokes vector provided');
                isValid = 0; %false; % changed to zero for compatibility
                N       = 0;
                msg = 'Wrong column size (must have 4 elements)';
                warning(msg);
             	return
            end

            % check the validity of the supplied Stokes vector(s)
            % from the physics point of view: §10.8, Eqn. (65)
            isValid = int32(vector(1,:).^2 >= ...
                         vector(2,:).^2 + vector(3,:).^2 + vector(4,:).^2);
            N       = size(vector,2);
            msg     = '';
        end % end of checkIntegrity

        %% Evaluate polarization degrees
        function [DOP, DOLP, DOCP, Ipol, Idepol] = evaluatePolarizationDegrees(I, Q, U, V)
            DOP    = sqrt(Q.^2 + U.^2 + V.^2) ./ I; % §10.8, Eqn. (68)
            DOLP   = sqrt(Q.^2 + U.^2)        ./ I;
            DOCP   = sqrt(V.^2)               ./ I;
            Ipol   = sqrt(Q.^2 + U.^2 + V.^2);
            Idepol = I - Ipol;
        end

        %% Evaluate phase
        function delta = evaluatePhase(S2,S3)
			delta = StokesVectorSet.computeAngle_0_2pi(S2, S3); % §1.4, Eqn. (43)
		end

        %% Evaluate ellipticity
        function chi = evaluateEllipticity(S0,S3)
            chi = 0.5*asin(S3./S0); % §1.4, Eqn. (45c)
        end

        %% Evaluate ellipse semi-axes
        function [a,b] = evaluateEllipseAxes(I,chi)
            % !valid only for the fully polarized beam
            % combining §1.4, Eqns. (43) and (32), we arrive at
            A = I ./ (1+tan(chi).^2);
            B = I - A;
            a = sqrt(A);
            b = sqrt(B);
        end

        %% Evaluate azimuth
        function psi = evaluateAzimuth(S1,S2)
			psi = 0.5*StokesVectorSet.computeAngle_0_2pi(S1, S2); % §1.4, Eqns. (45a),(45b)
		end

        %% Normalization procedures
        function [S1,S2,S3] = normalizeEachVectorToTotalPower(I,Q,U,V)
            %S0 = I ./ I;
            S1 = Q ./ I;
            S2 = U ./ I;
            S3 = V ./ I;
        end

        function [i,q,u,v,maxI] = normalizeToMaximumTotalPower(I,Q,U,V)
            maxI = max(I);
            i    = I / maxI;
            q    = Q / maxI;
            u    = U / maxI;
            v    = V / maxI;
        end

        function [s1,s2,s3,Ipol] = normalizeEachVectorToFullyPolarizedPower(Q,U,V)
            Ipol = sqrt(Q.^2 + U.^2 + V.^2);
            s1   = Q ./ Ipol;
            s2   = U ./ Ipol;
            s3   = V ./ Ipol;
        end

        %% Reverse normalization procedures
        function [Q,U,V] = unNormalizeEachVectorUsingTotalPower(I,S1,S2,S3)
            Q = S1 .* I;
            U = S2 .* I;
            V = S3 .* I;
        end

        function [I,Q,U,V] = unNormalizeUsingMaximumTotalPower(maxI,i,q,u,v)
            if numel(maxI)~=1
                I = [];
                Q = [];
                U = [];
                V = [];
                warning('maximum intensity should be a double scalar value');
                return;
            end
            I = i * maxI;
            Q = q * maxI;
            U = u * maxI;
            V = v * maxI;
        end

        function [Q,U,V] = unNormalizeEachVectorUsingPowerAndDOP(Power,DOP,s1,s2,s3)
            Q = s1 .* DOP .* Power;
            U = s2 .* DOP .* Power;
            V = s3 .* DOP .* Power;
        end

        %% Intensity computation procedures
        function [IH,IV,ID,IA,IR,IL] = computeIntensityFromStokes(I,Q,U,V)
            IH = (I+Q)./2;
            IV = (I-Q)./2;
            ID = (I+U)./2;
            IA = (I-U)./2;
            IR = (I+V)./2;
            IL = (I-V)./2;
        end

        function [I,Q,U,V] = computeStokesFromIntensity(IH,IV,ID,IA,IR,IL)
            % !need check for the equivalence of IH+IV=ID+IA=IR+IL
            I = IH + IV;
            Q = IH - IV;
            U = ID - IA;
            V = IR - IL;
        end

        function [I,Q,U,V] = computeStokesFromSetOfFourIntensities(IH,IV,ID,IR)
            % D. Goldstein, "Polarized Light", p.74, Eqns. (5.113)-(5.116)
            I = IH + IV;
            Q = IH - IV;
            U = 2*ID - IH - IV;
            V = 2*IR - IH - IV;
        end

        %% Wolf coherency matrix evaluation
        function [Jxx,Jxy,Jyx,Jyy] = computeWolfMatrixFromStokes(I,Q,U,V)
            % §10.8, Eqn. (63)
            Jxx = (I+Q)./2;
            Jxy = (U+1i.*V)./2;
            Jyx = (U-1i.*V)./2;
            Jyy = (I-Q)./2;
        end

        function [I,Q,U,V] = computeStokesFromWolfMatrix(Jxx,Jxy,Jyx,Jyy)
            % !need to ensure that output is real!
            I = Jxx + Jyy;
            Q = Jxx - Jyy;
            U = Jxy + Jyx;
            V = 1i.*(Jyx - Jxy);
        end

		%% helpers
		function phi = computeAngle_0_2pi(x,y)
			% Element-wise function for computing angle phi of polar
			% coordinate system if we know Cartesian coordinates x and y.
			% Phi is assumed to be in the specific range [0, 2pi).
            % Implemented to avoid wrapTo2Pi function.
			id1 = (x>0 & y>=0);
			id2 = (x>0 & y<0);
			id3 = (x<0);
			id4 = (x==0 & y>0);
			id5 = (x==0 & y<0);
			phi = id1.*atan(y./x)        + id2.*(atan(y./x) + 2*pi) + ...
				  id3.*(atan(y./x) + pi) + id4.*pi/2       + id5.*(3*pi/2);
			phi(isnan(phi)) = 0;
		end

		function [X,Y] = calculateEllipse(cx, cy, a, b, rotAngle)
			%# This function returns points to draw an ellipse
			%#	https://stackoverflow.com/questions/2153768/draw-ellipse-and-ellipsoid-in-matlab
			%#  @param x     X coordinate
			%#  @param y     Y coordinate
			%#  @param a     Semimajor axis
			%#  @param b     Semiminor axis
			%#  @param cx    center x position
			%#  @param cy    center y position
			%#  @param angle Angle of the ellipse (in radians)
			%#
			%# Example:
			%#[X,Y] = calculateEllipse(0, 0, 20, 10, 0);
			%#plot(X, Y, 'b'); hold on; % blue
			%#[X,Y] = calculateEllipse(0, 0, 20, 10, 45);
			%#plot(X, Y, 'r'); hold on; % red
			%#[X,Y] = calculateEllipse(30, 30, 20, 10, 135);
			%#plot(X, Y, 'g'); % green
			%#grid on;

			steps = 50;
			angle = linspace(0, 2*pi, steps);

			% Parametric equation of the ellipse
			X = a * cos(angle);
			Y = b * sin(angle);

			% rotate by rotAngle counter clockwise around (0,0)
			xRot = X*cos(rotAngle) - Y*sin(rotAngle);
			yRot = X*sin(rotAngle) + Y*cos(rotAngle);
			X = xRot;
			Y = yRot;

			% Coordinate transform
			X = X + cx;
			Y = Y + cy;
		end
    end
end

