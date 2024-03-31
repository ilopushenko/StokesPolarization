% This is an example script showcasing usage of the StokesVectorSet class.
% WebAssembly version: https://ilopushenko.github.io/projects/stokes

% prepare workspace
clear;

% 1st example
% (Attention: StokesVectorSet object has handle type!)
%   Object constructor accepts the following data as input:
%   (a) row [S0 S1 S2 S3] or column [S0; S1; S2; S3] vector;
%   (b) 4xN matrix.
example1 = StokesVectorSet([1 0.1 0.2 0.3]); % initialize object with row vector
example1.addCustomStokesData([1; 0.1; -0.2; -0.3]); % add another beam to the set (using column vector)
T1       = example1.serializeToStructure; % serialize all computed data
example1.plotEllipse; % plot polarization ellipses for all beams in the object
example1.plotSphere; % depict state of all beams on the Poincaré sphere
% Here, only normalized fully polarized part of the beam is depicted!



% 2nd example
%   StokesVectorSet object can also be supplied with normalized
%   measurements data:
%   (a) row [power dop s1 s2 s3] or column [power; dop; s1; s2; s3] vector;
%   (b) 5xN matrix.
%   NB: constructor does not accept this format, class method is used.
example2 = StokesVectorSet;
example2.addNormalizedStokesData([1 1 0 0 1]);
T2       = example2.serializeToStructure;



% 3rd example
% This example corresponds to the "Load demo data" in WebAssembly version.
% Here, we create object and fill it with degenerate polarization states:
example3 = StokesVectorSet([ 1   1   1   1   1   1; ...
                             1  -1   0   0   0   0; ...
                             0   0   1  -1   0   0; ...
                             0   0   0   0   1  -1     ]);
T3       = example3.serializeToStructure;
example3.plotEllipse;
example3.plotSphere;
% Here, only normalized fully polarized part of the beam is depicted!


% EOF
