# StokesPolarization
MATLAB/Octave class to extract data from Stokes vector polarimetric measurements (preliminary version, v0.5). 

**Intented usage**: evaluation of all common parameters of polarized light (like degrees of polarization, intensity values, Wolf's coherency matrix, azimuth, ellipticity and axes of the polarization ellipse) from Stokes vector, assuming that any other information (e.g., field complex vector) is not available. With this assumption, usage of the MATLAB Phased Array System toolbox (functions [stokes](https://se.mathworks.com/help/phased/ref/stokes.html), [polellip](https://se.mathworks.com/help/phased/ref/polellip.html), [polratio](https://se.mathworks.com/help/phased/ref/polratio.html) and others) can be either unconvenient, or not always possible. 

Provided script allows to plot intensity-dependent polarization ellipse for each supplied Stokes vector, as well as to depict polarization state normalized to the intensity of the fully polarized part on the Poincaré sphere.

**Input data**: can be either single beam with polarization state defined as $[I,Q,U,V]$ row or column vector, or several beams defined as matrix with $N$ specified polarization states:

$$
\left[\begin{matrix}
 I_1 & I_2 & ... & I_N \\ 
 Q_1 & Q_2 & ... & Q_N \\ 
 U_1 & U_2 & ... & U_N \\ 
 V_1 & V_2 & ... & V_N \\ 
\end{matrix}\right].
$$

All input parameters must have one unit system (user-defined).

It is not necessary to make an instance of the class in order to use it. Main functions are available as element-wise static methods of the class, so if you need to e.g. compute only azimuth of the polarization ellipse, you can call 
```
psi = StokesVectorSet.evaluateAzimuth(S1,S2);
``` 
and obtain it. For the full list of static methods, please refer to the source code.

**Compatibility**: both example and class files have been tested in MATLAB R2021b and Octave 9.1.0. Octave and some earlier versions of MATLAB (below R2016b, for the current implementation of the class) do not support serialization to tables and can have some OpenGL glitches when plotting results. All other functionality is available (if not, please let me know). In particular, example_octave.m script successfully worked in MATLAB R2015b.

**WebAssembly version**: this code has been compiled to run in web browser with [MATLAB Coder and Emscripten](https://www.mathworks.com/matlabcentral/fileexchange/69973-generatejavascriptusingmatlabcoder), and is available as [online tool](https://ilopushenko.github.io/projects/stokes). More information, including short overview of theory, is also available in the online tool.

**References**: all equations within code are enumerated with respect to  
M. Born and E. Wolf. Principles of Optics, 6th Edition. Pergamon Press (1980).