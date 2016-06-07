------------------------
VolumeRendererV2 for Multiple Vertex Next Event Estimation:

This is a small Volume Rendering Tool for homogeneous limitless participating media, 
featuring Random Walk Path Tracing, NEE and Multiple Vertex Next Event Estimation.
------------------------

------------------------
Installation Guide
------------------------
The entire code is written in C++ 11, all libraries are only provided in 64 bit version. 

	Preliminaries:
		OpenEXR support can be activated by	uncommenting line 7 in VolumeRenderer.h, yet the OpenEXR library is provided for Windows 7 (64 bit) only!
		The scenes are specified in xml format, with a default scene provided in "MVNEE/setups/default.xml". 	
		Models can be loaded from .obj files. These objects have to be specified in scene file as well, make sure the path to the objects is 
		specified relative to the location of the source code!
		In the integrator settings of the Scene file, make sure to adjust the thread count to your machine settings. Following integrators are provided:
		
		implemented Integrators:
		
			PATH_TRACING_NO_SCATTERING: 
				Standard path tracing, excluding medium interaction.
				
			PATH_TRACING_NEE_MIS_NO_SCATTERING:
				Standard path tracing combined with Next Event Estimation, excluding medium interaction.
				
			PATH_TRACING_RANDOM_WALK:
				Random Walk Path Tracing implementation for Multiple Scattering in homogeneous media.
				
			PATH_TRACING_NEE_MIS:
				Random Walk Path Tracing combined with Next Event Estimation for Multiple Scattering in homogeneous media.
				
			PATH_TRACING_MVNEE:
				First version of combination of Random Walk Path Tracing with Multiple Vertex Next Event Estimation. This version
				is rather slow, yet relatively easy to understand. 
				
			PATH_TRACING_MVNEE_FINAL:
				Optimized version of combination of Random Walk Path Tracing with Multiple Vertex Next Event Estimation. This version
				is faster, due to some improvements.  
			
			PATH_TRACING_MVNEE_GAUSS_PERTURB:
				Optimized version of combination of Random Walk Path Tracing with Multiple Vertex Next Event Estimation. This version
				uses a 2D Gaussian PDF for perturbation, which results in more variance.
				
			PATH_TRACING_MVNEE_Constants_ALPHA:
				Optimized version of combination of Random Walk Path Tracing with Multiple Vertex Next Event Estimation. This version
				uses a fixed GGX alpha value for perturbation, which results in a lot of variance.
	
	Linux:
		For Linux, a makefile is provided in the "VolumeRendererV2" folder. Just run the makefile with "make". The embree library
		used for object intersections will be used automatically. Note that this makefile only works for 64 bit systems.
		
	Windows:
		For Windows 7, a 64 bit Visual Studio .sln Solution file is provided. The necessary libraries are included
		with relative paths already, the dlls are provided as well. When using a 64 bit run time environment,
		the programm should be able to start immediately using the sln file.
		
	Error cases:
		In cases of errors, try compiling Embree for your own operating system and include it for compilation.
		The code is very simple, so you should be able to cater it to your own needs.