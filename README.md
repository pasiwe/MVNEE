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
		For a correct execution, some settings have to be adjusted before starting the project. All settings can be specified
		in the Settings.h file. Especially the thread count for parallel execution is important. OpenEXR support can be activated by
		uncommenting line 7 in VolumeRenderer.h, yet the OpenEXR library is provided for Windows 7 (64 bit) only!.
		Objects can be loaded from .obj files. These objects have to be specified in the Main.cpp file directly.
		
	
	Linux:
		For Linux, a makefile is provided in the "VolumeRendererV2" folder. Just run the makefile with "make". The embree library
		used for object intersections will be used automatically. Note that this makefile only works for 64 bit systems.
		
	Windows:
		For Windows 7, a 64 bit Visual Studio .sln Solution file is provided. The necesary libraries are included
		with relative paths already, the dlls are provided as well. When using a 64 bit run tim environment,
		the programm should be able to start immediately using the sln file.
		
	Error cases:
		In cases of errors, try compiling Embree for your own operating system and include it for compilation.
		The code is very simple, so you should be able to cater it to your own needs.