#pragma once

//Compilerflags: 
#define NDEBUG

//set in order to enable OpenEXR, if libraries are set:
//#define ENABLE_OPEN_EXR

#include "Settings.h"
#include "RenderingUtility.h"
#include "Scene.h"
#include "Path.h"
#include <omp.h>
#include <random>
#include <chrono>

using namespace std;
using glm::vec3;


#if defined _WIN32 && !defined OSG_LIBRARY_STATIC 
//Make the half format work against openEXR libs 
#define OPENEXR_DLL 
#endif 

#if defined ENABLE_OPEN_EXR
	#include <OpenEXR/ImfRgbaFile.h>
	#include <OpenEXR/ImfArray.h>
	using namespace OPENEXR_IMF_NAMESPACE;
#endif

class VolumeRenderer
{
private:
	RTCScene sceneGeometry;
	Scene* scene;
	Medium medium;
	Rendering rendering;

	double sigmaForHG;

	//variables for time measurement:
	std::chrono::system_clock::time_point measureStart;
	std::chrono::system_clock::time_point measureStop;


	//random generator, one for every thread (maximum 16 threads?)
	std::random_device rd;
	//mersenne twister random generator
	std::mt19937* mt;
	//for equally distributed random values between 0.0 and 1.0
	std::uniform_real_distribution<double>* distribution;

	//for gaussian distribution
	std::normal_distribution<double>* gaussStandardDist;

	//Every thread has its own Path:
	Path** pathTracingPaths;
	//seed and perturbed vertex arrays pre-initialized
	vec3** seedVertices;
	vec3** perturbedVertices;
	//arrays for the pdfs of all estimators
	double** estimatorPDFs;
	//pdfs for path tracing up to a specific vertex position
	double** cumulatedPathTracingPDFs;
	//arrays for the seed path segment lengths
	float** seedSegmentLengths;
	double** seedSegmentLengthSquares;

	//frame buffer:
	vec3* frameBuffer;

	/*
	* Integrator function pointer: the integrator function is chosen at run time.
	* All Integrators have to return a vec3 and have two vec3 parameters, the ray origin and
	* starting direction.
	*/
	vec3 (VolumeRenderer::*integrator)(const vec3& rayOrigin, const vec3& rayDir) = NULL;

public:
	VolumeRenderer(RTCScene sceneGeometry, Scene* scene, Medium& medium, Rendering& rendering);
	~VolumeRenderer();

	void renderScene();

	/**
	* Renders the scene and stops once the maximum duration is exceeded.
	*/
	void renderSceneWithMaximalDuration(double maxDurationMinutes);

	void writeBufferToFloatFile(const string& fileName, int width, int height, vec3* buffer);
	void readFloatFileToBuffer(const string& fileName, int* width, int* height, vec3* buffer);

	//writing to output image:
	void saveBufferToTGA(const char* filename, vec3* imageBuffer, int imageWidth, int imageHeight);

private:
	vec3 intersectionTestRendering(const vec3& rayOrigin, const vec3& rayDir);
	vec3 pathTracingNoScattering(const vec3& rayOrigin, const vec3& rayDir);
	vec3 pathTracing_NEE_MIS_NoScattering(const vec3& rayOrigin, const vec3& rayDir);

	vec3 pathTracingRandomWalk(const vec3& rayOrigin, const vec3& rayDir);
	vec3 pathTracing_NEE_MIS(const vec3& rayOrigin, const vec3& rayDir);

	/**
	* Calculates the measurement contribution as well as the path tracing PDF of the given path.
	* On top of that, the PDFs of all MVNEE estimators are calculated.
	*
	* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
	* This contribution is weighted by the MIS weight using the estimator PDFs.
	*
	* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
	* @param estimatorIndex index of the estimator that created the path:
	*			estimatorIndex = 0: path tracing
	*			estimatorIndex > 0: MVNEE starting after i path tracing segments
	* @return: the MIS weighted contribution of the path
	*/
	inline vec3 calcFinalWeightedContribution(Path* path, int estimatorIndex);


	/**
	* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
	* as well as on surfaces. Creates one path tracing path and multiple MVNEE pathsstarting at the given rayOrigin with the given direction.
	* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
	*
	* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
	* MVNEE perturbation is performed using GGX2D sampling in the u-v-plane of the seed vertices.
	*/
	vec3 pathTracing_MVNEE(const vec3& rayOrigin, const vec3& rayDir);

	/**
	* Calculates the measurement contribution as well as the path tracing PDF of the given path.
	* On top of that, the PDFs of all MVNEE estimators are calculated.
	*
	* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
	* This contribution is weighted by the MIS weight using the estimator PDFs.
	*
	* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
	* @param estimatorIndex index of the estimator that created the path:
	*			estimatorIndex = 0: path tracing
	*			estimatorIndex > 0: MVNEE starting after i path tracing segments
	* @return: the MIS weighted contribution of the path
	*/
	inline vec3 calcFinalWeightedContribution_FINAL(Path* path, int estimatorIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput);


	/**
	* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
	* as well as on surfaces. Creates one path tracing path and multiple MVNEE pathsstarting at the given rayOrigin with the given direction.
	* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
	*
	* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
	* MVNEE perturbation is performed using GGX2D sampling in the u-v-plane of the seed vertices.
	*/
	vec3 pathTracing_MVNEE_FINAL(const vec3& rayOrigin, const vec3& rayDir);

	/**
	* Calculates the measurement contribution as well as the path tracing PDF of the given path.
	* On top of that, the PDFs of all MVNEE estimators are calculated. THIS IS AN OPTIMIZED VERSION (see "pathTracing_MVNEE_GaussPerturb"):
	* many unnecessary loop iterations are avoided, further small tweaks are used in order to speed up execution. A lot of ppdf and measurement contrib
	* calcualtion steps can be avoided, as those are calcualted whilst sampling the path.
	*
	* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
	* This contribution is weighted by the MIS weight using the estimator PDFs.
	*
	* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
	* @param estimatorIndex index of the estimator that created the path:
	*			estimatorIndex = 0: path tracing
	*			estimatorIndex > 0: MVNEE starting after i path tracing segments
	* @param firstPossibleMVNEEEstimatorIndex index of the first vertex that can be used as the starting point for MVNEE
	* @param currentMeasurementContrib: measurement contrib of the path from start to the vertex at "estimatorIndex"
	* @param currentColorThroughput:  color threoughput of the path from start to the vertex at "estimatorIndex"
	* @return: the MIS weighted contribution of the path
	*/
	inline vec3 calcFinalWeightedContribution_GaussPerturb(Path* path, const int estimatorIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput);

	/**
	* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
	* as well as on surfaces. Creates one path tracing path and multiple MVNEE paths starting at the given rayOrigin with the given direction.
	* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
	*
	* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
	* MVNEE perturbation is performed using Gauss 2D sampling in the u-v-plane of the seed vertices.
	*
	* THIS IS ANOTHER OPTIMIZED VERSION: many unnecessary loop iterations are avoided, further small tweaks are used in order to speed up execution.
	* Further a lot of path tracing pdf and measurement contrib calcualtions are avoided by acumulating pdf and measurement contrib whilst
	* sampling the path.
	*/
	vec3 pathTracing_MVNEE_GaussPerturb(const vec3& rayOrigin, const vec3& rayDir);

	/**
	* Calculates the measurement contribution as well as the path tracing PDF of the given path.
	* On top of that, the PDFs of all MVNEE estimators are calculated. THIS IS AN OPTIMIZED VERSION (see "pathTracing_MVNEE_ConstantsAlpha"):
	* many unnecessary loop iterations are avoided, further small tweaks are used in order to speed up execution. A lot of pdf and measurement contrib
	* calcualtion steps can be avoided, as those are calcualted whilst sampling the path.
	*
	* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
	* This contribution is weighted by the MIS weight using the estimator PDFs.
	*
	* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
	* @param estimatorIndex index of the estimator that created the path:
	*			estimatorIndex = 0: path tracing
	*			estimatorIndex > 0: MVNEE starting after i path tracing segments
	* @param firstPossibleMVNEEEstimatorIndex index of the first vertex that can be used as the starting point for MVNEE
	* @param currentMeasurementContrib: measurement contrib of the path from start to the vertex at "estimatorIndex"
	* @param currentColorThroughput:  color threoughput of the path from start to the vertex at "estimatorIndex"
	* @return: the MIS weighted contribution of the path
	*/
	inline vec3 calcFinalWeightedContribution_ConstantsAlpha(Path* path, const int estimatorIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput);

	/**
	* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
	* as well as on surfaces. Creates one path tracing path and multiple MVNEE paths starting at the given rayOrigin with the given direction.
	* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
	*
	* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
	* MVNEE perturbation is performed using 2D GGX PDF sampling with a cosntant alpha-> so no calculation of position dependent perturbation is performed!
	*
	* THIS IS ANOTHER OPTIMIZED VERSION: many unnecessary loop iterations are avoided, further small tweaks are used in order to speed up execution.
	* Further a lot of path tracing pdf and measurement contrib calcualtions are avoided by acumulating pdf and measurement contrib whilst
	* sampling the path.
	*/
	vec3 pathTracing_MVNEE_ConstantsAlpha(const vec3& rayOrigin, const vec3& rayDir);

	inline double sample1D(const int threadID);
	inline double sample1DOpenInterval(const int threadID);

	inline double sampleArbitraryGauss1D(const double& stdDeviation, const int threadID);
	inline double gaussPDF(const float& x, const double& stdDeviation);

	/** Perturb a vertex on the tangent plane using a gaussian sampled distance, gaussian is defined by N(0, sigma), where sigma is the std-deviation
	* @param input: vector that has to be perturbed
	* @param u: first tangent vector as perturbation direction 1
	* @param v: second tangent vector as perturbation direction 2
	* @return output: perturbed input vector
	*/
	inline void perturbVertexGaussian2D(const double& sigma, const vec3& u, const vec3& v, const vec3& input, vec3* output, const int threadID);

	/**
	* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
	* until the curve_length is exceeded. Also returns the segments in an array (maximal MAX_PATH_LENGTH).
	* Return value is false if MAX_SEGMENT_LENGTH is exceeded, true otherwise.
	*/
	inline bool sampleSeedSegmentLengths(const double& curve_length, float* segments, int* segmentCount, const int threadID);

	/**
	* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
	* until the curve_length is exceeded. Also returns the segments in an array (maximal MAX_PATH_LENGTH) as well as the squared segment lengths.
	* Return value is false if MAX_SEGMENT_LENGTH is exceeded, true otherwise.
	*/
	inline bool sampleSeedSegmentLengths(const double& curve_length, float* segments, double* segmentSquares, int* segmentCount, const int threadID);

	/**
	* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
	* until the curve_length is exceeded or the provided maxSegments are reached. Also returns the segments in an array (maximal MAX_PATH_LENGTH) as well as the squared segment lengths.
	* Once maxSegments are reached, the sampling is stopped and the remaining distance is set as the last segment. This ensures that during segment sampling, no invalid paths are created.
	*/
	inline void sampleSeedSegmentLengths(const double& curve_length, float* segments, double* segmentSquares, int* segmentCount, const int threadID, const int maxSegments);

	/** Perturb a vertex on the tangent plane using a ggx sampled radius, and a uniformly sampled angle
	* @param input: vector that has to be perturbed
	* @param alpha width of ggx bell curve
	* @param u: first tangent vector as perturbation direction 1
	* @param v: second tangent vector as perturbation direction 2
	* @return output: perturbed input vector
	*/
	inline void perturbVertexGGX2D(const double& alpha, const vec3& u, const vec3& v, const vec3& input, vec3* output, const int threadID);	


#if defined ENABLE_OPEN_EXR
	void saveBufferToOpenEXR(const char* filename, vec3* imageBuffer, int imageWidth, int imageHeight);
#endif

	//output meta files:
	void printRenderingParameters(int sampleCount, double duration);
	vec3 calcMeanImageBrightnessBlockwise();
};