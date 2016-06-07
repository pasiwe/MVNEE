#pragma once

#include "glm/glm.hpp"
#include <string>

using std::string;
using glm::vec3;

/**
* Enum for all types of lights.
*/
enum LightTypeEnum {
	TypeLightDisk,
	TypeSpotlight
};

/**
* Enum for the identification of all implemented Integrators
*/
enum IntegratorEnum {
	PATH_TRACING_NO_SCATTERING,
	PATH_TRACING_NEE_MIS_NO_SCATTERING,
	PATH_TRACING_RANDOM_WALK,
	PATH_TRACING_NEE_MIS,
	PATH_TRACING_MVNEE,
	PATH_TRACING_MVNEE_FINAL,
	PATH_TRACING_MVNEE_GAUSS_PERTURB,
	PATH_TRACING_MVNEE_Constants_ALPHA,
	TEST_RENDERING
};

namespace Constants
{
	const float epsilon = 0.0001f;

	//sigma-values for mean cosines 0.0, 0.1,..., 0.9 for 1 mfp:
	const double sigmaPhi[10] = {
		1.07526,
		0.961113,
		0.839835,
		0.716209,
		0.594202,
		0.476602,
		0.365761,
		0.263209,
		0.169391,
		0.0830828
	};

	const double GGX_CONVERSION_Constants = 1.637618734;

	//blockwise mean image brightness
	const int TILES_SIDE = 8;
}

struct Medium
{
	double mu_s; //scattering coefficient
	double mu_a; //absorption coefficient
	double mu_t; //extinction coefficient

	double hg_g; //mean cosine for Henyey Greenstein
	float hg_g_F;

	double meanFreePath;
	float meanFreePathF;
};

struct Rendering
{
	bool RENDER_PARALLEL;
	int THREAD_COUNT;

	//maximal path segment count, before rendering is stopped
	int MAX_SEGMENT_COUNT;

	/* an initial maximum number of MVNEE segments:
	* if the distance to light divided by the mean free path is greater than MAX_MVNEE_SEGMENTS,
	* MVNEE will not be executed!
	*/
	int MAX_MVNEE_SEGMENTS;
	float MAX_MVNEE_SEGMENTS_F;

	//image settings
	int WIDTH;
	int HEIGHT;
	int SAMPLE_COUNT;

	string sessionName;

	//specify integrator here:
	IntegratorEnum integrator;
};
//
//namespace MediumParameters
//{
//	const double mu_s = 1.0;
//	const double mu_a = 0.0;
//	const double mu_t = mu_s + mu_a;
//
//	const double hg_g = 0.8;
//	const float hg_g_F = 0.8f;	
//
//	const double meanFreePath = 1.0 / mu_t;
//	const float meanFreePathF = 1.0f / (float)mu_t;
//	
//}
//
//namespace RenderingSettings
//{
//
//	const bool RENDER_PARALLEL = true;
//	//const int THREAD_COUNT = 8;
//	const int THREAD_COUNT = 1;
//
//	//maximal path segment count, before rendering is stopped
//	const int MAX_SEGMENT_COUNT = 20;
//
//	/* an initial maximum number of MVNEE segments:
//	* if the distance to light divided by the mean free path is greater than MAX_MVNEE_SEGMENTS,
//	* MVNEE will not be executed!
//	*/
//	const int MAX_MVNEE_SEGMENTS = 6;
//	const float MAX_MVNEE_SEGMENTS_F = (float)MAX_MVNEE_SEGMENTS;
//
//	//image settings
//	const int WIDTH = 600;
//	const int HEIGHT = 400;
//	const int SAMPLE_COUNT = 35000;
//
//	//blockwise mean image brightness
//	const int TILES_SIDE = 8;	
//
//	const string sessionName = "imageName";
//
//	//specify integrator here:
//	const IntegratorEnum integrator = PATH_TRACING_MVNEE_FINAL;
//	
//}
//
//namespace CameraSettings
//{
//	//standard setting:
//	const vec3 cameraOrigin(0.0f, 0.1f, -3.0f); //focal point, where all camera rays originate from
//	const vec3 camLookAt(0.0f, 0.0f, 1.0f); //direction camera looks at
//	const vec3 camUp(0.0f, 1.0f, 0.0f); //direction facing up
//	const vec3 camRight(1.0f, 0.0f, 0.0f); //last direction forming a left handed coordiante system
//
//	const float distanceToImagePlane = 1.0f;
//
//	//const float imagePlaneWidth = 1.0f;
//	const float imagePlaneWidth = 1.5f;
//	const float imagePlaneHeight = 1.0f;
//};
//
//namespace LightSettings
//{
//	const LightTypeEnum lightType = TypeLightDisk;
//
//	//front facing to left side
//	const vec3 lightCenter(0.0f, 1.7f, 1.0f);
//	const vec3 lightNormal = normalize(vec3(0.0f, -1.0f, 0.0f));
//
//	//left handed coordinate system of the light disk:
//	const vec3 lightV = normalize(vec3(1.0f, 0.0f, 0.0f));
//	//const vec3 lightU = cross(lightNormal, lightV);
//	const vec3 lightU = cross(lightV, lightNormal);
//
//	//brightness and light color:
//	const float lightBrightness = 60.0f;
//	const vec3 lightColor(1.0f, 1.0f, 1.0f);
//
//	//additional parameters:
//	const float lightRadius = 0.2f;
//}