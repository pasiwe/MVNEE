#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <iostream>
#include <glm/glm.hpp>
#include <omp.h>
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <embree2/rtcore_geometry.h>
#include <embree2/rtcore_scene.h>
#include "Scene.h"
#include "Settings.h"
#include "VolumeRenderer.h"

using namespace std;
using glm::vec3;


int main(int argc, const char* argv[])
{	

	cout << "-----------------------" << endl;
	cout << "VolumeRenderer started." << endl;
	cout << "-----------------------" << endl << endl;

	//tests for compability:
	RTCDevice device = rtcNewDevice(NULL);
	RTCScene scene = rtcDeviceNewScene(device, RTC_SCENE_STATIC, RTC_INTERSECT1);

	//create and fill scene
	Scene* sceneObject = new Scene(scene);



	bool geometryIncluded = true;
	if (geometryIncluded) {
		sceneObject->addGroundPlane("groundPlane", vec3(0.3f, 0.5f, 0.3f), 10.0f, -1.0f);
		unsigned int objID1 = sceneObject->addObject("../models/suzanne.obj", "suzanne", vec3(0.0f, 1.0f, 1.0f), vec3(-1.0f, 0.5f, -0.5f), 0.6f);
		//unsigned int objID = sceneObject->addObject("../models/GitterModell/gitter_square.obj", "gitter", vec3(0.11f, 0.1f, 0.1f), vec3(0.0f, 0.5f, 2.0f), 0.5f);
		//unsigned int objID2 = sceneObject->addObject("../models/GitterModell/gitter_schraeg.obj", "gitter", vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 0.8f, 0.0f), 1.0f);
		//unsigned int objID2 = sceneObject->addObject("../models/GitterModell/gitter_symmetrisch_schraeg.obj", "gitter", vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 0.8f, 0.0f), 1.0f);
		//unsigned int objID2 = sceneObject->addObject("../models/lochplatte/LochInMitte.obj", "gitter", vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 0.8f, 0.0f), 1.0f);
		//unsigned int objID2 = sceneObject->addObject("../models/GitterModell/cross.obj", "cross", vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 1.0f, 1.0f), 0.5f);
		//unsigned int objID2 = sceneObject->addObject("../models/GitterModell/cross_sides.obj", "cross", vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 1.2f, 1.0f), 0.6f);
		unsigned int objID3 = sceneObject->addObject("../models/sphere.obj", "sphere", vec3(1.0f, 1.0f, 0.0f), vec3(2.0f, 0.0f, 0.5f), 0.6f);				

	}

	//commit after all scene geometry is finished
	rtcCommit(scene);

	//create volumeRenderer and start rendering
	VolumeRenderer* volumeRenderer = new VolumeRenderer(scene, sceneObject);
	volumeRenderer->renderScene();	
	//volumeRenderer->renderSceneWithMaximalDuration(5.0);
	//volumeRenderer->renderSceneWithMaximalDuration(60.0f);

	delete volumeRenderer;

	rtcDeleteScene(scene);
	rtcDeleteDevice(device);
}