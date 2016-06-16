#pragma once
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <iostream>

#include <math.h>

#include "StringParser.h"

#include <glm/glm.hpp>

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <embree2/rtcore_geometry.h>
#include <embree2/rtcore_scene.h>

#include "Camera.h"
#include "LightSource.h"
#include "Settings.h"

#include <rapidXML/rapidxml.hpp>
#include <rapidXML/rapidxml_utils.hpp>

using namespace std;
using namespace rapidxml;
using glm::vec3;

struct vector3{ float x, y, z; };
struct EmbreeVertex   { float x, y, z, r; };
struct Triangle { int v0, v1, v2; };
struct Quad     { int v0, v1, v2, v3; };

struct ObjectData{ string name; vec3 albedo; };

const int MAX_OBJECT_COUNT = 10;

class Scene
{

private:
	RTCScene scene; //Scene object for embree	
	ObjectData objectData[MAX_OBJECT_COUNT];

public:
	Camera camera;
	LightSource* lightSource;
	
public:

	Scene(RTCScene scene) : scene(scene) {
		
	}

	~Scene() {
		if (lightSource->getType() == TypeLightDisk) {
			LightDisk* lightDisk = (LightDisk*) lightSource;
			delete lightDisk;
		}
	}

	/**
	* Adds a ground plane to the scene .
	* Returns the Embree geometry ID;
	*/
	unsigned int addGroundPlane(const string& name, const vec3& albedo, const float& sideLength, const float& yPosition)
	{
		/* create a triangulated plane with 2 triangles and 4 vertices */
		unsigned int mesh = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, 2, 4);

		/* set vertices */
		EmbreeVertex* vertices = (EmbreeVertex*)rtcMapBuffer(scene, mesh, RTC_VERTEX_BUFFER);
		vertices[0].x = -sideLength; vertices[0].y = yPosition; vertices[0].z = -sideLength;
		vertices[1].x = -sideLength; vertices[1].y = yPosition; vertices[1].z = +sideLength;
		vertices[2].x = +sideLength; vertices[2].y = yPosition; vertices[2].z = -sideLength;
		vertices[3].x = +sideLength; vertices[3].y = yPosition; vertices[3].z = +sideLength;
		rtcUnmapBuffer(scene, mesh, RTC_VERTEX_BUFFER);

		/* set triangles */
		Triangle* triangles = (Triangle*)rtcMapBuffer(scene, mesh, RTC_INDEX_BUFFER);
		triangles[0].v0 = 0; triangles[0].v1 = 2; triangles[0].v2 = 1;
		triangles[1].v0 = 1; triangles[1].v1 = 2; triangles[1].v2 = 3;
		rtcUnmapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		ObjectData objectData;
		objectData.albedo = albedo;
		objectData.name = name;
		addObjectData(mesh, objectData);

		return mesh;
	}

	/* adds a cube to the scene */
	unsigned int addCube(const vec3& center, float sideLengthHalf, const string& name, const vec3& albedo)
	{
		/* create a triangulated cube with 12 triangles and 8 vertices */
		unsigned int mesh = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, 12, 8);

		/* set vertices and vertex colors */
		EmbreeVertex* vertices = (EmbreeVertex*)rtcMapBuffer(scene, mesh, RTC_VERTEX_BUFFER);
		vertices[0].x = center.x - sideLengthHalf; vertices[0].y = center.y - sideLengthHalf; vertices[0].z = center.z - sideLengthHalf;
		vertices[1].x = center.x - sideLengthHalf; vertices[1].y = center.y - sideLengthHalf; vertices[1].z = center.z + sideLengthHalf;
		vertices[2].x = center.x - sideLengthHalf; vertices[2].y = center.y + sideLengthHalf; vertices[2].z = center.z - sideLengthHalf;
		vertices[3].x = center.x - sideLengthHalf; vertices[3].y = center.y + sideLengthHalf; vertices[3].z = center.z + sideLengthHalf;
		vertices[4].x = center.x + sideLengthHalf; vertices[4].y = center.y - sideLengthHalf; vertices[4].z = center.z - sideLengthHalf;
		vertices[5].x = center.x + sideLengthHalf; vertices[5].y = center.y - sideLengthHalf; vertices[5].z = center.z + sideLengthHalf;
		vertices[6].x = center.x + sideLengthHalf; vertices[6].y = center.y + sideLengthHalf; vertices[6].z = center.z - sideLengthHalf;
		vertices[7].x = center.x + sideLengthHalf; vertices[7].y = center.y + sideLengthHalf; vertices[7].z = center.z + sideLengthHalf;
		rtcUnmapBuffer(scene, mesh, RTC_VERTEX_BUFFER);

		/* set triangles and face colors */
		int tri = 0;
		Triangle* triangles = (Triangle*)rtcMapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		// left side
		triangles[tri].v0 = 0; triangles[tri].v1 = 2; triangles[tri].v2 = 1; tri++;
		triangles[tri].v0 = 1; triangles[tri].v1 = 2; triangles[tri].v2 = 3; tri++;

		// right side
		triangles[tri].v0 = 4; triangles[tri].v1 = 5; triangles[tri].v2 = 6; tri++;
		triangles[tri].v0 = 5; triangles[tri].v1 = 7; triangles[tri].v2 = 6; tri++;

		// bottom side
		triangles[tri].v0 = 0; triangles[tri].v1 = 1; triangles[tri].v2 = 4; tri++;
		triangles[tri].v0 = 1; triangles[tri].v1 = 5; triangles[tri].v2 = 4; tri++;

		// top side
		triangles[tri].v0 = 2; triangles[tri].v1 = 6; triangles[tri].v2 = 3; tri++;
		triangles[tri].v0 = 3; triangles[tri].v1 = 6; triangles[tri].v2 = 7; tri++;

		// front side
		triangles[tri].v0 = 0; triangles[tri].v1 = 4; triangles[tri].v2 = 2; tri++;
		triangles[tri].v0 = 2; triangles[tri].v1 = 4; triangles[tri].v2 = 6; tri++;

		// back side
		triangles[tri].v0 = 1; triangles[tri].v1 = 3; triangles[tri].v2 = 5; tri++;
		triangles[tri].v0 = 3; triangles[tri].v1 = 7; triangles[tri].v2 = 5; tri++;

		rtcUnmapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		ObjectData objectData;
		objectData.albedo = albedo;
		objectData.name = name;
		addObjectData(mesh, objectData);

		return mesh;
	}

	/**
	* Adds a circular plane specified by its center and radius, as well as the u and v vectors which together with the normal form a left handed coordinate system.
	* The resulting circualr plane lies in the u-v-plane.
	*/
	unsigned int addCircularPlane(const vec3& center, const vec3& normal, const vec3& u, const vec3& v, const float& radius, const int triangleCount, const string& name, const vec3& albedo)
	{
		/* create a triangulated circular plane with triangleCount triangles*/
		unsigned int mesh = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, triangleCount, triangleCount + 1);

		/* set vertices */
		EmbreeVertex* vertices = (EmbreeVertex*)rtcMapBuffer(scene, mesh, RTC_VERTEX_BUFFER);
		//first set center vertex:
		vertices[0].x = center.x; vertices[0].y = center.y; vertices[0].z = center.z;

		float theta;
		vec3 currentVertex;
		for (int i = 0; i < triangleCount; i++) {
			theta = ((float)i / (float)triangleCount) * (float)(2.0 * M_PI);
			currentVertex = center + radius * sinf(theta) * u + radius * cosf(theta) * v;
			vertices[i + 1].x = currentVertex.x; vertices[i + 1].y = currentVertex.y; vertices[i + 1].z = currentVertex.z;
		}
		rtcUnmapBuffer(scene, mesh, RTC_VERTEX_BUFFER);

		/* set triangle faces */
		Triangle* triangles = (Triangle*)rtcMapBuffer(scene, mesh, RTC_INDEX_BUFFER);
		for (int i = 0; i < triangleCount - 1; i++) {
			triangles[i].v0 = 0; triangles[i].v1 = i + 2; triangles[i].v2 = i + 1;
		}
		//last triangle to close to circle:
		triangles[triangleCount - 1].v0 = 0; triangles[triangleCount - 1].v1 = 1; triangles[triangleCount - 1].v2 = triangleCount;
		rtcUnmapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		ObjectData objectData;
		objectData.albedo = albedo;
		objectData.name = name;
		addObjectData(mesh, objectData);

		return mesh;
	}


	/**
	* Adds the geometry of the specified obj. file to the scene.
	* Returns the Embree geometry ID;
	*/
	unsigned int addObject(const string& objFilePath, const string& name, const vec3& albedo, const vec3& translationVector, const float& scaling, const bool& flipZAxis, const bool& flipVertexOrder) {

		cout << "Object: " << name;

		ifstream file;
		file.open(objFilePath, ios::in); // opens as ASCII!
		if (!file) {
			cout << "Object file could not be opened!" << endl;
			cout << "Please check the file path: " << objFilePath << endl;
			return RTC_INVALID_GEOMETRY_ID;
		}

		string currentLine;
		vector<vec3> vertices = vector<vec3>();
		vector<Triangle> faces = vector<Triangle>();

		while (!file.eof()) {
			getline(file, currentLine);
			StringParser line = StringParser(currentLine);
			if (!line.startsWith("#")) {
				//handle all different specifiers
				if (line.startsWith("vn")) {
				}
				else if (line.startsWith("vt")) {
				}
				else if (line.startsWith("v ")) {
					vec3 v = line.getVec3Param("v");
					if (flipZAxis) {
						vec3 vAdjusted = vec3(v.x, v.y, -v.z);
						vec3 transformedVec = (vAdjusted * scaling) + translationVector;
						vertices.push_back(transformedVec);
					}
					else {
						vec3 transformedVec = (v * scaling) + translationVector;
						vertices.push_back(transformedVec);
					}
				}
				else if (line.startsWith("f")) {

					StringParser faceStrings = StringParser(line.retrieveArgumentForParam("f"));
					//first split at ' '
					vector<StringParser> faceList = vector<StringParser>();
					faceStrings.split(' ', &faceList);

					int faceVertexCount = (int)faceList.size();
					//only triangels allowed for now!;
					//assert(faceVertexCount == 3);

					if (faceVertexCount == 3) {

						Triangle currentFace;

						vector<StringParser> perVertexIndices;
						faceList.at(0).split('/', &perVertexIndices);
						int v0 = atoi(perVertexIndices[0].getCharacterData());						

						vector<StringParser> perVertexIndices2;
						faceList.at(1).split('/', &perVertexIndices2);
						int v1 = atoi(perVertexIndices2[0].getCharacterData());						

						vector<StringParser> perVertexIndices3;
						faceList.at(2).split('/', &perVertexIndices3);
						int v2 = atoi(perVertexIndices3[0].getCharacterData());
						

						if (flipVertexOrder) {
							currentFace.v0 = v0 - 1;
							currentFace.v1 = v2 - 1;
							currentFace.v2 = v1 - 1;
						}
						else {
							currentFace.v0 = v0 - 1;
							currentFace.v1 = v1 - 1;
							currentFace.v2 = v2 - 1;
						}

						faces.push_back(currentFace);
					}
					else if (faceVertexCount > 3) {
						vector<int> faceIndices(faceVertexCount);
						for (int i = 0; i < faceVertexCount; i++) {
							vector<StringParser> perVertexIndices;
							faceList.at(i).split('/', &perVertexIndices);
							int v = atoi(perVertexIndices[0].getCharacterData()) - 1;

							faceIndices[i] = v;
						}

						//create triangles
						for (int i = 1; i < (faceVertexCount - 1); i++) {
							Triangle currentFace;
							if (flipVertexOrder) {
								currentFace.v0 = faceIndices[0];
								currentFace.v1 = faceIndices[i + 1];
								currentFace.v2 = faceIndices[i];
							}
							else {
								currentFace.v0 = faceIndices[0];
								currentFace.v1 = faceIndices[i];
								currentFace.v2 = faceIndices[i + 1];
							}
							faces.push_back(currentFace);
						}
					}
				}
			}
		}

		int vertexCount = (int)vertices.size();
		int faceCount = (int)faces.size();

		cout << " has " << faceCount << " triangles." << endl;

		/* create a mesh with faceCount triangles and vertexCount vertices */
		unsigned int mesh = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, faceCount, vertexCount);

		///////////////
		//read vertices
		///////////////

		/* set vertices */
		EmbreeVertex* verticesBuffer = (EmbreeVertex*)rtcMapBuffer(scene, mesh, RTC_VERTEX_BUFFER);
		for (int i = 0; i < vertexCount; i++) {
			//convert to our coordinate system:
			vec3 vertex = vertices.at(i);

			//set data
			verticesBuffer[i].x = vertex.x;
			verticesBuffer[i].y = vertex.y;
			verticesBuffer[i].z = vertex.z;
		}
		rtcUnmapBuffer(scene, mesh, RTC_VERTEX_BUFFER);

		///////////////
		//read faces
		///////////////				

		/* set triangles */
		Triangle* triangles = (Triangle*)rtcMapBuffer(scene, mesh, RTC_INDEX_BUFFER);
		for (int i = 0; i < faceCount; i++) {
			//set data: make sure indices start at 0!! 
			Triangle triangle = faces.at(i);

			triangles[i].v0 = triangle.v0;
			triangles[i].v1 = triangle.v1;
			triangles[i].v2 = triangle.v2;

			//triangles[i].v0 = triangle.v0;
			//triangles[i].v1 = triangle.v2;
			//triangles[i].v2 = triangle.v1;

		}
		rtcUnmapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		ObjectData objectData;
		objectData.albedo = albedo;
		objectData.name = name;
		addObjectData(mesh, objectData);

		return mesh;
	}

	bool addObjectData(int objectID, const ObjectData& data) {
		assert(objectID < MAX_OBJECT_COUNT);
		if (objectID >= 0 && objectID < MAX_OBJECT_COUNT) {
			objectData[objectID] = data;
			return true;
		}
		else {
			cout << "object id out of range! -> increase MAX_OBJECT_COUNT" << endl;
			return false;
		}
	}

	/* Returns the object data for the given geometryID in the ObjectData struct pointer
	* Return true if obejct was found, false otherwise.
	*/
	bool getObjectData(int geometryID, ObjectData* result) {

		if (geometryID >= 0 && geometryID < MAX_OBJECT_COUNT) {
			*result = objectData[geometryID];
			return true;
		}
		else {
			result->name = "none";
			result->albedo = vec3(0.0f);
			return false;
		}		
	}

	/**
	* Shoots a ray specified by its origin and direction through the Scene. Returns true if intersection is found.
	* The struct ray is used to output the intersection information when a hit was found.
	*
	* The intersection normal is stored in the output variable intersectionNormal
	*/
	bool intersectScene(const vec3& rayOrigin, const vec3& rayDir, RTCRay& ray, vec3& intersectionNormal)
	{
		ray.time = 0.0f;
		ray.org[0] = rayOrigin.x;
		ray.org[1] = rayOrigin.y;
		ray.org[2] = rayOrigin.z;
		ray.dir[0] = rayDir.x;
		ray.dir[1] = rayDir.y;
		ray.dir[2] = rayDir.z;
		ray.tnear = 0.0f;
		ray.tfar = FLT_MAX;
		ray.geomID = RTC_INVALID_GEOMETRY_ID;
		ray.primID = RTC_INVALID_GEOMETRY_ID;
		ray.mask = -1;

		rtcIntersect(scene, ray);

		bool hit = false;
		if (ray.geomID != RTC_INVALID_GEOMETRY_ID) {
			hit = true;
			vec3 ng = vec3(ray.Ng[0], ray.Ng[1], ray.Ng[2]);
			intersectionNormal = normalize(ng);
		}
		return hit;
	}

	/**
	* Checks for occlusion using a shadowray specified by a surface vertex and the direction to the light. 
	* Returns true if the surface point is occluded, i.e. the surface point is NOT visible from the light.
	* The struct ray is used to output the intersection information when a hit was found.
	*/
	bool surfaceOccluded(const vec3& surfacePosition, const vec3& dirToLight, const float& distanceToLight, RTCRay& shadowRay)
	{
		shadowRay.org[0] = surfacePosition[0];
		shadowRay.org[1] = surfacePosition[1];
		shadowRay.org[2] = surfacePosition[2];
		shadowRay.dir[0] = dirToLight[0];
		shadowRay.dir[1] = dirToLight[1];
		shadowRay.dir[2] = dirToLight[2];
		//shadowRay.tnear = Constants::epsilon;
		shadowRay.tnear = 0.0f;
		shadowRay.tfar = distanceToLight - Constants::epsilon;
		shadowRay.geomID = 1;
		shadowRay.primID = 0;
		shadowRay.mask = -1;
		shadowRay.time = 0;

		/* trace shadow ray */
		rtcOccluded(scene, shadowRay);

		return (shadowRay.geomID == 0);
	}

	void readNextLine(ifstream& inputData, StringParser& output) {
		string currentLine;
		StringParser line;
		bool jumpOver = true;

		while (jumpOver) {
			getline(inputData, currentLine);
			line = StringParser(currentLine);
			if (!line.startsWith("#") || currentLine == "") {
				jumpOver = false;
			}
		}
		output = line;
	}

	/**
	* Reads a scene from an xml scene file.
	*/
	bool readSceneXML(const string& settingsFilePath, Rendering* rendering, Medium* medium)
	{
		//check for correct file:
		ifstream file;
		file.open(settingsFilePath, ios::in); // opens as ASCII!
		if (!file) {
			cout << "invalid scene file path: " << settingsFilePath << endl;
			return false;
		}
		file.close();

		//convert xml fle to zero terminated string
		rapidxml::file<> xmlFile(settingsFilePath.c_str());

		if (xmlFile.size() <= 0) {
			cout << "invalid scene file: " << endl;
			return false;
		}

		xml_document<> doc;
		doc.parse<0>(xmlFile.data());

		//read the data:
		xml_node<>* sceneNode = doc.first_node("scene");
		xml_attribute<>* version =  sceneNode->first_attribute("version");
		string v = version->value();
		cout << "Scene: version " << v << endl;
		if (v == "1.0.0") {

			//Integrator
			xml_node<>* integratorNode = sceneNode->first_node("integrator");
			xml_attribute<>* integratorType = integratorNode->first_attribute("type");
			string integrator = integratorType->value();
			string sessionName;
			int width, height, samples, threadCount, MESC, maxSegments;
			bool renderParallel = true;

			for (xml_node<>* intSubNode = integratorNode->first_node(); intSubNode; intSubNode = intSubNode->next_sibling()) {
				string currName = intSubNode->name();
				if (currName == "output") {
					sessionName = intSubNode->first_attribute("sessionName")->value();
					StringParser widthS = StringParser(intSubNode->first_attribute("width")->value());
					width = widthS.getIntParam("");
					StringParser heightS = StringParser(intSubNode->first_attribute("height")->value());
					height = heightS.getIntParam("");
					if (width < 1 || height < 1) {
						cout << "invalid resolution! " << width << ", " << height << endl;
						return false;
					}
				}
				else if (currName == "samples") {
					StringParser samplesS = StringParser(intSubNode->first_attribute("spp")->value());
					samples = samplesS.getIntParam("");
					if (samples < 1) {
						cout << "invalid samples!" << samples << ", value has to be > 0!" << endl;
						return false;
					}
				}
				else if (currName == "maxPathSegments") {
					StringParser string = StringParser(intSubNode->first_attribute("value")->value());
					maxSegments = string.getIntParam("");
					if (maxSegments < 1) {
						cout << "invalid maxSegments!" << maxSegments << ", value has to be > 0!" << endl;
						return false;
					}
				}
				else if (currName == "MESC") {
					StringParser string = StringParser(intSubNode->first_attribute("value")->value());
					MESC = string.getIntParam("");
					if (MESC < 1) {
						cout << "invalid MESC!" << MESC << ", value has to be > 0!" << endl;
						return false;
					}
				}
				else if (currName == "threads") {
					StringParser string = StringParser(intSubNode->first_attribute("count")->value());
					threadCount = string.getIntParam("");
					if (threadCount == 1) {
						renderParallel = false;
					}
					else if (threadCount < 0) {
						cout << "invalid thread count!"<< threadCount << ", value has to be positive!" << endl;
						return false;
					}
				}
				else {
					cout << "invalid integrator param! " << currName << endl;
				}
			}
			rendering->WIDTH = width;
			rendering->HEIGHT = height;
			rendering->SAMPLE_COUNT = samples;
			rendering->MAX_MVNEE_SEGMENTS = MESC;
			rendering->MAX_MVNEE_SEGMENTS_F = (float)MESC;
			rendering->MAX_SEGMENT_COUNT = maxSegments;
			rendering->RENDER_PARALLEL = renderParallel;
			rendering->THREAD_COUNT = threadCount;
			rendering->sessionName = sessionName;
			IntegratorEnum integratorEnum;
			if (integrator == "PATH_TRACING_MVNEE_FINAL") {
				integratorEnum = PATH_TRACING_MVNEE_FINAL;
			}
			else if (integrator == "PATH_TRACING_MVNEE") {
				integratorEnum = PATH_TRACING_MVNEE;
			}
			else if (integrator == "PATH_TRACING_MVNEE_GAUSS_PERTURB") {
				integratorEnum = PATH_TRACING_MVNEE_GAUSS_PERTURB;
			}
			else if (integrator == "PATH_TRACING_MVNEE_Constants_ALPHA") {
				integratorEnum = PATH_TRACING_MVNEE_Constants_ALPHA;
			}
			else if (integrator == "PATH_TRACING_NEE_MIS") {
				integratorEnum = PATH_TRACING_NEE_MIS;
			}
			else if (integrator == "PATH_TRACING_NEE_MIS_NO_SCATTERING") {
				integratorEnum = PATH_TRACING_NEE_MIS_NO_SCATTERING;
			}
			else if (integrator == "PATH_TRACING_NO_SCATTERING") {
				integratorEnum = PATH_TRACING_NO_SCATTERING;
			}
			else if (integrator == "PATH_TRACING_RANDOM_WALK") {
				integratorEnum = PATH_TRACING_RANDOM_WALK;
			}
			else  {
				cout << "invalid integrator!! " << integrator << endl;
				return false;
			}
			rendering->integrator = integratorEnum;

			//Camera
			xml_node<>* cameraNode = sceneNode->first_node("camera");
			for (xml_node<>* camSubNode = cameraNode->first_node(); camSubNode; camSubNode = camSubNode->next_sibling()) {
				string currName = camSubNode->name();
				xml_attribute<>* camAttr = camSubNode->first_attribute();
				string camValue = camAttr->value();
				if (currName == "distanceToImagePlane") {
					camera.distanceToImagePlane = (float)atof(camValue.data());
				}
				else if (currName == "imagePlaneWidth") {
					camera.imagePlaneWidth = (float)atof(camValue.data());
				}
				else if (currName == "imagePlaneHeight") {
					camera.imagePlaneHeight = (float)atof(camValue.data());
				}
				else if (currName == "lookAt") {
					StringParser originS = StringParser(camSubNode->first_attribute("origin")->value());
					StringParser targetS = StringParser(camSubNode->first_attribute("target")->value());
					StringParser upS = StringParser(camSubNode->first_attribute("up")->value());

					vec3 origin = originS.getVec3Param("");
					vec3 target = targetS.getVec3Param("");
					vec3 up = normalize(upS.getVec3Param(""));
					vec3 lookAt = normalize(target - origin);
					vec3 right = cross(up, lookAt);

					camera.cameraOrigin = origin;
					camera.camLookAt = lookAt;
					camera.camUp = up;
					camera.camRight = right;
				}
				else {
					cout << "invalid camera param! " << currName << endl;
				}
			}

			//Lightsource
			xml_node<>* lightNode = sceneNode->first_node("lightsource");
			vec3 lightCenter, lightNormal, lightU, lightV, lightColor;
			float lightBrightness, lightRadius;

			for (xml_node<>* lightSubNode = lightNode->first_node(); lightSubNode; lightSubNode = lightSubNode->next_sibling()) {
				string currName = lightSubNode->name();

				if (currName == "position") {
					xml_attribute<>* posAttr = lightSubNode->first_attribute("center");
					StringParser posString = StringParser(posAttr->value());
					lightCenter = posString.getVec3Param("");

					posAttr = lightSubNode->first_attribute("normal");
					StringParser normalString = StringParser(posAttr->value());

					lightNormal = normalString.getVec3Param("");

					//create tangent frame from normal direction for u,v directions:
					coordinateSystem(lightNormal, lightU, lightV);
				}
				else if (currName == "radius") {
					xml_attribute<>* radAttr = lightSubNode->first_attribute("value");
					lightRadius = (float)atof(radAttr->value());
				}
				else if (currName == "brightness") {
					xml_attribute<>* brightAttr = lightSubNode->first_attribute("value");
					lightBrightness = (float)atof(brightAttr->value());
				}
				else if (currName == "color") {
					xml_attribute<>* colorAttr = lightSubNode->first_attribute("value");
					StringParser colorString = StringParser(colorAttr->value());
					lightColor = colorString.getVec3Param("");
				}
				else {
					cout << "invalid light param! " << currName << endl;
				}
			}
					
			lightSource = new LightDisk(lightCenter, lightNormal, lightU, lightV, lightColor, lightBrightness, lightRadius);
			//add light object geometry to embree scene
			addCircularPlane(lightCenter, lightNormal, lightU, lightV, lightRadius, 100, "light", lightColor);

			//MediumSettings
			xml_node<>* mediumNode = sceneNode->first_node("medium");
			xml_node<>* coefficientSubNode = mediumNode->first_node("coefficients");
			string mu_s_str = coefficientSubNode->first_attribute("mu_s")->value();
			string mu_a_str = coefficientSubNode->first_attribute("mu_a")->value();

			double mu_s = atof(mu_s_str.data());
			double mu_a = atof(mu_a_str.data());
			double mu_t = mu_s + mu_a;

			double meanFreePath = 1.0 / mu_t;
			float meanFreePathF = (float)meanFreePath;

			xml_node<>* phaseFunctionSubNode = mediumNode->first_node("phaseFunction");
			string g_str = phaseFunctionSubNode->first_attribute("g")->value();
			double g = atof(g_str.data());
			float gF = (float)g;

			medium->hg_g = g;
			medium->hg_g_F = gF;
			medium->meanFreePath = meanFreePath;
			medium->meanFreePathF = meanFreePathF;
			medium->mu_s = mu_s;
			medium->mu_a = mu_a;
			medium->mu_t = mu_t;

			//Models/Objects
			cout << "adding models to the scene..." << endl;

			for (xml_node<>* objectNode = sceneNode->first_node("model"); objectNode; objectNode = objectNode->next_sibling("model")) {
				string name = objectNode->first_attribute("name")->value();
				string modelType = objectNode->first_attribute("type")->value();
				unsigned int objID = RTC_INVALID_GEOMETRY_ID;

				if (modelType == "obj") {
					string objFilePath = objectNode->first_node("filename")->first_attribute("value")->value();
					StringParser translationS = StringParser(objectNode->first_node("transform")->first_attribute("translate")->value());
					vec3 translation = translationS.getVec3Param("");
					StringParser scaleS = StringParser(objectNode->first_node("transform")->first_attribute("scale")->value());
					float scale = scaleS.getFloatParam("");
					string flipZS = objectNode->first_node("transform")->first_attribute("flipZ")->value();
					bool flipZ = true;
					if (flipZS == "false") {
						flipZ = false;
					}

					string flipVertexOrderS = objectNode->first_node("transform")->first_attribute("flipVertexOrder")->value();
					bool flipVertexOrder = false;
					if (flipVertexOrderS == "true") {
						flipVertexOrder = true;
					}

					StringParser colorS = StringParser(objectNode->first_node("material")->first_attribute("albedo")->value());
					vec3 albedo = colorS.getVec3Param("");

					objID = addObject(objFilePath, name, albedo, translation, scale, flipZ, flipVertexOrder);
				}
				else if (modelType == "plane") {
					StringParser translationS = StringParser(objectNode->first_node("transform")->first_attribute("y")->value());
					float yTransl = translationS.getFloatParam("");
					StringParser scaleS = StringParser(objectNode->first_node("transform")->first_attribute("scale")->value());
					float scale = scaleS.getFloatParam("");
					StringParser colorS = StringParser(objectNode->first_node("material")->first_attribute("albedo")->value());
					vec3 albedo = colorS.getVec3Param("");

					objID = addGroundPlane(name, albedo, scale, yTransl);
				}
				else {
					cout << "invalid model type!" << modelType << endl;
				}

				if (objID == RTC_INVALID_GEOMETRY_ID) {
					cout << "invalid model: object file path incorrect?" << endl;
					return false;
				}
				
			}
		}
		else {
			cout << "incorrect scene file version!: " << v << endl;
			return false;
		}

		cout << "scene xml file successfully parsed." << endl;
		return true;
	}

};
