#pragma once

#include "glm/glm.hpp"
#include "RenderingUtility.h"
#include "Settings.h"

using glm::vec3;

class LightSource
{
public:
	vec3 center;

	//left handed coordinate system of the light disk:
	vec3 normal;
	vec3 u;
	vec3 v;
	//light brightness and color
	vec3 Le;
	vec3 lightColor;
	float lightBrightness;
public:
	LightSource(const vec3& center, const vec3& normal, const vec3& u, const vec3& v, const vec3& lightColor, const float& lightBrightness) :
		center(center),
		normal(normal),
		u(u),
		v(v),
		lightColor(lightColor),
		lightBrightness(lightBrightness),
		Le(lightBrightness * lightColor)		
	{

	}

	virtual ~LightSource() {

	}

	virtual vec3 sampleLightPosition(const double& xi1, const double& xi2) = 0;
	virtual double getPositionSamplingPDF() = 0;

	/* Checks, if the intersection position lies on the light source */
	virtual bool lightIntersected(const vec3& intersectionPoint) = 0;

	/* Used to implement normal culling etc. the hit direction is the direction, with which the light was hit */
	virtual bool validHitDirection(const vec3& hitDirection) = 0;

	/* Returns the type of light */
	virtual LightTypeEnum getType() = 0;

	virtual void printParameters(ofstream& o) = 0;
	
};

class LightDisk : public LightSource
{
private:
	double lightDiskSamplingPDF;
public:
	float radius;
	
public:
	LightDisk(const vec3& center, const vec3& normal, const vec3& u, const vec3& v, const vec3& lightColor, const float& lightBrightness, const float& radius) :
		LightSource(center, normal, u, v, lightColor, lightBrightness),
		radius(radius)
	{
		lightDiskSamplingPDF = (1.0 / (M_PI * (double)radius * (double)radius));
	}

	~LightDisk()
	{

	}

	vec3 sampleLightPosition(const double& xi1, const double& xi2)
	{
		//sample point on disk:
		float r = radius * (float)(sqrt(xi1));
		assert(isfinite(r));
		float theta = 2.0f * (float)(M_PI)* (float)(xi2);
		vec3 lightPosSample = center + u * r * cosf(theta) + v * r * sinf(theta);
		return lightPosSample;
	}

	double getPositionSamplingPDF()
	{
		return lightDiskSamplingPDF;
	}

	bool lightIntersected(const vec3& intersectionPoint)
	{
		bool intersected = false;

		vec3 centerToIsect = intersectionPoint - center;
		float distance = fabsf(length(centerToIsect));
		//distance has to be within radius
		if (distance <= radius) {
			if (distance <= RenderingSettings::epsilon) {
				//so close, it is definitively a hit
				intersected = true;
			}
			else {
				//check if vertex lies on the plane
				float zDist = dot(centerToIsect, normal);
				if (fabsf(zDist) <= RenderingSettings::epsilon) {
					intersected = true;
				}
			}
		}

		return intersected;
	}

	void printParameters(ofstream& o)
	{
		o << "light type = Disk Area Light" << endl;
		o << "light brightness " << lightBrightness << endl;
		o << "light radius " << radius << endl;
		o << "light center (" << center.x << ", " << center.y << ", " << center.z << ")" << endl;
		o << "light normal (" << normal.x << ", " << normal.y << ", " << normal.z << ")" << endl;
		o << "light u (" << u.x << ", " << u.y << ", " << u.z << ")" << endl;
		o << "light v (" << v.x << ", " << v.y << ", " << v.z << ")" << endl;
	}

	LightTypeEnum getType()
	{
		return LightTypeEnum::TypeLightDisk;
	}

	/* Implement normal culling */
	bool validHitDirection(const vec3& hitDirection)
	{
		return (dot(-hitDirection, normal) > 0.0f);
	}
};


