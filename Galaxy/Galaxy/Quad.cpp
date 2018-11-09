#include <iostream> 
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

struct Particle {
	int x;
	int y;
	double force;
	double velocityX;
	double velocityY;
	double mass;

	Particle(int _x, int _y) {
		x = _x;
		y = _y;
		mass = rand() % 100;
		force = 0.0;
		velocityX = (0.5 - (-0.5)) * ((double)rand() / (double)RAND_MAX) + (-0.5);
		velocityY = (0.5 - (-0.5)) * ((double)rand() / (double)RAND_MAX) + (-0.5);
	}
	Particle(int _x, int _y, int _mass) {
		x = _x;
		y = _y;
		mass = _mass;
		force = 0.0;
		velocityX = (0.5 - (-0.5)) * ((double)rand() / (double)RAND_MAX) + (-0.5);
		velocityY = (0.5 - (-0.5)) * ((double)rand() / (double)RAND_MAX) + (-0.5);
	}
	Particle() {
		x = 0;
		y = 0;
		mass = rand() % 100;
		force = 0.0;
		velocityX = 0.0;
		velocityY = 0.0;
	}
};

struct Node {
	Particle particle;

	__host__ __device__
	Node(Particle _particle) {
		particle = _particle;
	}
};

class Quad {
	int numberOfParticles;
	int centerOfMassX;
	int centerOfMassY;
	double mass;
	double force;
	double forcex;
	double forcey;
	Particle topLeft;
	Particle botRight;
	Node *node;
	Quad *topLeftTree;
	Quad *topRightTree;
	Quad *botLeftTree;
	Quad *botRightTree;

public:
	Quad(Particle _topleft, Particle _botRight) {
		numberOfParticles = 0;
		centerOfMassX = 0;
		centerOfMassY = 0;
		mass = 0;
		force = 0;
		topLeft = _topleft;
		botRight = _botRight;
		node = NULL;
		topLeftTree = NULL;
		topRightTree = NULL;
		botLeftTree = NULL;
		botRightTree = NULL;
	}
	Quad() {
		numberOfParticles = 0;
		centerOfMassX = 0;
		centerOfMassY = 0;
		mass = 0;
		force = 0;
		topLeft = Particle(0, 0);
		botRight = Particle(1000, 1000);
		node = NULL;
		topLeftTree = NULL;
		topRightTree = NULL;
	}
	~Quad() {}

	__host__ __device__
	void insert(Node *_node) {
		if (_node == NULL) {
			return;
		}
		if (!inBoundary(_node->particle)) {
			return;
		}
		// No subsequent division is possible
		if (numberOfParticles == 0 || abs(topLeft.x - botRight.x) <= 1 && abs(topLeft.y - botRight.y) <= 1) {
			if (node == NULL) {
				this->numberOfParticles++;
				this->node = _node;
			}
			return;
		}
		if ((topLeft.x + botRight.x) / 2 >= _node->particle.x) {
			//top left node
			if ((topLeft.y + botRight.y) / 2 >= _node->particle.y) {
				if (topLeftTree == NULL) {
					topLeftTree = new Quad(
						Particle(topLeft.x, topLeft.y),
						Particle((topLeft.x + botRight.x) / 2, (topLeft.y + botRight.y) / 2));
				}
				this->numberOfParticles++;
				topLeftTree->insert(_node);
			}
			// bottom left
			else {
				if (botLeftTree == NULL) {
					botLeftTree = new Quad(
						Particle(topLeft.x,
						(topLeft.y + botRight.y) / 2),
						Particle((topLeft.x + botRight.x) / 2, botRight.y));
				}
				this->numberOfParticles++;
				botLeftTree->insert(_node);
			}
		}
		else {
			// top right 
			if ((topLeft.y + botRight.y) / 2 >= _node->particle.y) {
				if (topRightTree == NULL) {
					topRightTree = new Quad(
						Particle((topLeft.x + botRight.x) / 2, topLeft.y),
						Particle(botRight.x,
						(topLeft.y + botRight.y) / 2));
				}
				this->numberOfParticles++;
				topRightTree->insert(_node);
			}
			// bottom right 
			else {
				if (botRightTree == NULL) {
					botRightTree = new Quad(
						Particle((topLeft.x + botRight.x) / 2, (topLeft.y + botRight.y) / 2),
						Particle(botRight.x, botRight.y));
				}
				this->numberOfParticles++;
				botRightTree->insert(_node);
			}
		}
	}

	bool inBoundary(Particle _particle) {
		return (_particle.x >= topLeft.x &&
			_particle.x <= botRight.x &&
			_particle.y >= topLeft.y &&
			_particle.y <= botRight.y);
	}

	__host__ __device__
	void calculateMass() {
		if (numberOfParticles == 1) {
			centerOfMassX = node->particle.x;
			centerOfMassY = node->particle.y;
			mass = node->particle.mass;
		}
		else {
			if (topLeftTree != NULL) {
				topLeftTree->calculateMass();
				mass += topLeftTree->mass;
				centerOfMassX += topLeftTree->mass * topLeftTree->centerOfMassX;
				centerOfMassY += topLeftTree->mass * topLeftTree->centerOfMassY;
			}
			if (botLeftTree != NULL) {
				botLeftTree->calculateMass();
				mass += botLeftTree->mass;
				centerOfMassX += botLeftTree->mass * botLeftTree->centerOfMassX;
				centerOfMassY += botLeftTree->mass * botLeftTree->centerOfMassY;
			}
			if (topRightTree != NULL) {
				topRightTree->calculateMass();
				mass += topRightTree->mass;
				centerOfMassX += topRightTree->mass * topRightTree->centerOfMassX;
				centerOfMassY += topRightTree->mass * topRightTree->centerOfMassY;
			}
			if (botRightTree != NULL) {
				botRightTree->calculateMass();
				mass += botRightTree->mass;
				centerOfMassX += botRightTree->mass * botRightTree->centerOfMassX;
				centerOfMassY += botRightTree->mass * botRightTree->centerOfMassY;
			}
			centerOfMassX /= mass;
			centerOfMassY /= mass;
		}
	};

	const double G = 6.67 * pow(10, -11);

	__host__ __device__
	double CalculateForce(Particle _particle) {
		double r = 0;
		double d = 0;
		r = sqrt(pow(centerOfMassX - _particle.x, 2) + pow(centerOfMassY - _particle.y, 2));
		d = botRight.x - topLeft.x;

		if (numberOfParticles == 1) {
			// Gravitational force between targetParticle and particle
			force = G * (_particle.mass * mass) / pow(1, 2);
		}
		else {
			if ((d / r) < 0) {
				// Gravitational force between targetParticle and node
				force = G *(_particle.mass * mass) / pow(r, 2);
			}
			else {
				if (topLeftTree != NULL) {
					if (topLeftTree->numberOfParticles >= 1) {
						force += topLeftTree->CalculateForce(_particle);
					}
				}
				if (botLeftTree != NULL) {
					if (botLeftTree->numberOfParticles >= 1) {
						force += botLeftTree->CalculateForce(_particle);
					}
				}
				if (topRightTree != NULL) {
					if (topRightTree->numberOfParticles >= 1) {
						force += topRightTree->CalculateForce(_particle);
					}
				}
				if (botRightTree != NULL) {
					if (botRightTree->numberOfParticles >= 1) {
						force += botRightTree->CalculateForce(_particle);
					}
				}
			}
		}
		return force;
	}

	__host__ __device__
	int newPositionInX(Particle _particle, double force) {

		_particle.velocityX = _particle.velocityX * force;

		return (_particle.velocityX + _particle.x) * 30;
	}

	__host__ __device__
	int newPositionInY(Particle _particle, double force) {

		_particle.velocityY = _particle.velocityY * force;

		return (_particle.velocityY + _particle.y) * 30;
	}
};
