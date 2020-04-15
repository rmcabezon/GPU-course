#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> TimePoint;
typedef std::chrono::duration<double> Time;

#define PI 3.14159265358979323846

using namespace std;

// Ignore this
double elapsedTime(TimePoint start, TimePoint stop)
{
	return std::chrono::duration_cast<Time>(stop-start).count();
}

// Also this
double compute_3d_k(double n)
{
    double b0 = 2.7012593e-2;
    double b1 = 2.0410827e-2;
    double b2 = 3.7451957e-3;
    double b3 = 4.7013839e-2;

    return b0 + b1 * std::sqrt(n) + b2 * n + b3 * std::sqrt(n * n * n);
}

// This is the SPH Kernel, but you can ignore it too
double wharmonic(double v)
{
    if (v == 0.0) return 1.0;
    const double Pv = (PI / 2.0) * v;
    return std::sin(Pv) / Pv;
}

// Structure that contains the position, mass and neighbors of every particles
struct ParticleData
{
	ParticleData(size_t n, size_t ngmax) : n(n), ngmax(ngmax), x(n), y(n), z(n), h(n), m(n), ro(n), neighborsCount(n), neighbors(n * ngmax) {}
	
	size_t n, ngmax;
	vector<double> x, y, z, h, m, ro;
	vector<int> neighborsCount, neighbors;
};

// Compute density here
void compute_density(ParticleData &particles)
{
	size_t n = particles.n;
	size_t ngmax = particles.ngmax;

	const double *h = particles.h.data();
    const double *m = particles.m.data();
    const double *x = particles.x.data();
    const double *y = particles.y.data();
    const double *z = particles.z.data();

	const int *neighbors = particles.neighbors.data();
	const int *neighborsCount = particles.neighborsCount.data();

	double *ro = particles.ro.data();

    const double K = compute_3d_k(6.0);

    // OpenMP and offloading directives go here!
    for (size_t i = 0; i < n; i++)
    {
        const int nn = neighborsCount[i];

        double roloc = 0.0;

        for (int pj = 0; pj < nn; pj++)
        {
            const int j = neighbors[i * ngmax + pj];

            double xx = x[i] - x[j];
		    double yy = y[i] - y[j];
		    double zz = z[i] - z[j];

		    double dist = std::sqrt(xx * xx + yy * yy + zz * zz);

		    // SPH Kernel
            double vloc = wharmonic(dist / h[i]);

            const double w = K * vloc * vloc * vloc * vloc * vloc * vloc;
            const double value = w / (h[i] * h[i] * h[i]);

            roloc += value * m[j];
        }

        ro[i] = roloc + m[i] * K / (h[i] * h[i] * h[i]);
	}
}

// You can pretty much ignore this one too
// It reads the input file
// Initializes the ParticleData structure
// Calls compute_density
// Write the result in out.txt 
int main(int argc, char **argv)
{
	// Read input file
    std::ifstream in;
    in.open("/home/acavelan/git/miniapp/pdata", std::ofstream::out | std::ofstream::binary);

    if(in.is_open() == false)
    {
    	cerr << "Error opening file pdata" << endl;
    	return -1;
    }

	size_t n = 0;
    size_t ngmax = 0;

    in.read((char *)&n, 1 * sizeof(size_t));
    in.read((char *)&ngmax, 1 * sizeof(size_t));

 	ParticleData p(n, ngmax);

    in.read((char *)&p.x[0], p.x.size() * sizeof(p.x[0]));
    in.read((char *)&p.y[0], p.y.size() * sizeof(p.y[0]));
    in.read((char *)&p.z[0], p.z.size() * sizeof(p.z[0]));
    in.read((char *)&p.h[0], p.h.size() * sizeof(p.h[0]));
    in.read((char *)&p.m[0], p.m.size() * sizeof(p.m[0]));
    in.read((char *)&p.neighborsCount[0], p.neighborsCount.size() * sizeof(p.neighborsCount[0]));
    in.read((char *)&p.neighbors[0], p.neighbors.size() * sizeof(p.neighbors[0]));

    in.close();

    // Call the main function with a timer
   	TimePoint tstart = Clock::now();
	compute_density(p);
	TimePoint tstop = Clock::now();

	cout << elapsedTime(tstart, tstop) << endl;

   	// Write the result
   	std::ofstream out;
    out.open("density.txt", std::ofstream::out);

    if(out.is_open() == false)
    {
    	cerr << "Error opening file out" << endl;
    	return -1;
    }

    for (size_t i = 0; i < n; i++)
    {
    	out << p.x[i] << " " << p.y[i] << " " << p.z[i] << " " << p.ro[i] << endl;
    }

    out.close();

	return 0;
}