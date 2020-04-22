#include <iostream>
#include <fstream>
#include "vector3/vector3.h"
#include "ray/ray.h"

using namespace std;

// If a ray from camera to projected image hits a sphere during transition,
// color that projection red
double hit_sphere(const vector3& center, double radius, const ray& r)
{
    vector3 oc = r.origin() - center;
    double a = dot_product(r.direction(), r.direction());
    double b = 2.0 * dot_product(oc, r.direction());
    double c = dot_product(oc, oc) - radius * radius;
    double discriminant = b*b - 4*a*c;
    if (discriminant < 0)
    {
        return -1.0;
    }

    else
    {
        // we're only concerned about the first hit point. Not the one on the other side of the sphere.
        return (-b - sqrt(discriminant)) / (2.0 * a);
    }
}

vector3 color(const ray& r)
{
    double t = hit_sphere(vector3(0, 0, -1), 0.5, r);
    if (t > 0.0)
    {
        vector3 N = unit_vector(r.point_at_parameter(t) - vector3(0, 0, -1));
        return 0.5 * vector3(N.getX() + 1, N.getY() + 1, N.getZ() + 1);
    }

    // making background
    vector3 unit_direction = unit_vector(r.direction());
    t = 0.5 * (unit_direction.getY() + 1.0);
    return (1.0 - t) * vector3(1.0, 1.0, 1.0) + t * vector3(0.5, 0.7, 1.0);
}

int main() {
    vector3 vec_Test = vector3(1, 1, 1);

    int nx = 200;
    int ny = 100;

    ofstream writeFile;

    writeFile.open("Hello_My_First_Camera.ppm");
    writeFile << "P3\n" << nx << " " << ny << "\n255\n";

    vector3 lower_left_corner(-2.0, -1.0, -1.0);
    vector3 horizontal(4.0, 0.0, 0.0);
    vector3 vertical(0.0, 2.0, 0.0);
    vector3 origin(0.0, 0.0, 0.0);

    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            double u = double(i) / double(nx);
            double v = double(j) / double(ny);

            ray r(origin, lower_left_corner + u * horizontal + v * vertical);
            vector3 col = color(r);

            int ir = int(255.99*col.getR());
            int ig = int(255.99*col.getG());
            int ib = int(255.99*col.getB());

            cout << ir << " " << ig << " " << ib << "\n";
            writeFile << ir << " " << ig << " " << ib << "\n";
        }
    }
    /*
    ofstream writeFile;
    writeFile.open("Hello_Graphics.ppm");
    int nx = 200;
    int ny = 100;


    writeFile << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            vector3 col(double(i) / double(nx), double(j) / double(ny), double((i+j)/2) / double((nx + ny) / 2));
            int ir = int(255.99*col.getR());
            int ig = int(255.99*col.getG());
            int ib = int(255.99*col.getB());

            writeFile << ir << " " << ig << " " << ib << "\n";
        }
    }
    writeFile.close();
    */
}
