//
// Created by 유승우 on 2020/05/14.
//

#ifndef FIRSTRAYTRACER_COLOR_HPP
#define FIRSTRAYTRACER_COLOR_HPP

#include "Vector3.hpp"

#include <iostream>

void write_color(std::ostream &out, Color pixel_color, int samples_per_pixel)
{
    auto r = pixel_color.X();
    auto g = pixel_color.Y();
    auto b = pixel_color.Z();

    // Divide the Color total by the number of samples and gamma-correct for gamma = 2.0
    auto scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Write the translated [0, 255] value of each Color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}


#endif //FIRSTRAYTRACER_COLOR_HPP
