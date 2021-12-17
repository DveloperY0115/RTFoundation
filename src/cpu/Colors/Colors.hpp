//
// Created by 유승우 on 2020/05/14.
//

#ifndef RTFOUNDATION_COLOR_HPP
#define RTFOUNDATION_COLOR_HPP

#include "../Math/Vector3.hpp"

#include <iostream>

void writeColor(std::ostream &out, Color PixelColor, int SamplesPerPixel)
{
    auto r = PixelColor.X();
    auto g = PixelColor.Y();
    auto b = PixelColor.Z();

    // Divide the Color total by the number of samples and gamma-correct for gamma = 2.0
    auto scale = 1.0 / SamplesPerPixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Write the translated [0, 255] value of each Color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

void writeColor(
        unsigned int PixelX,
        unsigned int PixelY,
        Color PixelColor,
        unsigned int SamplesPerPixel,
        unsigned int ImageWidth,
        unsigned int ImageHeight,
        int* ImageBuffer) {
    auto r = PixelColor.X();
    auto g = PixelColor.Y();
    auto b = PixelColor.Z();

    // Divide the Color total by the number of samples and gamma-correct for gamma = 2.0
    auto scale = 1.0 / SamplesPerPixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    unsigned int IndexX = PixelX;
    unsigned int IndexY = -PixelY + (ImageHeight - 1);
    unsigned int BufferIndex = 3 * ImageWidth * IndexY + IndexX;

    ImageBuffer[BufferIndex] = static_cast<int>(256 * clamp(r, 0.0, 0.999));
    ImageBuffer[BufferIndex + 1] = static_cast<int>(256 * clamp(g, 0.0, 0.999));
    ImageBuffer[BufferIndex + 2] = static_cast<int>(256 * clamp(b, 0.0, 0.999));
}


#endif //RTFOUNDATION_COLOR_HPP
