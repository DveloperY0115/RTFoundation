//
// Created by 유승우 on 2021/12/20.
//

#ifndef RTFOUNDATION_IMAGETEXTURE_HPP
#define RTFOUNDATION_IMAGETEXTURE_HPP

#include <iostream>

#include "../rtweekend.hpp"
#include "Texture.hpp"
#include "../Utilities/rtw_stb_image.hpp"

class ImageTexture : public Texture {
public:
    const static int BytesPerPixel = 3;

    ImageTexture()
    : Data(nullptr), TextureImageWidth(0), TextureImageHeight(0), BytesPerScanline(0)
    {
        // Do nothing.
    }

    ImageTexture(const char* ImageFilename) {
        auto ComponentsPerPixel = BytesPerPixel;

        Data = stbi_load(
                ImageFilename, &TextureImageWidth, &TextureImageHeight, & ComponentsPerPixel, ComponentsPerPixel);

        if (!Data) {
            std::cerr << "ERROR: Could not load texture image file '" << ImageFilename << "'.\n";
            TextureImageWidth = TextureImageHeight = 0;
        }

        // Number of bytes loaded per scanline
        BytesPerScanline = BytesPerPixel * TextureImageWidth;
    }

    ~ImageTexture() {
        delete Data;
    }

    Color getTexelColor(double u, double v, const Point3& SurfacePoint) const override {
        if (Data == nullptr)
            // if the image file is missing, return cyan color to indicate that something is missing
            return Color(0, 1,1);

        u = clamp(u, 0.0, 1.0);
        v = 1.0 - clamp(v, 0.0, 1.0);  // Flip V to image coordinate. The origin is located in the top left corner in images

        auto TextureImageX = static_cast<int>(u * TextureImageWidth);
        auto TextureImageY = static_cast<int>(v * TextureImageHeight);

        // Clamp the result of mapping from normalized texture coordinate to pixel coordinate
        if (TextureImageX >= TextureImageWidth)
            TextureImageX = TextureImageWidth - 1;
        if (TextureImageY >= TextureImageHeight)
            TextureImageY = TextureImageHeight - 1;

        const double ColorScale = 1.0 / 255.0;
        auto PixelAtXY = Data + TextureImageY * BytesPerScanline + TextureImageX * BytesPerPixel;

        return ColorScale * Color(PixelAtXY[0], PixelAtXY[1], PixelAtXY[2]);
    }

private:
    unsigned char* Data;
    int TextureImageWidth, TextureImageHeight;
    int BytesPerScanline;
};

#endif //RTFOUNDATION_IMAGETEXTURE_HPP
