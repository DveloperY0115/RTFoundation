//
// Created by 유승우 on 2021/12/19.
//

#ifndef RTFOUNDATION_BVH_HPP
#define RTFOUNDATION_BVH_HPP

#include <algorithm>

#include "../rtweekend.hpp"
#include "Hittable.hpp"
#include "HittableList.hpp"

bool compareBoundingBoxAlongX(const shared_ptr<Hittable> HittableA, const shared_ptr<Hittable> HittableB);
bool compareBoundingBoxAlongY(const shared_ptr<Hittable> HittableA, const shared_ptr<Hittable> HittableB);
bool compareBoundingBoxAlongZ(const shared_ptr<Hittable> HittableA, const shared_ptr<Hittable> HittableB);

class BVHNode : public Hittable {
public:
    BVHNode();
    BVHNode(const HittableList& HittableList, double t0, double t1)
    : BVHNode(HittableList.Objects, 0, HittableList.Objects.size(),
              t0, t1)
    {
        // Do nothing.
    }

    BVHNode(const std::vector<shared_ptr<Hittable>>& SourceObjects, size_t StartIndex, size_t EndIndex, double t0, double t1);

    bool hit(const Ray& Ray, double DepthMin, double DepthMax, HitRecord& Record) const override;

    bool computeBoundingBox(double t0, double t1, AABB& OutputBoundingBox) const override;

public:
    shared_ptr<Hittable> LeftChild, RightChild;
    AABB BVHNodeBoundingBox;
};

BVHNode::BVHNode(const std::vector<shared_ptr<Hittable>> &SourceObjects, size_t StartIndex, size_t EndIndex, double t0,
                 double t1) {
    auto Objects = SourceObjects;

    // Select a random axis to split the BVHs into two subtrees
    int Axis = generateRandomInt(0, 2);
    auto Comparator = \
            (Axis == 0) ? compareBoundingBoxAlongX    // divide BVHs along X-axis
            : (Axis == 1) ? compareBoundingBoxAlongY // divide BVHs along Y-axis
            : compareBoundingBoxAlongZ;  // divide BVHs along Z-axis

    size_t ObjectSpan = EndIndex - StartIndex;

    if (ObjectSpan == 1) {
        LeftChild = Objects[StartIndex];
        RightChild = Objects[StartIndex];
    } else if (ObjectSpan == 2) {
        if (Comparator(Objects[StartIndex], Objects[StartIndex+1])) {
            LeftChild = Objects[StartIndex];
            RightChild = Objects[StartIndex+1];
        } else {
            LeftChild = Objects[StartIndex+1];
            RightChild = Objects[StartIndex];
        }
    } else {
        std::sort(Objects.begin() + StartIndex, Objects.begin() + EndIndex, Comparator);

        auto MidIndex = StartIndex + ObjectSpan / 2;
        LeftChild = make_shared<BVHNode>(Objects, StartIndex, MidIndex, t0, t1);
        RightChild = make_shared<BVHNode>(Objects, MidIndex, EndIndex, t0, t1);
    }

    AABB LeftBoundingBox, RightBoundingBox;

    if (!LeftChild->computeBoundingBox(t0, t1, LeftBoundingBox) || !RightChild->computeBoundingBox(t0, t1, RightBoundingBox))
        std::cerr << "Cannot compute bounding box(es) for one (or two) object(s)!\n";

    BVHNodeBoundingBox = computeSurroundingBox(LeftBoundingBox, RightBoundingBox);
}

bool BVHNode::hit(const Ray &Ray, double DepthMin, double DepthMax, HitRecord &Record) const {
    if(!BVHNodeBoundingBox.hit(Ray, DepthMin, DepthMax))
        return false;

    bool isHitLeft = LeftChild->hit(Ray, DepthMin, DepthMax, Record);
    bool isHitRight = RightChild->hit(Ray, DepthMin, isHitLeft ? Record.Depth : DepthMax, Record);

    return isHitLeft || isHitRight;
}

bool BVHNode::computeBoundingBox(double t0, double t1, AABB &OutputBoundingBox) const {
    OutputBoundingBox = BVHNodeBoundingBox;
    return true;
}

inline bool compareBoundingBox(const shared_ptr<Hittable> HittableA, const shared_ptr<Hittable> HittableB, int Axis) {
    AABB BoundingBoxA, BoundingBoxB;

    if (!HittableA->computeBoundingBox(0, 0, BoundingBoxA) || !HittableB->computeBoundingBox(0, 0, BoundingBoxB))
        std::cerr << "Cannot compute bounding box(es) for one (or two) object(s)!\n";

    return BoundingBoxA.getMin()[Axis] < BoundingBoxB.getMin()[Axis];
}

bool compareBoundingBoxAlongX(const shared_ptr<Hittable> HittableA, const shared_ptr<Hittable> HittableB) {
    return compareBoundingBox(HittableA, HittableB, 0);
}

bool compareBoundingBoxAlongY(const shared_ptr<Hittable> HittableA, const shared_ptr<Hittable> HittableB) {
    return compareBoundingBox(HittableA, HittableB, 1);
}

bool compareBoundingBoxAlongZ(const shared_ptr<Hittable> HittableA, const shared_ptr<Hittable> HittableB) {
    return compareBoundingBox(HittableA, HittableB, 2);
}

#endif //RTFOUNDATION_BVH_HPP
