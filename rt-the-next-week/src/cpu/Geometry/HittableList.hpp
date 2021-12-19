//
// Created by 유승우 on 2020/05/15.
//

#ifndef RTFOUNDATION_HITTABLE_LIST_HPP
#define RTFOUNDATION_HITTABLE_LIST_HPP

#include <memory>
#include <vector>

#include "AABB.hpp"
#include "Hittable.hpp"

using std::shared_ptr;
using std::make_shared;

class HittableList: public Hittable
{
public:
    /*
     * Default constructor
     * Creates an instance of HittableList class.
     */
    HittableList() {}

    /*
     * Initialize an instance with a shared_ptr of 'Hittable' object
     */
    HittableList(shared_ptr<Hittable> object) { add(object); }

    /*
     * Clear the std::vector Objects that contains the shared_ptrs of 'Hittable' Objects
     */
    void clear() { Objects.clear(); }

    /*
     * Add an Objects to std::vector Objects
     */
    void add(shared_ptr<Hittable> object) { Objects.push_back(object); }

    /*
     * The member function hit, which is expected to be implemented in a derived class
     */
    bool hit(const Ray &Ray, double DepthMin, double DepthMax, HitRecord &Record) const override;
    bool computeBoundingBox(double t0, double t1, AABB& OutputBoundingBox) const override;

public:
    // dynamic array that stores shard_ptr of <Hittable> Objects
    std::vector<shared_ptr<Hittable>> Objects;
};

bool HittableList::hit(const Ray& Ray, double DepthMin, double DepthMax, HitRecord& Record) const
{
    HitRecord TempRecord;
    bool HitSomething = false;
    auto ClosestDepth = DepthMax;

    for (const auto& object : Objects)
    {
        if (object->hit(Ray, DepthMin, ClosestDepth, TempRecord))
        {
            HitSomething = true;
            ClosestDepth = TempRecord.Depth;
            Record = TempRecord;
        }
    }

    return HitSomething;
}

bool HittableList::computeBoundingBox(double t0, double t1, AABB &OutputBoundingBox) const {
    if (Objects.empty())
        return false;

    AABB tempBoundingBox;
    bool isFirstBox = true;

    for (const auto& object : Objects) {
        if (!object->computeBoundingBox(t0, t1, tempBoundingBox))
            return false;
        OutputBoundingBox = isFirstBox ? tempBoundingBox : computeSurroundingBox(OutputBoundingBox, tempBoundingBox);
        isFirstBox = false;
    }

    return true;
}

#endif //RTFOUNDATION_HITTABLE_LIST_HPP
