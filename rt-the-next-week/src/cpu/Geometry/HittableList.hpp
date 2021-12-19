//
// Created by 유승우 on 2020/05/15.
//

#ifndef RTFOUNDATION_HITTABLE_LIST_HPP
#define RTFOUNDATION_HITTABLE_LIST_HPP

#include "Hittable.hpp"

#include <memory>
#include <vector>

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
     * Clear the std::vector objects that contains the shared_ptrs of 'Hittable' objects
     */
    void clear() { objects.clear(); }

    /*
     * Add an objects to std::vector objects
     */
    void add(shared_ptr<Hittable> object) { objects.push_back(object); }

    /*
     * The member function Hit, which is expected to be implemented in a derived class
     */
    virtual bool Hit(const Ray &Ray, double DepthMin, double DepthMax, HitRecord &Record) const override;

public:
    // dynamic array that stores shard_ptr of <Hittable> objects
    std::vector<shared_ptr<Hittable>> objects;
};

bool HittableList::Hit(const Ray& Ray, double DepthMin, double DepthMax, HitRecord& Record) const
{
    HitRecord TempRecord;
    bool HitSomething = false;
    auto ClosestDepth = DepthMax;

    // TODO: Optimize this using BVH, kd-Tree, etc
    for (const auto& object : objects)
    {
        if (object->Hit(Ray, DepthMin, ClosestDepth, TempRecord))
        {
            HitSomething = true;
            ClosestDepth = TempRecord.Depth;
            Record = TempRecord;
        }
    }

    return HitSomething;
}

#endif //RTFOUNDATION_HITTABLE_LIST_HPP
