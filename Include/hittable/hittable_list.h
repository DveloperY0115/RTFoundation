//
// Created by 유승우 on 2020/05/15.
//

#ifndef FIRSTRAYTRACER_HITTABLE_LIST_H
#define FIRSTRAYTRACER_HITTABLE_LIST_H

#include "../Include/hittable/hittable.h"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list: public hittable
{
public:
    /*
     * Default constructor
     * Creates an instance of hittable_list class.
     */
    hittable_list() {}

    /*
     * Initialize an instance with a shared_ptr of 'hittable' object
     */
    hittable_list(shared_ptr<hittable> object) { add(object); }

    /*
     * Clear the std::vector objects that contains the shared_ptrs of 'hittable' objects
     */
    void clear() { objects.clear(); }

    /*
     * Add an objects to std::vector objects
     */
    void add(shared_ptr<hittable> object) { objects.push_back(object); }

    /*
     * The member function hit, which is expected to be implemented in a derived class
     */
    virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const;

public:
    // dynamic array that stores shard_ptr of <hittable> objects
    std::vector<shared_ptr<hittable>> objects;
};

#endif //FIRSTRAYTRACER_HITTABLE_LIST_H
