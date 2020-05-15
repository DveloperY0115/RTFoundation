//
// Created by 유승우 on 2020/05/15.
//

#ifndef FIRSTRAYTRACER_HITTABLE_LIST_HPP
#define FIRSTRAYTRACER_HITTABLE_LIST_HPP

#include "hittable.hpp"

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

bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    /*
     * What if a computer tries to draw an object which is behind of other objects?
     * It doesn't seem realistic. That what this member variable 'closest_so_far' stands for.
     */
    auto closest_so_far = t_max;

    for (const auto& object : objects)
    {
        /*
         * Object is a shared_ptr, it's basically a pointer.
         * Therefore to access the member function of this object,
         * use '->' instead of '.'.
         */
        if (object->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif //FIRSTRAYTRACER_HITTABLE_LIST_HPP
