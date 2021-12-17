//
// Created by 유승우 on 2020/05/15.
//

#ifndef RTFOUNDATION_HITTABLE_LIST_HPP
#define RTFOUNDATION_HITTABLE_LIST_HPP

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
    virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;

public:
    // dynamic array that stores shard_ptr of <hittable> objects
    std::vector<shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (const auto& object : objects)
    {
        if (object->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif //RTFOUNDATION_HITTABLE_LIST_HPP
