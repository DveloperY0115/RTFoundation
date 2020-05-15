//
// Created by 유승우 on 2020/05/15.
//

#include "../Include/hittable/hittable_list.h"

/*
 *
 */
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