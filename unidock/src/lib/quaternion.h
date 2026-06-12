/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>,
           The Olson Lab,
           The Scripps Research Institute

*/

#ifndef VINA_QUATERNION_H
#define VINA_QUATERNION_H

#include <boost/serialization/split_free.hpp>

#include "common.h"
#include "random.h"

// Custom quaternion type replacing boost::math::quaternion<fl>. Boost's
// GPU-enabled math headers (Boost >= 1.87) cannot be compiled by nvcc:
// boost/math/tools/numeric_limits.hpp expands BOOST_MATH_STATIC to `constexpr`
// under __CUDACC__ and then writes `BOOST_MATH_STATIC constexpr ...`, i.e.
// `constexpr constexpr`, which nvcc rejects. Since boost::math::quaternion is
// only pulled into the CUDA translation units via the `qt` type, we reimplement
// the (small) subset of quaternion operations the code uses. The same type is
// used by host and device translation units so there is no ABI mismatch.
// Component order and operation order match boost::math::quaternion.
struct qt {
    fl a, b, c, d;
    qt() : a(0), b(0), c(0), d(0) {}
    qt(fl a_, fl b_, fl c_, fl d_) : a(a_), b(b_), c(c_), d(d_) {}
    fl R_component_1() const { return a; }
    fl R_component_2() const { return b; }
    fl R_component_3() const { return c; }
    fl R_component_4() const { return d; }
    qt& operator*=(fl rhs) {
        a *= rhs;
        b *= rhs;
        c *= rhs;
        d *= rhs;
        return *this;
    }
    qt& operator/=(fl rhs) {
        a /= rhs;
        b /= rhs;
        c /= rhs;
        d /= rhs;
        return *this;
    }
    qt& operator*=(const qt& rhs) {  // Hamilton product (matches boost::math::quaternion)
        const fl ar = rhs.a, br = rhs.b, cr = rhs.c, dr = rhs.d;
        const qt result(a * ar - b * br - c * cr - d * dr, a * br + b * ar + c * dr - d * cr,
                        a * cr - b * dr + c * ar + d * br, a * dr + b * cr - c * br + d * ar);
        *this = result;
        return *this;
    }
    qt& operator/=(const qt& rhs) {  // *this * conj(rhs) / norm_sqr(rhs)
        const fl nrm2 = rhs.a * rhs.a + rhs.b * rhs.b + rhs.c * rhs.c + rhs.d * rhs.d;
        const qt conj(rhs.a, -rhs.b, -rhs.c, -rhs.d);
        *this *= conj;
        *this /= nrm2;
        return *this;
    }
};
inline qt operator*(const qt& lhs, const qt& rhs) {
    qt result(lhs);
    result *= rhs;
    return result;
}
inline qt operator*(fl lhs, const qt& rhs) {
    return qt(lhs * rhs.R_component_1(), lhs * rhs.R_component_2(), lhs * rhs.R_component_3(),
              lhs * rhs.R_component_4());
}
inline qt operator*(const qt& lhs, fl rhs) { return rhs * lhs; }
// Magnitude, equivalent to the former boost::math::abs(q) (scaled for overflow safety).
inline fl quaternion_norm(const qt& q) {
    const fl maxim
        = (std::max)((std::max)(std::abs(q.R_component_1()), std::abs(q.R_component_2())),
                     (std::max)(std::abs(q.R_component_3()), std::abs(q.R_component_4())));
    if (maxim == static_cast<fl>(0)) return maxim;
    const fl mixam = static_cast<fl>(1) / maxim;
    fl a = q.R_component_1() * mixam;
    fl b = q.R_component_2() * mixam;
    fl c = q.R_component_3() * mixam;
    fl d = q.R_component_4() * mixam;
    a *= a;
    b *= b;
    c *= c;
    d *= d;
    return maxim * std::sqrt(a + b + c + d);
}

// non-intrusive free function split serialization
namespace boost {
    namespace serialization {
        template <class Archive> void save(Archive& ar, const qt& q, const unsigned version) {
            fl q1 = q.R_component_1();
            fl q2 = q.R_component_2();
            fl q3 = q.R_component_3();
            fl q4 = q.R_component_4();

            ar & q1;
            ar & q2;
            ar & q3;
            ar & q4;
        }
        template <typename Archive> void load(Archive& ar, qt& q, const unsigned version) {
            fl a, b, c, d;
            ar & a;
            ar & b;
            ar & c;
            ar & d;
            q = qt(a, b, c, d);
        }
    }  // namespace serialization
}  // namespace boost
BOOST_SERIALIZATION_SPLIT_FREE(qt)

bool eq(
    const qt& a,
    const qt& b);  // elementwise approximate equality - may return false for equivalent rotations
const qt qt_identity(1, 0, 0, 0);
qt angle_to_quaternion(const vec& axis, fl angle);  // axis is assumed to be a unit vector
qt angle_to_quaternion(const vec& rotation);        // rotation == angle * axis
vec quaternion_to_angle(const qt& q);
mat quaternion_to_r3(const qt& q);
bool quaternion_is_normalized(const qt& q);

inline fl quaternion_norm_sqr(const qt& q) {  // equivalent to sqr(boost::math::abs(const qt&))
    return sqr(q.R_component_1()) + sqr(q.R_component_2()) + sqr(q.R_component_3())
           + sqr(q.R_component_4());
}

inline void quaternion_normalize(qt& q) {
    const fl s = quaternion_norm_sqr(q);
    assert(eq(s, sqr(quaternion_norm(q))));
    const fl a = std::sqrt(s);
    assert(a > epsilon_fl);
    q *= 1 / a;
    assert(quaternion_is_normalized(q));
}

inline void quaternion_normalize_approx(qt& q, const fl tolerance = 1e-6) {
    const fl s = quaternion_norm_sqr(q);
    assert(eq(s, sqr(quaternion_norm(q))));
    if (std::abs(s - 1) < tolerance)
        ;  // most likely scenario
    else {
        const fl a = std::sqrt(s);
        assert(a > epsilon_fl);
        q *= 1 / a;
        assert(quaternion_is_normalized(q));
    }
}

qt random_orientation(rng& generator);
void quaternion_increment(qt& q, const vec& rotation);
vec quaternion_difference(const qt& b,
                          const qt& a);  // rotation that needs to be applied to convert a to b
void print(const qt& q, std::ostream& out = std::cout);  // print as an angle

#endif
