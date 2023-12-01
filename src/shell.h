// written by Shinji Ogaki
#pragma once

#include "glm/glm/glm.hpp"
#include "glm/glm/gtx/compatibility.hpp"

#define CY_NO_INTRIN_H
#include "cyCodeBase/cyPolynomial.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace shell
{
// Types
//#define double_precision
#ifdef double_precision
using real = double;
using vec2 = glm::f64vec2;
using vec3 = glm::f64vec3;
using vec4 = glm::f64vec4;
using mat3 = glm::f64mat3;
#else
using real = float;
using vec2 = glm::f32vec2;
using vec3 = glm::f32vec3;
using vec4 = glm::f32vec4;
using mat3 = glm::f32mat3;
#endif

// Constants
const auto epsilon  = real(1.0e-6);
const auto delta    = real(1.0e-4);
const auto infinity = real(3.4e38);

using namespace std;

// Not used in the paper
// "Accurate Differences of Products with Kahan’s Algorithm"
// https://pharr.org/matt/blog/2019/11/03/difference-of-floats
inline auto ab_minus_cd(const real a, const real b, const real c, const real d)
{
	/*
	double w = d * c;
	double e = std::fma(-d, c,  w);
	double f = std::fma( a, b, -w);
	return f + e;
	*/
	return std::fma(a, b, - c * d);
}

inline auto ab_plus_cd(const real a, const real b, const real c, const real d)
{
	/*
	double w = d * c;
	double e = std::fma(d, c, -w);
	double f = std::fma(a, b,  w);
	return f + e;
	*/
	return std::fma(a, b, c * d);
}

inline auto _fma(const vec2& a, const real b, const vec2& c)
{
	return vec2(std::fma(a.x, b, c.x), std::fma(a.y, b, c.y));
}

inline auto _fma(const vec3& a, const real b, const vec3& c)
{
	return vec3(std::fma(a.x, b, c.x), std::fma(a.y, b, c.y), std::fma(a.z, b, c.z));
}

inline auto _cross(const vec2 &l, const vec2 &r)
{
	return ab_minus_cd(l.x, r.y, l.y, r.x);
}

inline auto _cross(const vec3& v1, const vec3& v2)
{
	return vec3(ab_minus_cd(v1.y, v2.z, v1.z, v2.y), ab_minus_cd(v1.z, v2.x, v1.x, v2.z), ab_minus_cd(v1.x, v2.y, v1.y, v2.x));
}

inline auto _dot(const vec3& l, const vec3& r)
{
	return std::fma(l.x, r.x, std::fma(l.y, r.y, l.z * r.z));
}

inline auto _normalize(const vec3& v)
{
	return v / sqrt(_dot(v, v));
}

// Solve a x^2 + 2 b x + c for x
// cy::QuadraticRoots is not used because it is unnecessary to sort the roots
inline auto solve_quadratic(const real a, const real b, const real c, real &x1, real &x2)
{
	if(a == 0)
	{
		if(b == 0)
		{
			return 0;
		}
		x1 = - c / (2 * b);
		return 1;
	}
	const auto dis = ab_minus_cd(b, b, a, c);
	if (0 > dis)
	{
		return 0;
	}
	const auto tmp = -b - copysign(sqrt(dis), b);
	x1 = c / tmp;
	x2 = tmp / a;
	return 2;
}

// Appendix C in "Tessellation-Free Displacement Mapping for Ray Tracing"
inline auto triangle_box(const vec2 uvs[3], const vec2 edge_normals[3], const vec3 &min, const vec3 &max)
{
	const auto c = vec2((min.x + max.x) / 2, (min.y + max.y) / 2);
	const auto w = vec2((max.x - min.x) / 2, (max.y - min.y) / 2);
	for(auto i = 0; i < 3; ++i)
	{
		const auto b = vec2(copysign(w.x, -edge_normals[i].x), copysign(w.y, -edge_normals[i].y));
		if(0 < dot(edge_normals[i], b + c - uvs[i]))
		{
			return false;
		}
	}
	return true;
}

// Eq.1
inline void get_S(vec3 S[3], const vec3 Pb[3], const vec3 VN[3], const real h)
{
	S[0] = _fma(VN[0], h, Pb[0]);
	S[1] = _fma(VN[1], h, Pb[1]);
	S[2] = _fma(VN[2], h, Pb[2]);
}

inline real get_signed_distance(const vec3 &origin, const vec3 &omega, const vec3 &world)
{
	return _dot(omega, world - origin);
}

inline auto major(const vec3 &v)
{
	auto size = abs(v.x); auto axis = 0;
	if(size < abs(v.y)) { axis = 1; size = abs(v.y); }
	if(size < abs(v.z)) { axis = 2; }
	return axis;
}

inline auto minor(const vec3 &v)
{
	auto size = abs(v.x); auto axis = 0;
	if(size > abs(v.y)) { axis = 1; size = abs(v.y); }
	if(size > abs(v.z)) { axis = 2; }
	return axis;
}

struct ray_lines
{
	real eE0, eE1, eE2, eE3;
	real eN0, eN1, eN2, eN3;
	real eo0, eo1;
	real eS0, eS1;

	// Eq.6 Convert a ray to 2 lines
	inline void convert_ray_to_lines(const vec3 &origin, const vec3 &omega, const vec3 Pb[3], const vec3 VN[3])
	{
		// "Building an Orthonormal Basis, Revisited" gives slightly poor results
		// http://jcgt.org/published/0006/01/01/

		static const vec3 axes[3] = { vec3(1,0,0), vec3(0,1,0), vec3(0,0,1) };
		const auto ex = axes[minor(omega)];
		const auto e1 = _normalize(_cross(omega, ex));
		const auto e0 = _normalize(_cross(e1, omega));
		
		// M00 M01: eE0 + h * eN0, eE1 + h * eN1
		// M10 M11: eE2 + h * eN2, eE3 + h * eN3
		eE0 = _dot(e0, Pb[1] - Pb[0]); eE1 = _dot(e0, Pb[2] - Pb[0]);
		eE2 = _dot(e1, Pb[1] - Pb[0]); eE3 = _dot(e1, Pb[2] - Pb[0]);

		eN0 = _dot(e0, VN[1] - VN[0]); eN1 = _dot(e0, VN[2] - VN[0]);
		eN2 = _dot(e1, VN[1] - VN[0]); eN3 = _dot(e1, VN[2] - VN[0]);
		
		// RHS: eO0 - h * eS0
		// RHS: eO1 - h * eS1
		eo0 = _dot(e0, origin - Pb[0]); eS0 = _dot(e0, VN[0]);
		eo1 = _dot(e1, origin - Pb[0]); eS1 = _dot(e1, VN[0]);
	}

	// Eq.7 Represent a ray as quadratic rational functions
	inline auto convert_ray_to_rational() const
	{
		// alpha(h)
		const auto a2 = ab_minus_cd(eN1, eS1, eN3, eS0);
		const auto a1 = ab_minus_cd(eN3, eo0, eE3, eS0) + ab_minus_cd(eE1, eS1, eN1, eo1);
		const auto a0 = ab_minus_cd(eE3, eo0, eE1, eo1);
		
		// beta(h)
		const auto b2 = ab_minus_cd(eN2, eS0, eN0, eS1);
		const auto b1 = ab_minus_cd(eE2, eS0, eN2, eo0) + ab_minus_cd(eN0, eo1, eE0, eS1);
		const auto b0 = ab_minus_cd(eE0, eo1, eE2, eo0);
		
		// d(h)
		const auto d2 = ab_minus_cd(eN0, eN3, eN1, eN2);
		const auto d1 = ab_minus_cd(eE0, eN3, eE1, eN2) + ab_minus_cd(eN0, eE3, eN1, eE2);
		const auto d0 = ab_minus_cd(eE0, eE3, eE1, eE2);
		
		return make_tuple(vec3(a0, a1, a2), vec3(b0, b1, b2), vec3(d0, d1, d2));
	}
};

inline auto eval_quad(const vec3 &coef, const real param)
{
	return fma(fma(coef[2], param, coef[1]), param, coef[0]);
}

inline auto get_tuv(const real param, const real denom, const vec3& nu, const vec3 &nv)
{
	return vec2(eval_quad(nu, param) / denom, eval_quad(nv, param) / denom);
}

inline auto interpolate(const vec2 v[3], const real alpha, const real beta)
{
	return _fma(v[1] - v[0], alpha, _fma(v[2] - v[0], beta, v[0]));
}

inline auto interpolate(const vec3 v[3], const real alpha, const real beta)
{
	return _fma(v[1] - v[0], alpha, _fma(v[2] - v[0], beta, v[0]));
}

inline auto in_range(const real param, const real range_min, const real range_max)
{
	return (range_min <= param && param <= range_max);
}

inline auto in_range(const real param, const real range_min, const real range_max, const real e)
{
	return (range_min - e <= param && param <= range_max + e);
}

inline void update_h(vec4& range_min, vec4& range_max, const real h)
{
	range_min.z = std::min(range_min.z, h);
	range_max.z = std::max(range_max.z, h);
}

inline void update_param(vec4& range_min, vec4& range_max, const real param)
{
	range_min.w = std::min(range_min.w, param);
	range_max.w = std::max(range_max.w, param);
}

inline void update_range(const vec3 &origin, const vec3 &omega, const vec3 Pb[3], const vec3 VN[3], const real h, const real alpha, const real beta, vec2& range)
{
	vec3 S[3]; get_S(S, Pb, VN, h);
	const auto world = interpolate(S, alpha, beta);
	const auto param = get_signed_distance(origin, omega, world);
	range.x = std::min(range.x, param);
	range.y = std::max(range.y, param);
}

inline auto intersect_plane(const vec3 &n, const vec3 &d, const real C, real roots[2])
{
	const auto a = fma(d[2], -C, n[2]);
	const auto b = fma(d[1], -C, n[1]);
	const auto c = fma(d[0], -C, n[0]);
	if(a == 0 && b == 0 && c == 0)
	{
		return 0;
	}
	return solve_quadratic(a * 2, b, c * 2, roots[0], roots[1]);
}

inline auto intersect_plane(const vec3 &origin, const vec3 &omega, const vec3 P[3], real& param, real& alpha, real& beta, real& gamma)
{
	const auto normal = _cross(P[1] - P[0], P[2] - P[0]);
	const auto ip = _dot(normal, omega);
	if (0 == ip)
	{
		return false;
	}
	const auto A = _dot(normal, normal);
	const auto s = P[0] - origin;
	const auto t = _dot(normal, s) / ip;
	const auto I = _fma(omega, t, -s);
	const auto c = _cross(normal, I);
	
	// Gets updated only when intersected
	param = t;
	alpha =  _dot(c, P[2] - P[0]) / A;
	beta  = -_dot(c, P[1] - P[0]) / A;
	gamma =  1 - (alpha + beta);
	
	return true;
}

inline auto intersect_triangle(const vec3 &origin, const vec3 &omega, const vec3 P[3], real& param, real& alpha, real& beta, real& gamma)
{
	const auto normal = _cross(P[1] - P[0], P[2] - P[0]);
	const auto ip = _dot(normal, omega);
	if (0 == ip)
	{
		return false;
	}
	const auto A = _dot(normal, normal);
	const auto s = P[0] - origin;
	const auto t = _dot(normal, s) / ip;
	const auto I = _fma(omega, t, -s);
	const auto c = _cross(normal, I);
	
	const auto u = _dot(c, P[2] - P[0]) / A;
	const auto v = _dot(c, P[0] - P[1]) / A;
	const auto w = 1 - (u + v);
	
	if (0 > u || 0 > v || 0 > w)
	{
		return false;
	}
	
	param = t;
	alpha = u;
	beta  = v;
	gamma = w;
	
	return true;
}

inline auto intersect_triangle(const vec3 &origin, const vec3 &omega, const vec3 P[3], vec4 &box_min, vec4 &box_max)
{
	real param, alpha, beta, gamma;
	if(intersect_triangle(origin, omega, P, param, alpha, beta, gamma))
	{
		update_param(box_min, box_max, param);
		return true;
	}
	return false;
}

// "Cool Patches: A Geometric Approach to Ray/Bilinear Patch Intersections"
// This function was written by Alexander Reshetov and modified by Shinji Ogaki.
inline void intersect_cool_patch(const vec3& origin,
								 const vec3& omega,
								 const vec3& Q00,
								 const vec3& q01,
								 const vec3& Q10,
								 const vec3& q11,
								 vec4 &box_min,
								 vec4 &box_max)
{
	// 01 ----------- 11
	// |               |
	// | e00       e11 |
	// |      e10      |
	// 00 ----------- 10
	const auto e10 = Q10 - Q00;
	const auto e11 = q11 - Q10;
	const auto e00 = q01 - Q00;
	
	const auto q00 = Q00 - origin;
	const auto q10 = Q10 - origin;
	
	// a + b u + c u^2
	auto a = _dot(_cross(q00      , omega), e00);
	auto b = _dot(_cross(q10      , omega), e11);
	auto c = _dot(_cross(q01 - q11, omega), e10);
	
	b -= a + c;
	const auto det = fma(b, b, - 4 * a * c);
	if (0 > det)
	{
		return;
	}
	
	// Solve for u
	real u1, u2;
	if (c == 0) // trapezoid
	{
		u1 = -a / b;
		u2 = -1;
	}
	else
	{
		u1 = (-b - copysign(sqrt(det), b)) / 2;
		u2 = a / u1;
		u1 /= c;
	}
	
	if (in_range(u1, 0, 1))
	{
		auto pa = lerp(q00, q10, u1);
		auto pb = lerp(e00, e11, u1);
		auto n  = _cross(omega, pb);
		auto n2 = _dot(n, n);
		n = _cross(n, pa);
		auto v1 = _dot(n, omega);
		if (0 <= v1 && v1 <= n2)
		{
			update_h    (box_min, box_max, v1 / n2);
			update_param(box_min, box_max, dot(n, pb) / n2);
		}
	}
	
	if (in_range(u2, 0, 1))
	{
		auto pa = lerp(q00, q10, u2);
		auto pb = lerp(e00, e11, u2);
		auto n  = _cross(omega, pb);
		auto n2 = _dot(n, n);
		n = _cross(n, pa);
		auto v2 = _dot(n, omega);
		if (0 <= v2 && v2 <= n2)
		{
			update_h    (box_min, box_max, v2 / n2);
			update_param(box_min, box_max, dot(n, pb) / n2);
		}
	}
}

// Eq.11
inline auto intersect_plane(const ray_lines &lines, const real h, const vec2 UV[3], const vec3&n, const real K, real& alpha, real &beta)
{
	const auto E0 = UV[1] - UV[0];
	const auto E1 = UV[2] - UV[0];

	const auto A0 = fma(lines.eN0, h, lines.eE0);
	const auto A1 = fma(lines.eN2, h, lines.eE2);
	const auto A2 = ab_plus_cd(n.x, E0.x, n.y, E0.y);

	const auto B0 = fma(lines.eN1, h, lines.eE1);
	const auto B1 = fma(lines.eN3, h, lines.eE3);
	const auto B2 = ab_plus_cd(n.x, E1.x, n.y, E1.y);

	const auto det0 = ab_minus_cd(A0, B1, B0, A1);
	const auto det1 = ab_minus_cd(A1, B2, B1, A2);
	const auto det2 = ab_minus_cd(A2, B0, B2, A0);

	// Using conditional numbers led to slightly poor results
	// Solve with the most reliable one
	if((0 != det0) || (0 != det1) || (0 != det2))
	{
		const auto C0 = fma(lines.eS0, -h, lines.eo0);
		const auto C1 = fma(lines.eS1, -h, lines.eo1);
		const auto C2 = -fma(n.x, UV[0].x, fma(n.y, UV[0].y, fma(n.z, h, K)));

		auto tmp = abs(det0); auto idx = 0;
		if  (tmp < abs(det1)) {    idx = 1; tmp = abs(det1); }
		if  (tmp < abs(det2)) {    idx = 2; tmp = abs(det2); }
		if(0 == idx)
		{
			alpha = ab_minus_cd(B1, C0, B0, C1) / det0;
			beta  = ab_minus_cd(A0, C1, A1, C0) / det0;
		}
		if(1 == idx)
		{
			alpha = ab_minus_cd(B2, C1, B1, C2) / det1;
			beta  = ab_minus_cd(A1, C2, A2, C1) / det1;
		}
		if(2 == idx)
		{
			alpha = ab_minus_cd(B0, C2, B2, C0) / det2;
			beta  = ab_minus_cd(A2, C0, A0, C2) / det2;
		}
		return true;
	}
	return false;
}

// From Eq.8
// nu = _fma(na, E0.x, _fma(nb, E1.x, d * UV[0].x));
// nv = _fma(na, E0.y, _fma(nb, E1.y, d * UV[0].y));
inline auto intersect_aabb(const vec3 &origin, const vec3 &omega,
						   const vec3 Pb[3], const vec3 VN[3], const vec2 UV[3],
						   const vec3 &box_min, const vec3 &box_max,
						   const vec3 &d,                  // denominator
						   const vec3 &nu, const vec3 &nv, // numerators in texture   space
						   const vec3 &na, const vec3 &nb, // numerators in canonical space
						   const ray_lines &lines)
{
	vec2 range(infinity, -infinity);

	real hs[2];
	real alpha, beta;
	
	const vec3 u_axis(1, 0, 0);
	const vec3 v_axis(0, 1, 0);
	
	// h
	{
		const auto h = box_min[2];
		const auto denom = eval_quad(d, h);
		if(0 != denom)
		{
			const auto uv = get_tuv(h, denom, nu, nv);
			if(in_range(uv.x, box_min[0], box_max[0], epsilon))
			{
				if(in_range(uv.y, box_min[1], box_max[1], epsilon))
				{
					alpha = eval_quad(na, h) / denom;
					beta  = eval_quad(nb, h) / denom;
					update_range(origin, omega, Pb, VN, h, alpha, beta, range);
				}
			}
		}
	}
	
	{
		const auto h = box_max[2];
		const auto denom = eval_quad(d, h);
		if(0 != denom)
		{
			const auto uv = get_tuv(h, denom, nu, nv);
			if(in_range(uv.x, box_min[0], box_max[0], epsilon))
			{
				if(in_range(uv.y, box_min[1], box_max[1], epsilon))
				{
					alpha = eval_quad(na, h) / denom;
					beta  = eval_quad(nb, h) / denom;
					update_range(origin, omega, Pb, VN, h, alpha, beta, range);
				}
			}
		}
	}
	
	// u
	{
		const auto u = box_min[0];
		const auto r = intersect_plane(nu, d, u, hs);
		for(auto k = 0; k < r; ++k)
		{
			const auto h = hs[k];
			if(in_range(h, box_min[2], box_max[2], epsilon))
			{
				if(intersect_plane(lines, h, UV, u_axis, -u, alpha, beta))
				{
					const auto uv = interpolate(UV, alpha, beta);
					if(in_range(uv.y, box_min[1], box_max[1], epsilon))
					{
						update_range(origin, omega, Pb, VN, h, alpha, beta, range);
					}
				}
			}
		}
	}
	{
		const auto u = box_max[0];
		const auto r = intersect_plane(nu, d, u, hs);
		for(auto k = 0; k < r; ++k)
		{
			const auto h = hs[k];
			if(in_range(h, box_min[2], box_max[2], epsilon))
			{
				if(intersect_plane(lines, h, UV, u_axis, -u, alpha, beta))
				{
					const auto uv = interpolate(UV, alpha, beta);
					if(in_range(uv.y, box_min[1], box_max[1], epsilon))
					{
						update_range(origin, omega, Pb, VN, h, alpha, beta, range);
					}
				}
			}
		}
	}
	
	// v
	{
		const auto v = box_min[1];
		const auto r = intersect_plane(nv, d, v, hs);
		for(auto k = 0; k < r; ++k)
		{
			const auto h = hs[k];
			if(in_range(h, box_min[2], box_max[2], epsilon))
			{
				if(intersect_plane(lines, h, UV, v_axis, -v, alpha, beta))
				{
					const auto uv = interpolate(UV, alpha, beta);
					if(in_range(uv.x, box_min[0], box_max[0], epsilon))
					{
						update_range(origin, omega, Pb, VN, h, alpha, beta, range);
					}
				}
			}
		}
	}
	{
		const auto v = box_max[1];
		const auto r = intersect_plane(nv, d, v, hs);
		for(auto k = 0; k < r; ++k)
		{
			const auto h = hs[k];
			if(in_range(h, box_min[2], box_max[2], epsilon))
			{
				if(intersect_plane(lines, h, UV, v_axis, -v, alpha, beta))
				{
					const auto uv = interpolate(UV, alpha, beta);
					if(in_range(uv.x, box_min[0], box_max[0], epsilon))
					{
						update_range(origin, omega, Pb, VN, h, alpha, beta, range);
					}
				}
			}
		}
	}
	
	return range;
}

inline auto intersect_prism(const vec3 &origin, const vec3 &omega, const vec3 Pb[3], const vec3 VN[3])
{
	// Offset
	const vec3 Po[3] = { Pb[0] + VN[0], Pb[1] + VN[1], Pb[2] + VN[2] };
	
	vec4 range_min(+infinity);
	vec4 range_max(-infinity);
	
	const auto hit0 = intersect_triangle(origin, omega, Pb, range_min, range_max);
	const auto hit1 = intersect_triangle(origin, omega, Po, range_min, range_max);
	
	range_min.z = hit0 ? 0 : range_min.z;
	range_max.z = hit1 ? 1 : range_max.z;
	
	if(!(hit0 && hit1))
	{
		intersect_cool_patch(origin, omega, Pb[0], Po[0], Pb[1], Po[1], range_min, range_max);
		intersect_cool_patch(origin, omega, Pb[1], Po[1], Pb[2], Po[2], range_min, range_max);
		intersect_cool_patch(origin, omega, Pb[2], Po[2], Pb[0], Po[0], range_min, range_max);
	}
	
	return make_tuple(range_min, range_max);
}

// Adjugate matrix Eq.13
inline auto adjugate(const mat3 &m)
{
	return mat3(ab_minus_cd(m[1][1], m[2][2], m[1][2], m[2][1]),
				ab_minus_cd(m[1][2], m[2][0], m[1][0], m[2][2]),
				ab_minus_cd(m[1][0], m[2][1], m[1][1], m[2][0]),
				ab_minus_cd(m[0][2], m[2][1], m[0][1], m[2][2]),
				ab_minus_cd(m[0][0], m[2][2], m[0][2], m[2][0]),
				ab_minus_cd(m[0][1], m[2][0], m[0][0], m[2][1]),
				ab_minus_cd(m[0][1], m[1][2], m[0][2], m[1][1]),
				ab_minus_cd(m[0][2], m[1][0], m[0][0], m[1][2]),
				ab_minus_cd(m[0][0], m[1][1], m[0][1], m[1][0]));
};

inline auto intersect_microtriangle(const vec3 &origin, const vec3 &omega,
									const ray_lines &lines,
									const vec3 Pb[3], const vec3 VN[3], const vec2 UV[3], // base triangle
									const vec3 &nu, const vec3 &nv, const vec3 &d, // ray in texture space
									const vec3 &p0, const vec3 &p1, const vec3 &p2, // microtriangle
									real min_h, real max_h,
									real &param, // must be set before call
									real &alpha, real &beta, vec3&ng)
{
	const auto E0 = UV[1] - UV[0];
	const auto E1 = UV[2] - UV[0];

	// <n, (u, v, h)> + K = 0
	// Normalization for robust computation
	const auto n =  _normalize(_cross(p1 - p0, p2 - p0));
	const auto K = -_dot(n, p0);
	
	auto flag = false;
	
	// Solve a cubic equation
	real roots[3];
	real coeffs[4] = {
		shell::fma(n.x, nu[0], shell::fma(n.y, nv[0], K * d[0])),
		shell::fma(n.z, d[0], shell::fma(n.x, nu[1], shell::fma(n.y, nv[1], K * d[1]))),
		shell::fma(n.z, d[1], shell::fma(n.x, nu[2], shell::fma(n.y, nv[2], K * d[2]))),
		n.z * d[2]
	};
	min_h = std::max(min_h, std::min({p0.z, p1.z, p2.z}));
	max_h = std::min(max_h, std::max({p0.z, p1.z, p2.z}));
	const auto R = (0 == d[2])
	? solve_quadratic(coeffs[2] * 2, coeffs[1], coeffs[0] * 2, roots[0], roots[1]) // This is faster
	: cy::CubicRoots<real, true>(roots, coeffs, min_h - delta, max_h + delta, real(1e-16)); // Use delta here
	
	// For each h
	for(auto r = 0; r < R; ++r)
	{
		const auto h = roots[r];
		
		// Use delta here
		if(!in_range(h, min_h, max_h, delta))
		{
			continue;
		}

		if(intersect_plane(lines, h, UV, n, K, alpha, beta))
		{
			// Inside base triangle
			if(0 <= alpha && 0 <= beta && 0 <= 1 - alpha - beta)
			{
				// Inside microtriangle
				const auto uv    = _fma(E1, beta, _fma(E0, alpha, UV[0]));
				const auto n_uvh = _cross(p1 - p0, p2 - p0);
				const auto A = _dot(n_uvh, n_uvh);
				const auto c = _cross(n_uvh, vec3(uv.x, uv.y, h) - p0);
				const auto u = _dot(c, p2 - p0) / A;
				const auto v = _dot(c, p0 - p1) / A;
				const auto w =  real(1) - (u + v);

				// epsilon is needed to render Fig.9(b)
				if (-epsilon <= u && -epsilon <= v && -epsilon <= w)
				{
					shell::vec3 S[3]; get_S(S, Pb, VN, std::clamp(h, min_h, max_h));
					const auto world = interpolate(S, alpha, beta);
					const auto tmp   = get_signed_distance(origin, omega, world);
					if(in_range(tmp, epsilon, param - epsilon))
					{
						const auto Ns = interpolate(VN, alpha, beta);
						const auto a  = ab_plus_cd(n.x, E0.x, n.y, E0.y);
						const auto b  = ab_plus_cd(n.x, E1.x, n.y, E1.y);
						const auto V1 = S[1] - S[0];
						const auto V2 = S[2] - S[0];
						const auto M = mat3(V1, V2, Ns);
						// Eq.13
						ng    = adjugate(M) * vec3(a, b, n.z);
						param = tmp;
						flag  = true;
					}
				}
			}
		}
	}
	return flag;
}
}
