'''
* Vectorised contact energy for the fabric simulations (numpy).
* Computes the same sum as the nested idx1/n/m/idx2 loop in contact_energy()
* of stockinette.py, garter.py, rib.py and seed.py, about 100x faster.
* The loop is still used when print_to_file=True, so ContactMapOut.dat and
* ContactDensityOut.dat are written exactly as before.
'''
import numpy as np


def floof_vec(x, contact_rigidity, contact_exponent, yarn_radius, core_radius):
	# array version of floof() in the simulation scripts (same formula)
	k = contact_rigidity
	p = contact_exponent
	dia_outer = 2*yarn_radius
	dia_core  = 2*core_radius
	dia_diff  = dia_outer - dia_core
	EPS = 1e-5

	dist = x + dia_outer
	z = (dist - dia_core)/dia_diff

	v = lambda y: k*( ((dia_diff)**2)/(p*(p - 1)) )*( y**(1-p) - 1 + (1-p)*(1-y) )

	stiff_core = v(EPS) + 100.*((dist - (dia_core + EPS*dia_diff))/(dia_core + EPS*dia_diff))**2
	return np.where(z >= EPS, v(np.maximum(z, EPS)), stiff_core)


def contact_energy_sum(points, speed, dt, s_mid, sTot, tr_c, tr_w, rad, s_excl,
                       floof_params, n_range, m_range, image_rotation=None):
	'''
	points        (N,3) midpoints of the mesh edges (curve_points_mid)
	speed, dt     (N,)  curve_points_mid_dt and dt_array
	s_mid         (N,)  curve_arclength_mid
	n_range, m_range    the ranges of the n and m loops in the original code
	image_rotation      None, or a function (m, n) -> 3x3 matrix applied to the
	                    neighbouring cell's points (garter, rib, seed use powers of f_z)
	'''
	P = np.asarray(points, dtype=float)
	w = np.asarray(speed, dtype=float)*np.asarray(dt, dtype=float)
	W = np.outer(w, w)
	ds = np.asarray(s_mid, dtype=float)[:, None] - np.asarray(s_mid, dtype=float)[None, :]

	ret = 0.0
	for n in n_range:
		for m in m_range:
			Q = P if image_rotation is None else P @ np.asarray(image_rotation(m, n), dtype=float).T
			Q = Q + m*tr_c + n*tr_w
			dist = np.sqrt(((P[:, None, :] - Q[None, :, :])**2).sum(axis=2)) - 2*rad
			mask = dist < 0
			if n == 0:
				mask &= np.abs(ds - m*sTot) > s_excl
			if mask.any():
				ret += np.sum(floof_vec(dist[mask], *floof_params)*W[mask])
	return ret
