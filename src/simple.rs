use numpy::{PyReadonlyArray2, PyArray1};
use pyo3::prelude::*;

/// Hybrid upwind scheme.
#[inline(always)]
fn hybrid(F: f64, D: f64) -> f64 {
    (D - 0.5 * F.abs()).max(0.0) + (-F).max(0.0)
}

/// Full SIMPLE steady-state solver in Rust.
/// Returns (ux_flat, uy_flat, p_flat) as 1D NumPy arrays (reshape in Python).
#[pyfunction]
pub fn solve_steady_simple<'py>(
    py: Python<'py>,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    nu: f64,
    alpha_u: f64,
    alpha_p: f64,
    max_iter: usize,
    tol: f64,
    inlet_velocity: f64,
    lid_velocity: f64,
    case: &str,
    brinkman: PyReadonlyArray2<'py, f64>,
    nu_field: Option<PyReadonlyArray2<'py, f64>>,
    u_body_x: Option<PyReadonlyArray2<'py, f64>>,
    u_body_y: Option<PyReadonlyArray2<'py, f64>>,
    buoyancy_src: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
    // Convert ndarray views to contiguous f64 vecs
    let bk_vec = brinkman.as_slice()?.to_vec();
    let nu_f_vec: Option<Vec<f64>> = nu_field.map(|a| a.as_slice().unwrap().to_vec());
    let ubx_vec: Option<Vec<f64>> = u_body_x.map(|a| a.as_slice().unwrap().to_vec());
    let uby_vec: Option<Vec<f64>> = u_body_y.map(|a| a.as_slice().unwrap().to_vec());
    let buoy_vec: Option<Vec<f64>> = buoyancy_src.map(|a| a.as_slice().unwrap().to_vec());

    let mut ux = vec![0.0f64; ny * (nx + 1)];
    let mut uy = vec![0.0f64; (ny + 1) * nx];
    let mut p = vec![0.0f64; ny * nx];

    apply_bcs(&mut ux, &mut uy, nx, ny, inlet_velocity, lid_velocity, case);

    let ny_int = ny - 2;
    let nu_int = ny_int * (nx - 1);
    let nv_int = (ny - 1) * nx;
    let np_int = ny * nx;
    let pin_cell = (ny / 2) * nx + (nx - 1);

    for _iter in 0..max_iter {
        let p_old = p.clone();

        let (ux_sol, a_ux) = solve_momentum_u(
            &ux, &uy, &p, &bk_vec, nu, dx, dy, alpha_u, nx, ny,
            inlet_velocity, lid_velocity, case,
            nu_f_vec.as_deref(), ubx_vec.as_deref(),
        );
        ux = ux_sol;

        let (uy_sol, a_uy) = solve_momentum_v(
            &ux, &uy, &p, &bk_vec, nu, dx, dy, alpha_u, nx, ny,
            nu_f_vec.as_deref(), uby_vec.as_deref(), buoy_vec.as_deref(),
        );
        uy = uy_sol;

        apply_bcs(&mut ux, &mut uy, nx, ny, inlet_velocity, lid_velocity, case);

        let div_star = compute_divergence(&ux, &uy, dx, dy, nx, ny);

        let lp = build_pressure_triplets(&a_ux, &a_uy, dx, dy, nx, ny, pin_cell);
        let mut rhs_p = vec![0.0f64; np_int];
        for i in 0..np_int {
            rhs_p[i] = -div_star[i];
        }
        rhs_p[pin_cell] = 0.0;

        let p_prime = dense_solve(&lp, &rhs_p, np_int);

        for j in 0..ny {
            for i in 1..nx {
                let dp = (p_prime[j * nx + i] - p_prime[j * nx + i - 1]) / dx;
                let a_p = a_ux[j * (nx - 1) + i - 1];
                ux[j * (nx + 1) + i] -= dp * dy / a_p.max(1e-10);
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                let dp = (p_prime[j * nx + i] - p_prime[(j - 1) * nx + i]) / dy;
                let a_p = a_uy[(j - 1) * nx + i];
                uy[j * nx + i] -= dp * dx / a_p.max(1e-10);
            }
        }

        for i in 0..np_int {
            p[i] = p_old[i] + alpha_p * p_prime[i];
        }

        apply_bcs(&mut ux, &mut uy, nx, ny, inlet_velocity, lid_velocity, case);
        let p_ref = p[ny / 2 * nx + nx - 1];
        for v in p.iter_mut() {
            *v -= p_ref;
        }

        let div_new = compute_divergence(&ux, &uy, dx, dy, nx, ny);
        let max_div = div_new.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        if max_div < tol {
            break;
        }
    }

    Ok((
        PyArray1::from_vec(py, ux).unbind(),
        PyArray1::from_vec(py, uy).unbind(),
        PyArray1::from_vec(py, p).unbind(),
    ))
}

fn apply_bcs(
    ux: &mut [f64], uy: &mut [f64],
    nx: usize, ny: usize,
    inlet_velocity: f64, lid_velocity: f64,
    case: &str,
) {
    if case == "channel" {
        for j in 0..ny { ux[j * (nx + 1)] = inlet_velocity; }
        for j in 0..ny { ux[j * (nx + 1) + nx] = ux[j * (nx + 1) + nx - 1]; }
        for i in 0..=nx {
            ux[i] = 0.0;
            ux[(ny - 1) * (nx + 1) + i] = 0.0;
        }
    } else {
        for i in 0..=nx {
            ux[i] = 0.0;
            ux[(ny - 1) * (nx + 1) + i] = lid_velocity;
        }
        for j in 0..=ny {
            uy[j * nx] = 0.0;
            if nx > 0 { uy[j * nx + nx - 1] = 0.0; }
        }
    }
    for i in 0..nx {
        uy[i] = 0.0;
        uy[ny * nx + i] = 0.0;
    }
}

fn compute_divergence(ux: &[f64], uy: &[f64], dx: f64, dy: f64, nx: usize, ny: usize) -> Vec<f64> {
    let mut div = vec![0.0f64; ny * nx];
    for j in 0..ny {
        for i in 0..nx {
            div[j * nx + i] = (ux[j * (nx + 1) + i + 1] - ux[j * (nx + 1) + i]) / dx
                + (uy[(j + 1) * nx + i] - uy[j * nx + i]) / dy;
        }
    }
    div
}

fn solve_momentum_u(
    ux: &[f64], uy: &[f64], p: &[f64], bk: &[f64],
    nu: f64, dx: f64, dy: f64, alpha_u: f64,
    nx: usize, ny: usize,
    inlet_velocity: f64, lid_velocity: f64, case: &str,
    nu_f: Option<&[f64]>,
    ubx: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let ny_int = ny - 2;
    let nu_int = ny_int * (nx - 1);
    let u_north_wall = if case == "cavity" { lid_velocity } else { 0.0 };
    let u_inlet_val = if case == "channel" { inlet_velocity } else { 0.0 };

    let mut rows: Vec<usize> = Vec::with_capacity(nu_int * 5);
    let mut cols: Vec<usize> = Vec::with_capacity(nu_int * 5);
    let mut vals: Vec<f64> = Vec::with_capacity(nu_int * 5);
    let mut b = vec![0.0f64; nu_int];
    let mut a_p = vec![1e30f64; ny * (nx - 1)];

    for j_int in 0..ny_int {
        let j = j_int + 1;
        for ii in 0..(nx - 1) {
            let k = j_int * (nx - 1) + ii;
            let u_c = ux[j * (nx + 1) + ii + 1];

            let u_e_face = 0.5 * ux[j * (nx + 1) + (ii + 2).min(nx)] + 0.5 * u_c;
            let u_w_face = 0.5 * ux[j * (nx + 1) + ii] + 0.5 * u_c;
            let v_c = 0.25 * (
                uy[j * nx + ii] + uy[j * nx + ii + 1] +
                uy[(j + 1) * nx + ii] + uy[(j + 1) * nx + ii + 1]
            );

            let nu_cell = nu_f.map_or(nu, |nf| nf[j * nx + ii + 1]);
            let d_e = nu_cell / dx;
            let d_w = nu_cell / dx;
            let d_n = nu_cell / dy;
            let d_s = nu_cell / dy;

            let (a_e_val, a_e_mat, src_e) = if ii + 1 >= nx - 1 {
                (0.0, 0.0, 0.0)
            } else {
                let v = hybrid(u_e_face, d_e);
                (v, v, 0.0)
            };
            let (a_w_val, src_w, a_w_mat) = if ii == 0 {
                let v = hybrid(-u_w_face, d_w);
                (v, v * u_inlet_val, 0.0)
            } else {
                let v = hybrid(-u_w_face, d_w);
                (v, 0.0, v)
            };
            let (a_n_val, src_n, a_n_mat) = if j + 1 == ny - 1 {
                let v = hybrid(v_c, d_n);
                (v, v * u_north_wall, 0.0)
            } else if j_int + 1 < ny_int {
                let v = hybrid(v_c, d_n);
                (v, 0.0, v)
            } else {
                (0.0, 0.0, 0.0)
            };
            let (a_s_val, src_s, a_s_mat) = if j - 1 == 0 {
                let v = hybrid(-v_c, d_s);
                (v, 0.0, 0.0) // south wall u=0
            } else if j_int > 0 {
                let v = hybrid(-v_c, d_s);
                (v, 0.0, v)
            } else {
                (0.0, 0.0, 0.0)
            };

            let bk_face = 0.5 * (bk[j * nx + ii] + bk[j * nx + ii + 1]);
            let a_p0 = a_e_val + a_w_val + a_n_val + a_s_val + bk_face;
            let a_p_val = a_p0 / alpha_u;

            let mut src = -(p[j * nx + ii + 1] - p[j * nx + ii]) * dy / dx;
            src += (1.0 - alpha_u) / alpha_u * a_p0 * u_c;
            src += src_n + src_s + src_w + src_e;
            if let Some(ub) = ubx {
                let ub_face = 0.5 * (ub[j * nx + ii] + ub[j * nx + ii + 1]);
                src += bk_face * ub_face;
            }

            b[k] = src;
            a_p[j * (nx - 1) + ii] = a_p_val;

            rows.push(k); cols.push(k); vals.push(a_p_val);
            if a_e_mat > 0.0 { rows.push(k); cols.push(k + 1); vals.push(-a_e_mat); }
            if a_w_mat > 0.0 { rows.push(k); cols.push(k - 1); vals.push(-a_w_mat); }
            if a_n_mat > 0.0 { rows.push(k); cols.push(k + (nx - 1)); vals.push(-a_n_mat); }
            if a_s_mat > 0.0 { rows.push(k); cols.push(k - (nx - 1)); vals.push(-a_s_mat); }
        }
    }

    let sol = dense_solve_coo(&rows, &cols, &vals, &b, nu_int);
    let mut ux_new = ux.to_vec();
    for j_int in 0..ny_int {
        for ii in 0..(nx - 1) {
            ux_new[(j_int + 1) * (nx + 1) + ii + 1] = sol[j_int * (nx - 1) + ii];
        }
    }
    (ux_new, a_p)
}

fn solve_momentum_v(
    ux: &[f64], uy: &[f64], p: &[f64], bk: &[f64],
    nu: f64, dx: f64, dy: f64, alpha_u: f64,
    nx: usize, ny: usize,
    nu_f: Option<&[f64]>,
    uby: Option<&[f64]>,
    buoy: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let nv_int = (ny - 1) * nx;

    let mut rows: Vec<usize> = Vec::with_capacity(nv_int * 5);
    let mut cols: Vec<usize> = Vec::with_capacity(nv_int * 5);
    let mut vals: Vec<f64> = Vec::with_capacity(nv_int * 5);
    let mut b = vec![0.0f64; nv_int];
    let mut a_p = vec![0.0f64; nv_int];

    for jj in 0..(ny - 1) {
        let j = jj + 1;
        for i in 0..nx {
            let k = jj * nx + i;
            let v_c = uy[j * nx + i];
            let v_n_face = 0.5 * if j + 1 <= ny - 1 { uy[(j + 1) * nx + i] } else { 0.0 } + 0.5 * v_c;
            let v_s_face = 0.5 * uy[(j - 1) * nx + i] + 0.5 * v_c;

            let u_sw = ux[(j - 1) * (nx + 1) + i];
            let u_se = ux[(j - 1) * (nx + 1) + i + 1];
            let u_nw = ux[j * (nx + 1) + i];
            let u_ne = ux[j * (nx + 1) + i + 1];
            let u_c = 0.25 * (u_sw + u_se + u_nw + u_ne);

            let f_n = if j + 1 <= ny - 1 { v_n_face } else { 0.0 };
            let f_s = v_s_face;
            let f_e = if i < nx - 1 { u_c } else { 0.0 };
            let f_w = if i > 0 { u_c } else { 0.0 };

            let nu_cell = nu_f.map_or(nu, |nf| nf[j * nx + i]);
            let d_n = if j + 1 < ny { nu_cell / dy } else { 0.0 };
            let d_s = nu_cell / dy;
            let d_e = if i < nx - 1 { nu_cell / dx } else { 0.0 };
            let d_w = if i > 0 { nu_cell / dx } else { 0.0 };

            let a_n = if jj + 1 < ny - 1 { hybrid(f_n, d_n) } else { 0.0 };
            let a_s = if jj > 0 { hybrid(-f_s, d_s) } else { 0.0 };
            let a_e = if i + 1 < nx { hybrid(f_e, d_e) } else { 0.0 };
            let a_w = if i > 0 { hybrid(-f_w, d_w) } else { 0.0 };

            let bk_face = 0.5 * (bk[(j - 1) * nx + i] + bk[j.min(ny - 1) * nx + i]);
            let a_p0 = a_n + a_s + a_e + a_w + bk_face;
            let a_p_val = a_p0 / alpha_u;

            let mut src = -(p[j * nx + i] - p[(j - 1) * nx + i]) * dx / dy;
            src += (1.0 - alpha_u) / alpha_u * a_p0 * v_c;
            if let Some(bu) = buoy { src += bu[j * nx + i] * dx * dy; }
            if let Some(ub) = uby {
                let ub_face = 0.5 * (ub[(j - 1) * nx + i] + ub[j.min(ny - 1) * nx + i]);
                src += bk_face * ub_face;
            }

            b[k] = src;
            a_p[k] = a_p_val;

            rows.push(k); cols.push(k); vals.push(a_p_val);
            if jj + 1 < ny - 1 { rows.push(k); cols.push(k + nx); vals.push(-a_n); }
            if jj > 0 { rows.push(k); cols.push(k - nx); vals.push(-a_s); }
            if i + 1 < nx { rows.push(k); cols.push(k + 1); vals.push(-a_e); }
            if i > 0 { rows.push(k); cols.push(k - 1); vals.push(-a_w); }
        }
    }

    let sol = dense_solve_coo(&rows, &cols, &vals, &b, nv_int);
    let mut uy_new = uy.to_vec();
    for jj in 0..(ny - 1) {
        for i in 0..nx {
            uy_new[(jj + 1) * nx + i] = sol[jj * nx + i];
        }
    }
    (uy_new, a_p)
}

struct Trips { rows: Vec<usize>, cols: Vec<usize>, vals: Vec<f64>, n: usize }

fn build_pressure_triplets(
    a_ux: &[f64], a_uy: &[f64], dx: f64, dy: f64,
    nx: usize, ny: usize, pin_cell: usize,
) -> Trips {
    let n = nx * ny;
    let mut rows = Vec::with_capacity(n * 5);
    let mut cols = Vec::with_capacity(n * 5);
    let mut vals = Vec::with_capacity(n * 5);

    for j in 0..ny {
        for i in 0..nx {
            let k = j * nx + i;
            if k == pin_cell {
                rows.push(k); cols.push(k); vals.push(1.0);
                continue;
            }
            let mut d = 0.0;
            if i + 1 < nx {
                let c_e = dy / (a_ux[j * (nx - 1) + i] * dx * dx + 1e-30);
                rows.push(k); cols.push(k + 1); vals.push(-c_e);
                d += c_e;
            }
            if i > 0 {
                let c_w = dy / (a_ux[j * (nx - 1) + i - 1] * dx * dx + 1e-30);
                rows.push(k); cols.push(k - 1); vals.push(-c_w);
                d += c_w;
            }
            if j + 1 < ny {
                let c_n = dx / (a_uy[j * nx + i] * dy * dy + 1e-30);
                rows.push(k); cols.push(k + nx); vals.push(-c_n);
                d += c_n;
            }
            if j > 0 {
                let c_s = dx / (a_uy[(j - 1) * nx + i] * dy * dy + 1e-30);
                rows.push(k); cols.push(k - nx); vals.push(-c_s);
                d += c_s;
            }
            rows.push(k); cols.push(k); vals.push(d);
        }
    }
    Trips { rows, cols, vals, n }
}

/// Dense LU solve from COO-format sparse matrix (for moderate-size 2D grids).
fn dense_solve_coo(rows: &[usize], cols: &[usize], vals: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for idx in 0..rows.len() {
        a[rows[idx] * n + cols[idx]] += vals[idx];
    }
    use faer::linalg::solvers::SpSolver;
    let mat = faer::Mat::from_fn(n, n, |r, c| a[r * n + c]);
    let rhs = faer::Col::from_fn(n, |i| b[i]);
    let sol = mat.partial_piv_lu().solve(&rhs);
    let mut result = vec![0.0f64; n];
    for i in 0..n { result[i] = sol.read(i); }
    result
}

fn dense_solve(t: &Trips, b: &[f64], n: usize) -> Vec<f64> {
    dense_solve_coo(&t.rows, &t.cols, &t.vals, b, n)
}
