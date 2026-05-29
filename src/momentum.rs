use numpy::{PyReadonlyArray2, PyArray1};
use pyo3::prelude::*;

#[inline(always)]
fn hybrid(F: f64, D: f64) -> f64 {
    (D - 0.5 * F.abs()).max(0.0) + (-F).max(0.0)
}

/// Build u-momentum sparse system. Returns (indptr, indices, data, rhs, aP_field) as 1D arrays.
#[pyfunction]
pub fn build_momentum_u<'py>(
    py: Python<'py>,
    ux: PyReadonlyArray2<'py, f64>,
    uy: PyReadonlyArray2<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
    brinkman: PyReadonlyArray2<'py, f64>,
    nu: f64,
    dx: f64,
    dy: f64,
    alpha_u: f64,
    nx: usize,
    ny: usize,
    inlet_velocity: f64,
    lid_velocity: f64,
    case: &str,
    nu_field: Option<PyReadonlyArray2<'py, f64>>,
    u_body: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
    let ux = ux.as_array();
    let uy = uy.as_array();
    let p = p.as_array();
    let bk = brinkman.as_array();
    let nu_f = nu_field.as_ref().map(|a| a.as_array());
    let ubody = u_body.as_ref().map(|a| a.as_array());

    let ny_int = ny - 2;
    let nu_int = ny_int * (nx - 1);

    let u_north_wall = if case == "cavity" { lid_velocity } else { 0.0 };
    let u_inlet_val = if case == "channel" { inlet_velocity } else { 0.0 };

    let mut rows: Vec<i64> = Vec::with_capacity(nu_int * 5);
    let mut cols: Vec<i64> = Vec::with_capacity(nu_int * 5);
    let mut vals: Vec<f64> = Vec::with_capacity(nu_int * 5);
    let mut b = vec![0.0f64; nu_int];
    let mut a_p = vec![1e30f64; ny * (nx - 1)];

    for j_int in 0..ny_int {
        let j = j_int + 1;
        for ii in 0..(nx - 1) {
            let k = (j_int * (nx - 1) + ii) as i64;
            let u_c = ux[[j, ii + 1]];

            let u_e_face = 0.5 * if ii + 2 <= nx { ux[[j, ii + 2]] } else { ux[[j, nx]] } + 0.5 * u_c;
            let u_w_face = 0.5 * ux[[j, ii]] + 0.5 * u_c;
            let v_c = 0.25 * (uy[[j, ii]] + uy[[j, ii + 1]] + uy[[j + 1, ii]] + uy[[j + 1, ii + 1]]);

            let nu_cell = if let Some(ref nf) = nu_f { nf[[j, ii + 1]] } else { nu };
            let d_e = nu_cell / dx;
            let d_w = nu_cell / dx;
            let d_n = nu_cell / dy;
            let d_s = nu_cell / dy;

            let (a_e_val, a_e_mat, src_e) = if ii + 1 >= nx - 1 { (0.0, 0.0, 0.0) } else { let v = hybrid(u_e_face, d_e); (v, v, 0.0) };
            let (a_w_val, src_w, a_w_mat) = if ii == 0 { let v = hybrid(-u_w_face, d_w); (v, v * u_inlet_val, 0.0) } else { let v = hybrid(-u_w_face, d_w); (v, 0.0, v) };
            let (a_n_val, src_n, a_n_mat) = if j + 1 == ny - 1 { let v = hybrid(v_c, d_n); (v, v * u_north_wall, 0.0) } else if j_int + 1 < ny_int { let v = hybrid(v_c, d_n); (v, 0.0, v) } else { (0.0, 0.0, 0.0) };
            let (a_s_val, _src_s, a_s_mat) = if j - 1 == 0 { let v = hybrid(-v_c, d_s); (v, 0.0, 0.0) } else if j_int > 0 { let v = hybrid(-v_c, d_s); (v, 0.0, v) } else { (0.0, 0.0, 0.0) };

            let bk_face = 0.5 * (bk[[j, ii]] + bk[[j, ii + 1]]);
            let a_p0 = a_e_val + a_w_val + a_n_val + a_s_val + bk_face;
            let a_p_val = a_p0 / alpha_u;

            let mut src = -(p[[j, ii + 1]] - p[[j, ii]]) * dy / dx;
            src += (1.0 - alpha_u) / alpha_u * a_p0 * u_c;
            src += src_n + _src_s + src_w + src_e;
            if let Some(ref ub) = ubody {
                let ub_face = 0.5 * (ub[[j, ii]] + ub[[j, ii + 1]]);
                src += bk_face * ub_face;
            }

            b[k as usize] = src;
            a_p[j * (nx - 1) + ii] = a_p_val;

            rows.push(k); cols.push(k); vals.push(a_p_val);
            if a_e_mat > 0.0 { rows.push(k); cols.push(k + 1); vals.push(-a_e_mat); }
            if a_w_mat > 0.0 { rows.push(k); cols.push(k - 1); vals.push(-a_w_mat); }
            if a_n_mat > 0.0 { rows.push(k); cols.push(k + (nx - 1) as i64); vals.push(-a_n_mat); }
            if a_s_mat > 0.0 { rows.push(k); cols.push(k - (nx - 1) as i64); vals.push(-a_s_mat); }
        }
    }

    let (indptr, indices_csr, data_csr) = coo_to_csr(&rows, &cols, &vals, nu_int);
    Ok((
        PyArray1::from_vec(py, indptr).unbind(),
        PyArray1::from_vec(py, indices_csr).unbind(),
        PyArray1::from_vec(py, data_csr).unbind(),
        PyArray1::from_vec(py, b).unbind(),
        PyArray1::from_vec(py, a_p).unbind(),
    ))
}

/// Build v-momentum sparse system. Returns (indptr, indices, data, rhs, aP_field) as 1D arrays.
#[pyfunction]
pub fn build_momentum_v<'py>(
    py: Python<'py>,
    ux: PyReadonlyArray2<'py, f64>,
    uy: PyReadonlyArray2<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
    brinkman: PyReadonlyArray2<'py, f64>,
    nu: f64,
    dx: f64,
    dy: f64,
    alpha_u: f64,
    nx: usize,
    ny: usize,
    buoyancy_src: Option<PyReadonlyArray2<'py, f64>>,
    nu_field: Option<PyReadonlyArray2<'py, f64>>,
    u_body: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
    let ux = ux.as_array();
    let uy = uy.as_array();
    let p = p.as_array();
    let bk = brinkman.as_array();
    let nu_f = nu_field.as_ref().map(|a| a.as_array());
    let ubody = u_body.as_ref().map(|a| a.as_array());
    let buoy = buoyancy_src.as_ref().map(|a| a.as_array());

    let nv_int = (ny - 1) * nx;

    let mut rows: Vec<i64> = Vec::with_capacity(nv_int * 5);
    let mut cols: Vec<i64> = Vec::with_capacity(nv_int * 5);
    let mut vals: Vec<f64> = Vec::with_capacity(nv_int * 5);
    let mut b = vec![0.0f64; nv_int];
    let mut a_p = vec![0.0f64; nv_int];

    for jj in 0..(ny - 1) {
        let j = jj + 1;
        for i in 0..nx {
            let k = (jj * nx + i) as i64;
            let v_c = uy[[j, i]];
            let v_n_face = 0.5 * if j + 1 <= ny - 1 { uy[[j + 1, i]] } else { 0.0 } + 0.5 * v_c;
            let v_s_face = 0.5 * uy[[j - 1, i]] + 0.5 * v_c;

            let u_sw = ux[[j - 1, i]]; let u_se = ux[[j - 1, i + 1]];
            let u_nw = ux[[j, i]];     let u_ne = ux[[j, i + 1]];
            let u_c = 0.25 * (u_sw + u_se + u_nw + u_ne);

            let f_n = if j + 1 <= ny - 1 { v_n_face } else { 0.0 };
            let f_s = v_s_face;
            let f_e = if i < nx - 1 { u_c } else { 0.0 };
            let f_w = if i > 0 { u_c } else { 0.0 };

            let nu_cell = if let Some(ref nf) = nu_f { nf[[j, i]] } else { nu };
            let d_n = if j + 1 < ny { nu_cell / dy } else { 0.0 };
            let d_s = nu_cell / dy;
            let d_e = if i < nx - 1 { nu_cell / dx } else { 0.0 };
            let d_w = if i > 0 { nu_cell / dx } else { 0.0 };

            let a_n = if jj + 1 < ny - 1 { hybrid(f_n, d_n) } else { 0.0 };
            let a_s = if jj > 0 { hybrid(-f_s, d_s) } else { 0.0 };
            let a_e = if i + 1 < nx { hybrid(f_e, d_e) } else { 0.0 };
            let a_w = if i > 0 { hybrid(-f_w, d_w) } else { 0.0 };

            let bk_face = 0.5 * (bk[[j - 1, i]] + bk[[j.min(ny - 1), i]]);
            let a_p0 = a_n + a_s + a_e + a_w + bk_face;
            let a_p_val = a_p0 / alpha_u;

            let mut src = -(p[[j, i]] - p[[j - 1, i]]) * dx / dy;
            src += (1.0 - alpha_u) / alpha_u * a_p0 * v_c;
            if let Some(ref bu) = buoy { src += bu[[j, i]] * dx * dy; }
            if let Some(ref ub) = ubody {
                let ub_face = 0.5 * (ub[[j - 1, i]] + ub[[j.min(ny - 1), i]]);
                src += bk_face * ub_face;
            }

            b[k as usize] = src;
            a_p[k as usize] = a_p_val;

            rows.push(k); cols.push(k); vals.push(a_p_val);
            if jj + 1 < ny - 1 { rows.push(k); cols.push(k + nx as i64); vals.push(-a_n); }
            if jj > 0 { rows.push(k); cols.push(k - nx as i64); vals.push(-a_s); }
            if i + 1 < nx { rows.push(k); cols.push(k + 1); vals.push(-a_e); }
            if i > 0 { rows.push(k); cols.push(k - 1); vals.push(-a_w); }
        }
    }

    let (indptr, indices_csr, data_csr) = coo_to_csr(&rows, &cols, &vals, nv_int);
    Ok((
        PyArray1::from_vec(py, indptr).unbind(),
        PyArray1::from_vec(py, indices_csr).unbind(),
        PyArray1::from_vec(py, data_csr).unbind(),
        PyArray1::from_vec(py, b).unbind(),
        PyArray1::from_vec(py, a_p).unbind(),
    ))
}

fn coo_to_csr(rows: &[i64], cols: &[i64], vals: &[f64], n: usize) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let nnz = rows.len();
    let mut row_counts = vec![0usize; n + 1];
    for &r in rows { row_counts[r as usize + 1] += 1; }
    for i in 0..n { row_counts[i + 1] += row_counts[i]; }

    let mut order: Vec<usize> = (0..nnz).collect();
    order.sort_by_key(|&i| (rows[i], cols[i]));

    let mut indptr = vec![0i64; n + 1];
    let mut indices_out = vec![0i64; nnz];
    let mut data_out = vec![0.0f64; nnz];

    let mut pos = 0;
    for row in 0..n {
        indptr[row] = pos as i64;
        while pos < order.len() && rows[order[pos]] == row as i64 {
            let idx = order[pos];
            indices_out[pos] = cols[idx];
            data_out[pos] = vals[idx];
            pos += 1;
        }
    }
    indptr[n] = pos as i64;
    indices_out.truncate(pos);
    data_out.truncate(pos);

    (indptr, indices_out, data_out)
}
