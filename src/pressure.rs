use numpy::{PyReadonlyArray2, PyArray1};
use pyo3::prelude::*;
use crate::utils::coo_to_csr;

/// Build pressure correction system. Returns (indptr, indices, data, pin_idx).
#[pyfunction]
pub fn build_pressure_system<'py>(
    py: Python<'py>,
    a_ux: PyReadonlyArray2<'py, f64>,
    a_uy: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dy: f64,
    nx: usize,
    ny: usize,
) -> PyResult<(
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<f64>>,
    usize,
)> {
    let a_ux_np = a_ux.as_array();
    let a_uy_np = a_uy.as_array();

    let n = nx * ny;
    let pin_idx = (ny / 2) * nx + (nx - 1);

    let mut rows: Vec<i64> = Vec::with_capacity(n * 5);
    let mut cols: Vec<i64> = Vec::with_capacity(n * 5);
    let mut vals: Vec<f64> = Vec::with_capacity(n * 5);

    for j in 0..ny {
        for i in 0..nx {
            let k = (j * nx + i) as i64;
            if (j * nx + i) == pin_idx {
                rows.push(k); cols.push(k); vals.push(1.0);
                continue;
            }
            let mut d = 0.0;
            if i + 1 < nx {
                let c_e = dy / (a_ux_np[[j, i]] * dx * dx + 1e-30);
                rows.push(k); cols.push(k + 1); vals.push(-c_e);
                d += c_e;
            }
            if i > 0 {
                let c_w = dy / (a_ux_np[[j, i - 1]] * dx * dx + 1e-30);
                rows.push(k); cols.push(k - 1); vals.push(-c_w);
                d += c_w;
            }
            if j + 1 < ny {
                let c_n = dx / (a_uy_np[[j, i]] * dy * dy + 1e-30);
                rows.push(k); cols.push(k + nx as i64); vals.push(-c_n);
                d += c_n;
            }
            if j > 0 {
                let c_s = dx / (a_uy_np[[j - 1, i]] * dy * dy + 1e-30);
                rows.push(k); cols.push(k - nx as i64); vals.push(-c_s);
                d += c_s;
            }
            rows.push(k); cols.push(k); vals.push(d);
        }
    }

    let (indptr, indices_csr, data_csr) = coo_to_csr(&rows, &cols, &vals, n);
    Ok((
        PyArray1::from_vec(py, indptr).unbind(),
        PyArray1::from_vec(py, indices_csr).unbind(),
        PyArray1::from_vec(py, data_csr).unbind(),
        pin_idx,
    ))
}
