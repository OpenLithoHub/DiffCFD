#[inline(always)]
pub fn hybrid(F: f64, D: f64) -> f64 {
    (D - 0.5 * F.abs()).max(0.0) + (-F).max(0.0)
}

pub fn coo_to_csr(rows: &[i64], cols: &[i64], vals: &[f64], n: usize) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
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
