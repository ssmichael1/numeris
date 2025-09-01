use super::Matrix;

impl<const ROWS: usize, const COLS: usize, T> std::fmt::Display for Matrix<ROWS, COLS, T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for row in 0..ROWS {
            write!(f, "|")?;
            for col in 0..COLS {
                write!(f, "{} ", self[(row, col)])?;
            }
            write!(f, "|")?;
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_matrix_display() {
        let matrix = Matrix::from_row_major([[1.0, 2.0], [3.0, 4.0]]);
        let expected = "|1 2 |\n|3 4 |\n";
        assert_eq!(format!("{}", matrix), expected);
    }
}
