use super::DynMatrix;
use crate::matrix::MatrixElem;

impl<T> std::fmt::Display for DynMatrix<T>
where
    T: std::fmt::Display + MatrixElem,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows() {
            write!(f, "|")?;
            for j in 0..self.cols() {
                write!(f, "{} ", self.data[i * self.cols() + j])?;
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
    fn test_dynmatrix_display() {
        let matrix = DynMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let expected = "|1 2 |\n|3 4 |\n";
        assert_eq!(format!("{}", matrix), expected);
    }
}
