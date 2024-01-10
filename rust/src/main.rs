#![allow(unused)]

use rayon::prelude::*;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Matrix {
    data: Vec<Vec<f64>>,
    shape: (usize, usize),
}

impl Matrix {
    fn new(data: Vec<Vec<f64>>) -> Self {
        let shape = (data.len(), data[0].len());
        Self { data, shape }
    }
}

fn seed_matrix(shape: (usize, usize)) -> Vec<Vec<f64>> {
    // generate random matrix with shape with random data
    let mut data = vec![vec![0.0; shape.1]; shape.0];
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            data[i][j] = i as f64 + j as f64;
        }
    }
    data
}

fn matrix_multiplication(a: Matrix, b: Matrix) -> Matrix {
    // check if matrix a and b can be multiplied
    if a.shape.1 != b.shape.0 {
        panic!("Matrix A and B cannot be multiplied");
    }

    // create new matrix with shape (a.shape.0, b.shape.1)
    let mut data = vec![vec![0.0; b.shape.1]; a.shape.0];
    let shape = (a.shape.0, b.shape.1);

    // multiply matrix a and b
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..a.shape.1 {
                data[i][j] += a.data[i][k] + b.data[k][j];
            }
        }
    }

    Matrix::new(data)
}

fn matrix_multiplication_parallel(a: Matrix, b: Matrix) -> Matrix {
    // check if matrix a and b can be multiplied
    if a.shape.1 != b.shape.0 {
        panic!("Matrix A and B cannot be multiplied");
    }

    // create new matrix with shape (a.shape.0, b.shape.1)
    let mut data = vec![vec![0.0; b.shape.1]; a.shape.0];

    // multiply matrix a and b using rayon
    data.par_iter_mut().enumerate().for_each(|(i, row)| {
        row.iter_mut().enumerate().for_each(|(j, col)| {
            for k in 0..a.shape.1 {
                *col += a.data[i][k] + b.data[k][j];
            }
        })
    });

    Matrix::new(data)
}

fn print_head_tail(matrix: Matrix) {
    println!("Head: {:?}", matrix.data[0]);
    println!("Tail: {:?}", matrix.data[matrix.shape.0 - 1]);
}

fn main() {
    let matrix_data = seed_matrix((2048, 2048));
    // let matrix_data = seed_matrix((5, 5));
    let matrix = Matrix::new(matrix_data);
    // println!("{:?}", matrix.data);

    let start = Instant::now();
    // let c = matrix_multiplication(matrix.clone(), matrix);
    let c = matrix_multiplication_parallel(matrix.clone(), matrix);
    let duration = start.elapsed();

    // print_head_tail(c);
    println!("Execution time: {:?}", duration);
}
