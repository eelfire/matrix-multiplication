#![allow(unused)]
#![feature(portable_simd)]

use rayon::prelude::*;
use std::simd::num::SimdUint;
use std::simd::{u64x16, u64x2, u64x32, u64x4, u64x64, u64x8};
use std::time::Instant;

type Element = u64;
// 16 can be replaced with 2, 4, 8, 16, 32, 64 depending on the best results
type T = u64x16;
const N: usize = 16;

#[derive(Debug, Clone)]
struct Matrix {
    data: Vec<Vec<Element>>,
    shape: (usize, usize),
}

impl Matrix {
    fn new(data: Vec<Vec<Element>>) -> Self {
        let shape = (data.len(), data[0].len());
        Self { data, shape }
    }
}

fn seed_matrix(shape: (usize, usize)) -> Vec<Vec<Element>> {
    let mut data = vec![vec![1; shape.1]; shape.0];
    // for i in 0..shape.0 {
    //     for j in 0..shape.1 {
    //         // data[i][j] = ((i + j) as Element).clamp(0, 789);
    //         data[i][j] = 5;
    //     }
    // }
    data
}

// A = BC + BD
// C and D are assumed transposed
// Therefore, sum(row(B) * (row(C) + row(D)))
// shape(B) = (m, n), shape(C) = (p, n), shape(D) = (p, n)
// shape(A) = (m, p)

// simple
fn matrix_multiplication(b: Matrix, c: Matrix, d: Matrix) -> Matrix {
    if b.shape.1 != d.shape.1 || b.shape.1 != c.shape.1 {
        panic!("Matrix B and C and/or D cannot be multiplied");
    }

    let mut data = vec![vec![0; d.shape.0]; b.shape.0];

    data.iter_mut().enumerate().for_each(|(i, row)| {
        row.iter_mut().enumerate().for_each(|(j, col)| {
            for k in 0..b.shape.1 {
                *col += b.data[i][k] * (c.data[j][k] + d.data[j][k]);
            }
        })
    });

    Matrix::new(data)
}

// simple + simd
fn matrix_multiplication_simd(b: Matrix, c: Matrix, d: Matrix) -> Matrix {
    if b.shape.1 != d.shape.1 || b.shape.1 != c.shape.1 {
        panic!("Matrix A and B cannot be multiplied");
    }

    let mut data = vec![vec![0; d.shape.0]; b.shape.0];

    data.iter_mut().enumerate().for_each(|(i, row)| {
        row.iter_mut().enumerate().for_each(|(j, col)| {
            *col = simd_n(&b, i, &c, &d, j, N);
        })
    });

    Matrix::new(data)
}

// parallel iterations (using rayon)
fn matrix_multiplication_parallel(b: Matrix, c: Matrix, d: Matrix) -> Matrix {
    if b.shape.1 != d.shape.1 || b.shape.1 != c.shape.1 {
        panic!("Matrix A and B cannot be multiplied");
    }

    let mut data = vec![vec![0; d.shape.0]; b.shape.0];

    data.par_iter_mut().enumerate().for_each(|(i, row)| {
        row.iter_mut().enumerate().for_each(|(j, col)| {
            for k in 0..b.shape.1 {
                *col += b.data[i][k] * (c.data[k][j] + d.data[k][j]);
            }
        })
    });

    Matrix::new(data)
}

// parallel iterations + simd
fn matrix_multiplication_parallel_simd(b: Matrix, c: Matrix, d: Matrix) -> Matrix {
    if b.shape.1 != d.shape.1 || b.shape.1 != c.shape.1 {
        panic!("Matrix A and B cannot be multiplied");
    }

    let mut data = vec![vec![0; d.shape.0]; b.shape.0];

    data.par_iter_mut().enumerate().for_each(|(i, row)| {
        row.iter_mut().enumerate().for_each(|(j, col)| {
            *col = simd_n(&b, i, &c, &d, j, N);
        })
    });

    Matrix::new(data)
}

fn simd_n(b: &Matrix, i: usize, c: &Matrix, d: &Matrix, j: usize, n: usize) -> u64 {
    let mut sum = T::splat(0);
    for k in (0..b.shape.1).step_by(n) {
        let b_simd = T::from_slice(&b.data[i][k..]);
        let c_simd = T::from_slice(&c.data[j][k..]); // taken c and d are transposed matrices
        let d_simd = T::from_slice(&d.data[j][k..]);
        sum += b_simd * (c_simd + d_simd);
    }
    // println!("{:?} {:?}", sum, sum.reduce_sum());
    return sum.reduce_sum();
}

fn print_head_tail(matrix: Matrix) {
    println!("Head: {:?}", matrix.data[0]);
    println!("Tail: {:?}", matrix.data[matrix.shape.0 - 1]);
}

fn main() {
    let main_start = Instant::now();

    let matrix = Matrix::new(seed_matrix((2048, 2048)));
    // let matrix2 = Matrix::new(seed_matrix((16, 16)));
    // println!("{:?}", matrix.data);

    let start = Instant::now();
    // let a = matrix_multiplication(matrix.clone(), matrix.clone(), matrix);
    let a = matrix_multiplication_simd(matrix.clone(), matrix.clone(), matrix);
    // let a = matrix_multiplication_parallel(matrix.clone(), matrix.clone(), matrix);
    // let a = matrix_multiplication_parallel_simd(matrix.clone(), matrix.clone(), matrix); // best result for 2048x2048

    // let a = matrix_multiplication_simd(matrix2.clone(), matrix2.clone(), matrix2);
    // let a = matrix_multiplication_parallel_simd(matrix2.clone(), matrix2.clone(), matrix2);
    let duration = start.elapsed();

    let main_duration = main_start.elapsed();

    // print_head_tail(a);
    println!("Execution time (main fn): {:?}", main_duration);
    // println!("Execution time (matrix multiplication): {:?}", duration);

    // println!("usize max:\t{}\nu64 max:\t{}", usize::MAX, u64::MAX);
}
