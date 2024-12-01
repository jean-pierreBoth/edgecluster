//! module for dimension reduction  
//! gaussmat module implements multiplication by a random gaussian matrix G(n,m)  with n < m
//! romg module is based upon [skorski](https://proceedings.mlr.press/v134/skorski21a/skorski21a.pdf) and
//!  [mezzadri](https://arxiv.org/pdf/math-ph/0609050) articles
//!
//! They are equivalent if n << m (large reduction of dimension). But for moderate dimension reduction romg gives a better norm
//! conservation (see tests)
pub mod gaussmat;
pub mod reducer;
pub mod romg;
