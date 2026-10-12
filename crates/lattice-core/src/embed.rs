//! Geometric embedding of the lattice.
//!
//! Every node is mapped to a point in `R^DIM` via signed feature hashing of
//! its identifiers, ports, kind and code tokens (unigrams + bigrams). The
//! graph's position is the sum of its node vectors, so a delta has a direction
//! `E(T') - E(T)` that the Loop-Breaking Matrix can reason about. Hashing keeps
//! this fully offline, deterministic and fast.

use crate::graph::{LatticeGraph, Node};

pub const DIM: usize = 64;

pub type Vector = Vec<f32>;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.is_empty() {
            out.push(cur.to_lowercase());
            cur.clear();
        }
    };
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            cur.push(c);
        } else {
            flush(&mut cur, &mut out);
            if !c.is_whitespace() && "+-*/%<>=!&|[]{}().#".contains(c) {
                out.push(c.to_string());
            }
        }
    }
    flush(&mut cur, &mut out);
    out
}

fn add_feature(v: &mut [f32], feature: &str, weight: f32) {
    let h = fnv1a(feature.as_bytes());
    let idx = (h % DIM as u64) as usize;
    let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
    v[idx] += sign * weight;
}

pub fn embed_text(text: &str) -> Vector {
    let mut v = vec![0.0; DIM];
    let toks = tokens(text);
    for t in &toks {
        add_feature(&mut v, t, 1.0);
    }
    for w in toks.windows(2) {
        add_feature(&mut v, &format!("{} {}", w[0], w[1]), 0.5);
    }
    normalize(&mut v);
    v
}

pub fn embed_node(node: &Node) -> Vector {
    let mut v = vec![0.0; DIM];
    add_feature(&mut v, &format!("kind:{:?}", node.kind), 2.0);
    for t in tokens(&node.id) {
        add_feature(&mut v, &format!("id:{t}"), 1.5);
    }
    for p in &node.params {
        add_feature(&mut v, &format!("param:{p}"), 1.0);
    }
    let code = embed_text(&node.body);
    for (a, b) in v.iter_mut().zip(code) {
        *a += 3.0 * b;
    }
    if let Some(f) = &node.fallback {
        for (a, b) in v.iter_mut().zip(embed_text(f)) {
            *a += b;
        }
    }
    normalize(&mut v);
    v
}

pub fn embed_graph(g: &LatticeGraph) -> Vector {
    let mut v = vec![0.0; DIM];
    for n in g.nodes.values() {
        for (a, b) in v.iter_mut().zip(embed_node(n)) {
            *a += b;
        }
    }
    v
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

pub fn normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > 1e-9 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

pub fn sub(a: &[f32], b: &[f32]) -> Vector {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (na, nb) = (norm(a), norm(b));
    if na < 1e-9 || nb < 1e-9 {
        0.0
    } else {
        dot(a, b) / (na * nb)
    }
}

/// Component of `v` orthogonal to the unit vector `axis` (Gram–Schmidt step).
pub fn reject(v: &[f32], axis: &[f32]) -> Vector {
    let p = dot(v, axis);
    v.iter().zip(axis).map(|(x, a)| x - p * a).collect()
}

/// Project points onto their top-3 principal components (power iteration
/// with deflation). Output coordinates are scaled to roughly `[-1, 1]`.
pub fn pca3(points: &[Vector]) -> Vec<[f32; 3]> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = points[0].len();
    let mut mean = vec![0.0f32; dim];
    for p in points {
        for (m, x) in mean.iter_mut().zip(p) {
            *m += x / n as f32;
        }
    }
    let centered: Vec<Vector> = points.iter().map(|p| sub(p, &mean)).collect();

    let mut cov = vec![vec![0.0f32; dim]; dim];
    for p in &centered {
        for i in 0..dim {
            if p[i] == 0.0 {
                continue;
            }
            for j in 0..dim {
                cov[i][j] += p[i] * p[j];
            }
        }
    }

    let mut components: Vec<Vector> = Vec::new();
    for k in 0..3 {
        let mut v: Vector = (0..dim).map(|i| ((i * 7 + k * 13) % 11) as f32 - 5.0).collect();
        normalize(&mut v);
        for _ in 0..60 {
            let mut next: Vector = (0..dim).map(|i| dot(&cov[i], &v)).collect();
            for c in &components {
                next = reject(&next, c);
            }
            if norm(&next) < 1e-9 {
                break;
            }
            normalize(&mut next);
            v = next;
        }
        components.push(v);
    }

    let mut coords: Vec<[f32; 3]> = centered
        .iter()
        .map(|p| [dot(p, &components[0]), dot(p, &components[1]), dot(p, &components[2])])
        .collect();
    let max = coords
        .iter()
        .flat_map(|c| c.iter().map(|x| x.abs()))
        .fold(0.0f32, f32::max);
    if max > 1e-9 {
        for c in &mut coords {
            for x in c.iter_mut() {
                *x /= max;
            }
        }
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similar_code_is_closer_than_unrelated_code() {
        let a = embed_text("let total = 0; for x in items { total += x; } total");
        let b = embed_text("let sum = 0; for x in items { sum += x; } sum");
        let c = embed_text("text.split(\",\").map(|s| s.trim())");
        assert!(cosine(&a, &b) > cosine(&a, &c));
    }

    #[test]
    fn reject_is_orthogonal() {
        let mut axis = embed_text("for while loop");
        normalize(&mut axis);
        let v = embed_text("map filter reduce loop");
        assert!(dot(&reject(&v, &axis), &axis).abs() < 1e-5);
    }

    #[test]
    fn pca_returns_bounded_coords() {
        let pts: Vec<Vector> = ["a b c", "a b d", "x y z", "map filter"].iter().map(|t| embed_text(t)).collect();
        let coords = pca3(&pts);
        assert_eq!(coords.len(), 4);
        assert!(coords.iter().flatten().all(|x| x.abs() <= 1.0 + 1e-5));
    }
}
