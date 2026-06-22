//! blmrs — the Rust core for LBLM (flat open-addressing engine).
//!
//! Online logistic context mixing over byte-aware orders 0..4 with the same LOSSLESS integer context
//! keys as `mixfast.py`. The context tables are FIXED-SIZE flat arrays (open addressing with an 8-bit
//! checksum tag, evict-on-collision) -- bounded memory, no allocation/resizing, the native design the
//! scaling work pointed to. The exact algorithm was verified bit-for-bit against Python `mixfast`
//! using a HashMap (0.253666 == 0.253666); this flat engine is the same model with a small, measured
//! collision cost in exchange for bounded memory and native speed.
//!
//! Usage: blmrs <path> [byte_cap] [obits]   ->   whole-stream / last-20% bits/bit + timing.

use std::env;
use std::fs;
use std::time::Instant;

const ORDERS: [usize; 5] = [0, 1, 2, 3, 4];
const MULT: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline]
fn stretch(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

#[inline]
fn squash(t: f64) -> f64 {
    if t > 30.0 {
        1.0 - 1e-6
    } else if t < -30.0 {
        1e-6
    } else {
        1.0 / (1.0 + (-t).exp())
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("data/corpus.txt");
    let cap: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300_000);
    let obits: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);
    let osize: usize = 1usize << obits;
    let omask: u64 = (osize as u64) - 1;

    let mut raw = fs::read(path).expect("could not read input file");
    if cap > 0 && raw.len() > cap {
        raw.truncate(cap);
    }
    let n: usize = raw.len() * 8;
    let mut bits: Vec<u8> = Vec::with_capacity(n);
    for &byte in &raw {
        for j in (0..8).rev() {
            bits.push((byte >> j) & 1);
        }
    }

    let nm = ORDERS.len();
    let maxb = *ORDERS.iter().max().unwrap();
    let maskb: Vec<u64> = (0..=maxb)
        .map(|l| if 8 * l >= 64 { u64::MAX } else { (1u64 << (8 * l)) - 1 })
        .collect();
    let lr = 0.02f64;
    let delta = 0.2f64;

    // fixed-size flat tables: counts[order] holds (n0,n1) per slot; tags[order] is an 8-bit checksum.
    let mut counts: Vec<Vec<u32>> = (0..nm).map(|_| vec![0u32; 2 * osize]).collect();
    let mut tags: Vec<Vec<u8>> = (0..nm).map(|_| vec![0u8; osize]).collect();
    let mut w = [0.0f64; 5];
    let mut sts = [0.0f64; 5];
    let mut slots = [0usize; 5];

    let mut cur: u64 = 0;
    let mut phase: usize = 0;
    let mut htail: u64 = 0;
    let mut byte_pos: usize = 0;

    let split = (n as f64 * 0.8) as usize;
    let mut tot = 0.0f64;
    let mut tail = 0.0f64;
    let mut tailn: usize = 0;

    let t0 = Instant::now();
    for i in 0..n {
        for k in 0..nm {
            let b = ORDERS[k];
            let l = if byte_pos >= b { b } else { byte_pos };
            let run_ = ((htail & maskb[l]) << phase) | cur;
            let key = (((1u64 << (8 * l + phase)) | run_) << 3) | (phase as u64);
            let h = key.wrapping_mul(MULT);
            let ti = (h >> (64 - obits)) as usize;          // HIGH bits of a multiplicative hash mix well
            let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
            let _ = omask;
            if tags[k][ti] != want {
                tags[k][ti] = want;
                counts[k][2 * ti] = 0;
                counts[k][2 * ti + 1] = 0;
            }
            slots[k] = ti;
            let n0 = counts[k][2 * ti] as f64;
            let n1 = counts[k][2 * ti + 1] as f64;
            sts[k] = stretch((n1 + delta) / (n0 + n1 + 2.0 * delta));
        }
        let mut dot = 0.0f64;
        for k in 0..nm {
            dot += w[k] * sts[k];
        }
        let p = squash(dot);
        let y = bits[i];
        let cost = -(if y == 1 { p } else { 1.0 - p }).log2();
        tot += cost;
        if i >= split {
            tail += cost;
            tailn += 1;
        }
        let err = (y as f64) - p;
        for k in 0..nm {
            w[k] += lr * err * sts[k];
            counts[k][2 * slots[k] + y as usize] += 1;
        }
        cur = (cur << 1) | (y as u64);
        phase += 1;
        if phase == 8 {
            htail = ((htail << 8) | cur) & maskb[maxb];
            cur = 0;
            phase = 0;
            byte_pos += 1;
        }
    }
    let secs = t0.elapsed().as_secs_f64();

    let whole = tot / n as f64;
    let last = if tailn > 0 { tail / tailn as f64 } else { 0.0 };
    println!("corpus={}  bytes={}  bits={}  obits={}", path, raw.len(), n, obits);
    println!("  blmrs (flat)  whole-stream = {:.6}   last-20% = {:.6}  bits/bit   [{:.2}s, {:.1} Mbits/s]",
             whole, last, secs, (n as f64 / 1e6) / secs);
}
